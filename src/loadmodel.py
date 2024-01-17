#import open3d
import torch.backends.cudnn as cudnn
from model_all import Model 
from model import SDF
#form model_relu import Decoder
from utils import * # normalize_pts, normalize_normals, SdfDataset, mkdir_p, isdir
from gradient import *
from train import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)
   

def train(dataset,  model, optimizer, epoch, args, mcube_points=None, mcube_sdf=None):
    model.train()  # switch to train mode
    loss, indexcount, gaussavg = runmodel(dataset,  model, optimizer, epoch, args, mcube_points, mcube_sdf)
    return loss, indexcount, gaussavg

# validation function
def val(dataset,  model, optimizer, epoch, args, mcube_points=None, mcube_sdf=None):
    model.eval()  # switch to test mode
    loss, indexcount, gaussavg = runmodel(dataset,  model, optimizer, epoch, args, mcube_points, mcube_sdf, False)
    return loss, indexcount, gaussavg

# testing function


def loadEvaluateModelHelper(model, use_model, checkpoint_folder):
    if use_model == "best":
        print("\nloading training pretrained checkpoint"+use_model)
        path_to_resume_file = os.path.join(checkpoint_folder, use_model+'.pth.tar')
        #if not os.path.exists(path_to_resume_file):
        #    path_to_resume_file = os.path.join(args.checkpoint_folder, 'model_best_pth.tar')
    elif use_model == "last":
        print("\nloading training pretrained checkpoint model last")
        path_to_resume_file = os.path.join(checkpoint_folder, 'model_last.pth.tar')
        if not os.path.exists(path_to_resume_file):
            path_to_resume_file = os.path.join(checkpoint_folder, 'model_last_pth.tar')
    else:
        print("\nloading training pretrained checkpoint model given")
        path_to_resume_file = os.path.join(checkpoint_folder, 'model_best_'+use_model+'.pth.tar')
        print("path to model = ", path_to_resume_file, flush=True)
     
    if not os.path.exists(path_to_resume_file):
        print("path does not exists")
        return None,None
    print("=> Loading training checkpoint '{}'".format(path_to_resume_file),flush=True)
    checkpoint = torch.load(path_to_resume_file)
    model.load_state_dict(checkpoint['state_dict'])
    if "train_latent" in checkpoint:
        latent = checkpoint['train_latent']
    elif "test_latent" in checkpoint:
        latent = checkpoint['test_latent']
    else:
        latent = None
    model.to(device)
    return model, latent

def loadPretrainedModelHelper(model, args):
    pretrained_model_folder = args.pretrained_model_folder
    if args.use_model == "best":
        print("\nUsing best pretr ained model")
        path_to_resume_file = os.path.join(pretrained_model_folder, 'model_best.pth.tar')
        if not os.path.exists(path_to_resume_file):
            path_to_resume_file = os.path.join(pretrained_model_folder, 'model_best_pth.tar')
    elif args.use_model == "last":
        print("\nUsing last pretrained model") 
        path_to_resume_file = os.path.join(pretrained_model_folder, 'model_last.pth.tar')
    print("=> Loading pretrained checkpoint '{}'".format(path_to_resume_file))
    if os.path.exists(path_to_resume_file):
        checkpoint = torch.load(path_to_resume_file)
        model.load_state_dict(checkpoint['state_dict'])# strict=False)
        if "train_latent" in checkpoint:
            latent = checkpoint['train_latent']
        else:
            latent = None
        print("loaded pretrained model ", path_to_resume_file)
        return model, latent
    else:
        print("no pretrained model exists ")
        return None, None

def loadCheckpointModelHelper(model, use_model, checkpoint_folder):
    
    if use_model == "best": 
        print("\nloading training  checkpoint model best")
        path_to_resume_file = os.path.join(checkpoint_folder, 'model_best.pth.tar')
        if not os.path.exists(path_to_resume_file):
            path_to_resume_file = os.path.join(checkpoint_folder, 'model_best_pth.tar')
    elif use_model == "last":
        print("\nloading training checkpoint model last")
        path_to_resume_file = os.path.join(checkpoint_folder, 'model_last.pth.tar')
    else:
        print("\nloading training checkpoint model last")
        path_to_resume_file = os.path.join(checkpoint_folder, 'model_best_'+use_model+'.pth.tar')

    if os.path.exists(path_to_resume_file):
        print("=> Loading training checkpoint '{}'".format(path_to_resume_file))
        checkpoint = torch.load(path_to_resume_file)
        model.load_state_dict(checkpoint['state_dict'])
        if 'train_latent' in checkpoint:
            latent = checkpoint['train_latent']
        elif 'test_latent' in checkpoint:
            latent = checkpoint['test_latent']
        best_loss = checkpoint['best_loss'] 
        #args.start_epoch = checkpoint['epoch']
        #args.lr = checkpoint['lr']
        return model, latent, best_loss
    else:
        print( "No training checkpoint model exists")
        return None, None, None

def loadEvaluateModel(model,use_model, args, checkpoint_folder = None):

    if model is None:
        model = initModel(args)

    if checkpoint_folder is None:
        checkpoint_folder = args.checkpoint_folder

    print("use_model =", use_model, flush=True)
    
    return loadEvaluateModelHelper(model, use_model, checkpoint_folder)


def loadPretrainedModel(model,args):
    if not args.use_pretrained_model: 
        print("Use pretrained model flag not set")
        return None 

    if model is None:
        model = initModel(args)
    #if args.use_checkpoint_model:
    #    args.use_model='last'
    #    model, latent = loadCheckpointModelHelper(model, 'last', args.checkpoint_folder)
    #    if not (model is None):
    #        return model, latent
    #model = initModel(args)
    model, latent = loadPretrainedModelHelper(model,args)
    return model, latent, 2e10, args


def loadCheckpointModel(model, args):
    if not args.use_checkpoint_model: 
        print("Use continous checkpoint flag not set")
        return None

    if model is None:
        model = initModel(args)
    print(args.use_model)
    return loadCheckpointModelHelper(model, args.use_model, args.checkpoint_folder)

def initModel(args):
        withomega_0 = True
        if not args.withomega:
            withomega_0 = False

        if args.activation == 'sine':
            if args.model == 'simple5hnd256lat2':
                model = SDF(channels=[256,256,256,256,256],
                        latent_in=[2],
                        norm=args.normalization,
                        omega=args.omega,
                        dropout=None)

            if args.model == 'simple3hnd256sine':
                model = SDF(channels=[256,256,256],
                        latent_in=[1],
                        norm=args.normalization,
                        omega=args.omega,
                        dropout=None)
            if args.model == 'simplesine5hnd256lat2':
                model = SDF(channels=[256,256,256,256,256],
                        latent_in=[2],
                        norm=args.normalization,
                        omega=args.omega,
                        dropout=None)
            if args.model == 'simple5hnd256sine':
                model = SDF(channels=[256,256,256,256,256],
                        latent_in=[2],
                        norm=args.normalization,
                        omega=args.omega,
                        dropout=None)
            if args.model == 'simplesine8hnd256lat2':
                model = SDF(channels=[256,256,256,256,256,256,256,256],
                        latent_in=[4],
                        norm=args.normalization,
                        omega=args.omega,
                        dropout=None)
            if args.model == 'simple8hnd512sine':
                model = SDF(channels=[512,512,512,512,512,512,512,512],
                        latent_in=[4],
                        norm=args.normalization,
                        omega=args.omega,
                        dropout=None)
        else:
            if args.model == 'simple8h512lat2':
                model = Model(channels=[512,512,512,512,512,512,512,512],
                        latent_in=[4],
                        omega=args.omega,
                        withomega= withomega_0,
                        dropout_prob=0.2,
                        latcode=args.lat,
                        norm=args.normalization,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=[0,1,2,3,4,5,6,7])
            if args.model == 'simple8h512latall':
                model = Model(channels=[512,512,512,512,512,512,512,512],
                        latent_in=[1,2,3,4,5,6,7,8],
                        omega=args.omega,
                        withomega= withomega_0,
                        dropout_prob=0.2,
                        latcode=args.lat,
                        norm=args.normalization,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=[0,1,2,3,4,5,6,7])
            if args.model == 'simple8hnd512':
                model = Model(channels=[512,512,512,512,512,512,512,512],
                        latent_in=[4],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=True,
                        dropout=None)
            if args.model == 'simple3h256':
                model = Model(channels=[256,256,256],
                        latent_in=[1],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        dropout_prob=0.2,
                        activationafternorm=False,
                        activation=args.activation,
                        dropout=[0,1,2])
            if args.model == 'simple5h256latall':
                model = Model(channels=[256,256,256,256,256],
                        latent_in=[1,2,3,4,5],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        dropout_prob=0.2,
                        latcode=args.lat,
                        activationafternorm=False,
                        activation=args.activation,
                        dropout=[0,1,2,3,4])
            if args.model == 'simple5h256lat2':
                model = Model(channels=[256,256,256,256,256],
                        latent_in=[2],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        dropout_prob=0.2,
                        latcode=args.lat,
                        activationafternorm=False,
                        activation=args.activation,
                        dropout=[0,1,2,3,4])
            if args.model == 'simple5h256':
                model = Model(channels=[256,256,256,256,256],
                        latent_in=[2], 
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        dropout_prob=0.2,
                        latcode=args.lat,
                        activationafternorm=False,
                        activation=args.activation,
                        dropout=[0,1,2,3,4])
            if args.model == 'simple8hnd256lat0':
                model = Model(channels=[256,256,256,256,256,256,256,256],
                        latent_in=[],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=None)
            if args.model == 'simple8hnd256lat2':
                model = Model(channels=[256,256,256,256,256,256,256,256],
                        latent_in=[4],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=None)
            if args.model == 'simple8h512':
                model = Model(channels=[512,512,512,512,512,512,512,512],
                        latent_in=[4],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=[0,1,2,3,4,5,6,7])
            if args.model == 'simple8h256lat0':
                model = Model(channels=[256,256,256,256,256,256,256,256],
                        latent_in=[],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        dropout_prob=0.2,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=[0,1,2,3,4,5,6,7])
            if args.model == 'simple8h256lat2':
                model = Model(channels=[256,256,256,256,256,256,256,256],
                        latent_in=[4],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        dropout_prob=0.2,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=[0,1,2,3,4,5,6,7])
            if args.model == 'simple3hnd256latall':
                model = Model(channels=[256,256,256],
                        latent_in=[1,2,3],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=None)
            if args.model == 'simple3hnd256lat2':
                model = Model(channels=[256,256,256],
                        latent_in=[2],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=None)
            if args.model == 'simple5hnd256lat0':
                model = Model(channels=[256,256,256,256,256],
                        latent_in=[],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=None)
            if args.model == 'simple5hnd256lat2':
                model = Model(channels=[256,256,256,256,256],
                        latent_in=[2],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=None)
            if args.model == 'simple5hnd256latall':
                model = Model(channels=[256,256,256,256,256],
                        latent_in=[1,2,3,4,5],
                        norm=args.normalization,
                        withomega= withomega_0,
                        omega=args.omega,
                        activation=args.activation,
                        activationafternorm=False,
                        dropout=None)

        cudnn.benchmark = True
        return model

