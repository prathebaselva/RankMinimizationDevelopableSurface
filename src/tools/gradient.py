import torch

deviceids = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        deviceids.append(i)
    torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ids = ", deviceids)

def getGradient(predicted_sdf, batch_sample_sampled_points):
    batch_gradient_outputs = torch.ones_like(predicted_sdf).to(device) 
    gradient_predicted_sdf, = torch.autograd.grad(outputs=predicted_sdf, inputs=[batch_sample_sampled_points], grad_outputs=batch_gradient_outputs, retain_graph=True, create_graph=True)    
    return gradient_predicted_sdf

def getHessian(model, batch_sample_sampled_points):
    bs = batch_sample_sampled_points.shape[0]
    hess = torch.zeros(bs, 3,3)
    for i in range(bs):
        hess[i] = torch.autograd.functional.hessian(model, batch_sample_sampled_points[i])
        if i == 0 or i == bs-1:
            print(hess[i])
    return hess

def getGradientAndHessian(predicted_sdf, batch_sample_sampled_points):
    batch_gradient_outputs = torch.ones_like(predicted_sdf).to(device) 
    gradient_predicted_sdf, = torch.autograd.grad(outputs=predicted_sdf, inputs=[batch_sample_sampled_points], grad_outputs=batch_gradient_outputs, retain_graph=True, create_graph=True)    
    
    hessian_matrix = calculateGradient(gradient_predicted_sdf, batch_sample_sampled_points, True)
    return gradient_predicted_sdf, hessian_matrix


def calculateGradient(outputs, inputs,create_graph=False):
    jac = []
    transpose_outputs = outputs.transpose(1,0)
    for i in range(3):
        if transpose_outputs.shape[1] <= 1:
            batch_gradient_outputs = torch.ones(transpose_outputs[i].size()).to(device) 
        else:
            batch_gradient_outputs = torch.ones(transpose_outputs[i].squeeze().size()).to(device) 
        grad_x, = torch.autograd.grad(outputs=transpose_outputs[i], inputs=inputs, grad_outputs=batch_gradient_outputs, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x)
    jac = torch.stack(jac)
    jac = torch.stack([jac[:,i] for i in range(len(outputs))])
    return jac

##########################################
def jacobian(y, x, create_graph=False):
  with torch.autograd.set_detect_anomaly(True):    
    jac = []
    #print(y.size())
    flat_y = y.reshape(-1) # torch.flatten(y)
    #print(flat_y.size())
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0
    #print(y.shape + x.shape)
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y,x):
    return jacobian(jacobian(y,x,create_graph=True),x)


def jacobian_grad(y, x, create_graph=False):
    jac = []
    flat_y = torch.flatten(y)
    #print(flat_y.size())
    grad_y = torch.zeros_like(flat_y)
    batch_size = len(x)
    index = -1
    for i in range(len(flat_y)):
        grad_y[i] = 1
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        print(grad_x)
        if i % 3 == 0:
            index += 1
        jac.append(grad_x[index])#.reshape(x.shape))
        grad_y[i] = 0
    #print(jac)
    #print(torch.stack(jac).reshape(batch_size, 3,3))
    return torch.stack(jac).reshape(batch_size, 3,3)
