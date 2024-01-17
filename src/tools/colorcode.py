import numpy as np
import os


def writeGaussPLYPts(outfolder, fname, points, values, outputfilename):
    fwritepts = open(os.path.join(outfolder, outputfilename+'_'+str(fname)+'.pts'), 'w')
    with open(os.path.join(outfolder, outputfilename+'_'+str(fname)+'.ply'), 'w') as outfile:
        outfile.write('''ply
        format ascii 1.0
        comment VCGLIB generated
        element vertex '''+str(len(points))+'''
        property float x
        property float y
        property float z
        property float '''+fname+'''
        element face 0
        property list uchar int vertex_indices
        end_header\n''')
        for p,v in zip(points,values):
            outfile.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " "+str(v)+"\n" )
            fwritepts.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " "+str(v)+"\n" )
        outfile.close()
    fwritepts.close()

def writePLY(outfolder, points, colors, values, fname, outputfilename):
    numpoints = min(len(points), len(colors))
    with open(os.path.join(outfolder, outputfilename+'.ply'), 'w') as outfile:
        outfile.write('''ply
        format ascii 1.0
        comment VCGLIB generated
        element vertex '''+str(numpoints)+'''
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property float '''+fname+'''
        element face 0
        property list uchar int vertex_indices
        end_header\n''')
        for p,c,v in zip(points,colors, values):
            outfile.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " "+str(int(c[0]))+" "+str(int(c[1]))+" "+str(int(c[2]))+" "+str(v)+"\n" )
        outfile.close()

def getSDFColors(IF):
        dcolors = np.zeros((len(IF),4))
        #trimesh.visual.interpolate(np.log10(np.abs(1+IF)), color_map='viridis')
        IF_clone = IF.copy()
        IF = np.abs(IF)    
        v1 = np.where(IF >= 1e-2)[0]  
        dcolors[v1] = np.array([255,0,0,255])
        v2 = np.where((IF > 1e-4)&(IF < 1e-2))[0]  
        dcolors[v2] = np.array([0,255,0,255])
        v3 = np.where((IF > 1e-6)&(IF < 1e-4))[0]  
        dcolors[v3] = np.array([0,0,255,255])
        v4 = np.where((IF > 1e-8)&(IF < 1e-6))[0]  
        dcolors[v4] = np.array([255,255,0,255])
        v5 = np.where((IF > 1e-10)&(IF < 1e-8))[0]  
        dcolors[v5] = np.array([255,0,255,255])
        v6 = np.where((IF < 1e-10))[0]  
        dcolors[v6] = np.array([0,255,255,255])
        return dcolors


def getGradientColors(gradient):
        #hcolors = trimesh.visual.interpolate(np.log10(np.abs(1+Curvature)),color_map='viridis')
        hcolors = np.zeros((len(gradient),4))
        v1 = np.where(np.abs(gradient) >= 1)[0]
        hcolors[v1] = np.array([255,0,0,255])
        v2 = np.where((np.abs(Curvature) < 1) & (np.abs(Curvature) >= 1e-1))[0]
        hcolors[v2] = np.array([0,255,0,255])
        v3 = np.where((np.abs(Curvature) < 1e-1) & (np.abs(Curvature) >= 1e-2))[0]
        hcolors[v3] = np.array([0,0,255,255])
        v4 = np.where((np.abs(Curvature) < 1e-2) & (np.abs(Curvature) >= 1e-3))[0]
        hcolors[v4] = np.array([255,0,255,255])
        v5 = np.where((np.abs(Curvature) < 1e-3) & (np.abs(Curvature) > 1e-4))[0]
        hcolors[v5] = np.array([255,255,0,255])
        v6 = np.where((np.abs(Curvature) <= 1e-4))[0]
        hcolors[v6] = np.array([255,255,255,100])
        return hcolors

def getCurvatureColors(Curvature):
        #hcolors = trimesh.visual.interpolate(np.log10(np.abs(1+Curvature)),color_map='viridis')
        hcolors = np.zeros((len(Curvature),4))
        v1 = np.where(np.abs(Curvature) >= 1000)[0]
        hcolors[v1] = np.array([255,0,0,255])
        v2 = np.where((np.abs(Curvature) < 1000) & (np.abs(Curvature) >= 100))[0]
        hcolors[v2] = np.array([0,255,0,255])
        v3 = np.where((np.abs(Curvature) < 100) & (np.abs(Curvature) >= 10))[0]
        hcolors[v3] = np.array([0,0,255,255])
        v4 = np.where((np.abs(Curvature) < 10) & (np.abs(Curvature) >= 1))[0]
        hcolors[v4] = np.array([255,0,255,255])
        v5 = np.where((np.abs(Curvature) < 1) & (np.abs(Curvature) > 0))[0]
        hcolors[v5] = np.array([255,255,0,255])
        v6 = np.where((np.abs(Curvature) == 0))[0]
        hcolors[v6] = np.array([255,255,255,100])
        return hcolors

def getCurvatureColors2(Curvature):
        #hcolors = trimesh.visual.interpolate(np.log10(np.abs(1+Curvature)),color_map='viridis')
        hcolors = np.zeros((len(Curvature),4))
        v1 = np.where(np.abs(Curvature) >= 1000)[0]
        hcolors[v1] = np.array([255,0,0,255])
        v2 = np.where((np.abs(Curvature) < 1000) & (np.abs(Curvature) >= 100))[0]
        hcolors[v2] = np.array([0,255,0,255])
        v3 = np.where((np.abs(Curvature) < 100) & (np.abs(Curvature) >= 10))[0]
        hcolors[v3] = np.array([0,0,255,255])
        v4 = np.where((np.abs(Curvature) < 10) & (np.abs(Curvature) >= 1))[0]
        hcolors[v4] = np.array([255,0,255,255])
        v5 = np.where((np.abs(Curvature) < 1) & (np.abs(Curvature) > 0))[0]
        hcolors[v5] = np.array([255,255,0,255])
        v6 = np.where((np.abs(Curvature) == 0))[0]
        hcolors[v6] = np.array([255,255,255,100])
        return hcolors

def getCurvatureColors1(Curvature):
        #hcolors = trimesh.visual.interpolate(np.log10(np.abs(1+Curvature)),color_map='viridis')
        hcolors = np.zeros((len(Curvature),4))
        v1 = np.where(np.abs(Curvature) >= 10)[0]
        hcolors[v1] = np.array([255,0,0,255])
        #v2 = np.where((np.abs(Curvature) < 10) & (np.abs(Curvature) >= 1))[0]
        #hcolors[v2] = np.array([0,255,0,255])
        #v3 = np.where((np.abs(Curvature) < 1) & (np.abs(Curvature) > 0))[0]
        #hcolors[v3] = np.array([0,0,255,255])
        v6 = np.where((np.abs(Curvature) < 10))[0]
        hcolors[v6] = np.array([255,255,255,100])
        return hcolors

def getCurvatureColors3(Curvature, threshold):
        #hcolors = trimesh.visual.interpolate(np.log10(np.abs(1+Curvature)),color_map='viridis')
        hcolors = np.zeros((len(Curvature),4))
        v1 = np.where(np.abs(Curvature) >= threshold)[0]
        hcolors[v1] = np.array([255,0,0,255])
        #v2 = np.where((np.abs(Curvature) < 10) & (np.abs(Curvature) >= 1))[0]
        #hcolors[v2] = np.array([0,255,0,255])
        #v3 = np.where((np.abs(Curvature) < 1) & (np.abs(Curvature) > 0))[0]
        #hcolors[v3] = np.array([0,0,255,255])
        v6 = np.where((np.abs(Curvature) < threshold))[0]
        hcolors[v6] = np.array([255,255,255,255])
        return hcolors
   
def getSDFandGaussColorcoded(points, IF, SVD, gaussCurvature, meanCurvature, outputfilename):
        print("numpoint = ",len(points))
        print("minIF = {} and maxIF = {}".format(min(IF), max(IF)))
        print("minSVD = {} and maxSVD = {}".format(min(SVD), max(SVD)))
        print("mingauss = {} and maxgauss = {}".format(min(gaussCurvature), max(gaussCurvature)))

        numneg = (np.where(IF < 0)[0])
        numpos = (np.where(IF > 0)[0])
        numzero = (np.where(IF == 0)[0])
        print("IF numpos {} , numneg = {}, numzero = {}".format(len(numpos), len(numneg), len(numzero)))

        fname = "IF"
        dcolors = getSDFColors(IF)
        writePLY(outfolder, fname, points,dcolors, IF, outputfilename)

        if args.gauss:
            hcolors = getCurvatureColors(SVD)
            fname = "SVD"
            writePLY(outfolder, fname, points, hcolors, SVD, outputfilename)

            hcolors = getCurvatureColors(gaussCurvature)
            fname = "gauss"
            writePLY(outfolder, fname, points, hcolors, gaussCurvature, outputfilename)

            hcolors = getCurvatureColors(meanCurvature)
            fname = "mean"
            writePLY(outfolder, fname, points, hcolors, meanCurvature, outputfilename)

def getGaussColorcoded(points, gaussCurvature, threshold, fname, outputfilename):
        print("numpoint = ",len(points))
        print("mingauss = {} and maxgauss = {}".format(min(gaussCurvature), max(gaussCurvature)))

        hcolors = getCurvatureColors3(gaussCurvature, threshold)
        #hcolors = getCurvatureColors2(gaussCurvature)
        #print(points)
        #print(hcolors)
        print(len(hcolors))
        print(len(points))
        outfolder = ''
        writePLY(outfolder, points, hcolors, gaussCurvature, fname, outputfilename)
