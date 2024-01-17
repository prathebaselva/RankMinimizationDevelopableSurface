import numpy as np
import os
outfolder = 'plyfiles'


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

def writePLY(outfolder, fname, points, colors, values, outputfilename):
    with open(os.path.join(outfolder, outputfilename+'_'+str(fname)+'_color.ply'), 'w') as outfile:
        outfile.write('''ply
        format ascii 1.0
        comment VCGLIB generated
        element vertex '''+str(len(points))+'''
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
        print(Curvature, flush=True)
        #green = Color("green")
        #colors = list(green.range_to(Color("red"),2))
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

def getSignColors(Curvature):
        #hcolors = trimesh.visual.interpolate(np.log10(np.abs(1+Curvature)),color_map='viridis')
        hcolors = np.zeros((len(Curvature),4))
        v1 = np.where(Curvature > 0)[0]
        hcolors[v1] = np.array([255,0,0,255])
        v2 = np.where(Curvature < 0)[0]
        hcolors[v2] = np.array([0,255,0,255])
        v3 = np.where(Curvature ==0)[0]
        hcolors[v3] = np.array([0,0,255,255])
        return hcolors

def getLinearInterpolationColor(values):
    from matplotlib.colors import to_rgb
    red = np.array(to_rgb('red'))
    green = np.array(to_rgb('green'))
    blue = np.array([0.0, 0.0, 1.0])
    white = np.array([1.0, 1.0, 1.0])

    hcolors = []
    print(np.max(values))
    print(np.min(values))

    for v in values:
        #hcolor = 255.0*(red*v+green*(1-v))
        hcolor = 255.0*(blue*v + white*(1-v))
        hcolors.append(hcolor)

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


def getVarianceColorcoded(points, variance, outputfilename):
        print("numpoint = ",len(points))
        fname = "var"
        #print(np.max(variance))
        #print(np.min(variance))
        dcolors = getCurvatureColors(variance)
        varabs = np.abs(variance)
        print(np.max(varabs))
        print(np.min(varabs))
        highvalindex = np.where(varabs >= 1)[0]
        varabs[highvalindex] = 1
        lowvalindex = np.where(varabs <= 1e-7)[0]
        varabs[lowvalindex] = 0
        varnorm = (varabs - np.min(varabs))/(np.max(varabs) - np.min(varabs))
        dcolors = getLinearInterpolationColor(varnorm)
        writePLY(outfolder, fname, points, dcolors, variance, outputfilename)

def getDeterminantColorcoded(points, determinant, outputfilename):
        print("numpoint = ",len(points))
        fname = "det"
        detabs = np.abs(determinant)
        print(np.max(detabs))
        print(np.min(detabs))
        highvalindex = np.where(detabs >= 10)[0]
        detabs[highvalindex] = 10
        lowvalindex = np.where(detabs <= 1e-7)[0]
        detabs[lowvalindex] = 0
        detnorm = (detabs - np.min(detabs))/(np.max(detabs) - np.min(detabs))
        dcolors = getLinearInterpolationColor(detnorm)
        #dcolors = getCurvatureColors(determinant)
        #print(determinant)
        writePLY(outfolder, fname, points, dcolors, determinant, outputfilename)
        scolors = getSignColors(determinant)
        fname = "detsign"
        writePLY(outfolder, fname, points, scolors, determinant, outputfilename)

def getGaussColorcoded(points, gauss, outputfilename):
        print("numpoint = ",len(points))
        fname = "gauss"
        dcolors = getCurvatureColors(gauss)
        writePLY(outfolder, fname, points, dcolors, gauss, outputfilename)

