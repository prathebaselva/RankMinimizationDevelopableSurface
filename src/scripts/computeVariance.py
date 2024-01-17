import numpy as np
import os
from scipy.spatial import KDTree

def computeVariance(samplefiles, determinantfiles, outfolder, outfilename, k=20):
    points = np.load(samplefiles)
    determinant = np.load(determinantfiles)
    tree = KDTree(points)
    dd, index = tree.query(points, k=k+1)
    allvar = []
    allneigh = []
    allneighdist = []

    for i in range(len(points)):
        neighindex = index[i]
        neighdist = dd[i]
        neighdeterminant = determinant[neighindex]
        #print(neighdeterminant) 
        var = np.var(neighdeterminant)
        allvar.append(var)
        allneigh.append(neighindex)
        allneighdist.append(neighdist)

    np.save(os.path.join(outfolder,outfilename+'_var_'+str(k)+'.npy'), allvar)
    np.save(os.path.join(outfolder,outfilename+'_dist_'+str(k)+'.npy'), allneighdist)
    np.save(os.path.join(outfolder,outfilename+'_index_'+str(k)+'.npy'), allneigh)


if __name__ == '__main__':
    #computeVariance('../../npyfiles/dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_imppoints.npy','../../npyfiles/dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test_impdet_hist.npy', '../../npyfiles', 'dragon_25k_gaussthin_100_7_60_gelu_lr1e4_dn0_notanh_test', k=20)
    computeVariance('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impdet_hist.npy', '../../npyfiles', 'dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test', k=10)
    computeVariance('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impdet_hist.npy', '../../npyfiles', 'dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test', k=50)
    computeVariance('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test_impdet_hist.npy', '../../npyfiles', 'dragon_gelu_lr1e4_hlr1e-5_dn0_dh1e1_hessiandethat_notanh_test', k=100)
    #computeVariance('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test_impdet_hist.npy', '../../npyfiles', 'dragon_gelu_lr1e4_hlr1e-5_dn0_dh1_hessiandethat_notanh_test', k=20)
    #computeVariance('../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test_imppoints.npy','../../npyfiles/dragon_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test_impdet_hist.npy', '../../npyfiles', 'dragon_gelu_lr1e4_hlr1e-5_dn0_dh5e-1_hessiandethat_notanh_test', k=20)
