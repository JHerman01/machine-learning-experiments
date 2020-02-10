import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

'''
    Jeremy Herman
    
    Clustering via k-means

    This function generates a cluster plot for myKMeans and returns:
        - new reference vectors
        - a value to check against 2^-23 
'''
def cluster_plot(data, observations, ref_vec_1, ref_vec_2, name, save_path, k):
    fig = plt.figure()
    fig.suptitle(name)
    ax = plt.axes(projection='3d')
    cluster1 = []
    cluster2 = []
    for i in range(np.shape(data)[0]):
        if observations[i, 0] < observations[i, 1]:
            cluster1.append(data[i])
            ax.scatter3D(
                data[i, 0],
                data[i, 1],
                data[i, 2],
                marker='x',
                color=(0, 0, 1, 0.25)
            )
        else:
            cluster2.append(data[i])
            ax.scatter3D(
                data[i, 0],
                data[i, 1],
                data[i, 2],
                marker='x',
                color=(1, 0, 0, 0.25)
            )
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)

    c1_mean = np.mean(cluster1[:,:], 0)
    c2_mean = np.mean(cluster2[:,:], 0)
    
    ax.scatter3D(
        ref_vec_1[0],
        ref_vec_1[1],
        ref_vec_1[2],
        marker='o', color='blue', s=80, zorder=1)
    ax.scatter3D(
        ref_vec_2[0],
        ref_vec_2[1],
        ref_vec_2[2],
        marker='o', color='red', s=80, zorder=1)

    plt.savefig(save_path)

    check = 0
    for old_ref, new_ref in zip([ref_vec_1, ref_vec_2], [c1_mean, c2_mean]):
        for i in range(len(old_ref)):
            check += np.abs(old_ref[i] - new_ref[i])
    
    return [c1_mean, c2_mean, check]
