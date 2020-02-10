import numpy as np
from generate_plots import cluster_plot
from generate_video import generate_video
from sklearn.decomposition import PCA

'''
    Jeremy Herman
    
    Clustering via k-means

    This file contains:
        - myKMeans function
        - function utilizing PCA to reduce dimensionality
        - function to get (random) initial reference vectors
'''

# Saves a video where each frame is an iteration of k-means using mean cluster vectors for reference.
def myKMeans(data, k):
    # Check if we need to reduce the dimensionality
    if np.shape(data)[1] > 3:
        data = reduce_dimensionality(data, 3)
    
    # Get indices of the reference vectors
    reference_indices = get_reference_vector_inds(np.shape(data)[0], k)

    observations = []
    for i in range(np.shape(data)[0]):
        dists = []
        for rp in reference_indices:
            dists.append(np.linalg.norm(data[i] - data[rp]))
        observations.append(dists)
    
    observations = np.array(observations)

    ref_vec_1 = data[reference_indices[0]]
    ref_vec_2 = data[reference_indices[1]]

    count = 0
    check = 1
    bound = 2**(-23)
    
    while check > bound:
        count += 1
        name = 'Iteration ' + str(count)
        save_path = './plots/iteration_' + str(count) + '.png'
        ref_vec_1, ref_vec_2, check = cluster_plot(data, observations, ref_vec_1, ref_vec_2, name, save_path, k)
        observations = []
        for i in range(np.shape(data)[0]):
            dists = []
            for rp in [ref_vec_1, ref_vec_2]:
                dists.append(np.linalg.norm(data[i] - rp))
            observations.append(dists)
        observations = np.array(observations)

    generate_video('./plots/', './K_2_F_all.avi')


# Use sklearn's PCA to reduce the dimensionality of a matrix to a specified size
def reduce_dimensionality(data, size):
    pca = PCA(n_components = size)
    return pca.fit_transform(data)

# Read in the number of indices, create a list of numbers, shuffle the list, return first k indices
def get_reference_vector_inds(inds, k):
    l = []
    for i in range(inds):
        l.append(i)
    np.random.shuffle(l)
    return l[:k]