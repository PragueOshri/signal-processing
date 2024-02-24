import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
import scipy as sci


# create my own PCA
'''
This function takes data with more than 3 dimensions and 
reducing the number of dimensions to 3 using PCA calculation 
Input: Matrix of data where number of columns represents the number of dimensions
Output: 1. The data after projection
        2. The eigenvectors (the 3 components)
'''
def calculate_PCA(data_mat):
    # subtract mean from each dim
    shiftted_data = data_mat - np.mean(data_mat, 0)
    # calc the covariance matrix
    cov_mat = np.cov(shiftted_data.T)
    # extract the eigen values and the eigen vectors
    w, v = eig(cov_mat)
    # find the max 3
    highest_w_index = np.argsort(w)[-3:]
    eigvec = [v[:, highest_w_index[2]], v[:, highest_w_index[1]], v[:, highest_w_index[0]]]
    # calc the projection
    proj_data = np.dot(data_mat, np.array(eigvec).T)

    return proj_data, eigvec

'''
This is helper function to plot the eigenvector (components)
'''
def plot_coeff(coeff):

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(np.arange(0,20),coeff[0], 'o--', color='turquoise', label='component 1')
    ax.plot(np.arange(0,20),coeff[1], 'o--', color='teal', label='component 2')
    ax.plot(np.arange(0,20),coeff[2], 'o--', color='blue', label='component 3')
    ax.set_xticks(np.arange(0,20,1))
    ax.set_title('MY PCA: Coefficients for the first 3 Components')
    ax.set_xlabel('dimension (can also represent as time)')
    ax.set_ylabel('coefficients value')
    ax.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    pca_data = sci.io.loadmat('PCA and Clustering/dataPCA.mat')
    pca_data = pca_data['lfp']
    projection, coeff = calculate_PCA(pca_data)
    plot_coeff(coeff)