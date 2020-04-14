'''
    The functions to be used in our project.
'''

import numpy as np


def gen_dataset(data_orig, p=13, q=2):
    '''
    Function used to generate the labeled dataset.
    - Param(s):
        data_orig : The origin data. Of format [[seq, clv_ind], ...].
        p : Number of acids before the clv site.
        q : Number of acids after the clv site.
    - Return:
        dataset : Of format [[seq, label]], seq is of length p+q.
    '''
    dataset = []
    for seq, ind in data_orig:
        seq_len = len(seq)
        for i in range(p, seq_len-q):
            seq_cut = seq[i-p:i+q]
            label = 1 if i==ind else 0
            dataset.append([seq_cut, label])
            
    return dataset

def gen_dataset_v2(data_orig, p=13, q=2):
    '''
    Function used to generate the labeled dataset.
    - Param(s):
        data_orig : The origin data. Of format [[seq, clv_ind], ...].
        p : Number of acids before the clv site.
        q : Number of acids after the clv site.
    - Return:
        dataset : Of format [[seq, label]], seq is of length p+q.
    '''
    dataset = []
    for seq, ind in data_orig:
        seq_len = len(seq)
        for i in range(ind, ind+3):
            seq_cut = seq[i-p:i+q]
            label = 1 if i==ind else 0
            dataset.append([seq_cut, label])
            
    return dataset

def to_ind_mat(seq):
    '''
    Function used to convert the acid alphabet sequence to an index matrix.
    - Param(s):
        seq : str, The sequence of alphabets.
    - Return:
        mat : numpy.ndarray, (n, 26), The index matrix.
    '''
    mat = []
    for alp in seq:
        ind = ord(alp.lower()) - 97
        vec = np.zeros(26)
        vec[ind] = 1
        mat.append(vec)
    
    return np.array(mat)
    
    
def to_ind_vec(seq):
    '''
    Function used to convert the acid alphabet sequence to an index vector.
    - Param(s):
        seq : str, The sequence of alphabets.
    - Return:
        vec : numpy.ndarray, (26n,), The index vector.
    '''
    return to_ind_mat(seq).reshape(-1)

def get_matrix(matrix_name):
    '''
    Function used to import a matrix used to calculate the similarity score.
    - Param(s):
        matrix_name : name of the file of matrix to use.
    - Return:
        mat : a 26 * 26 matrix.
    '''
    path = './Matrix/'
    path = path + matrix_name + '_'
    data_orig = np.loadtxt(path)
    return data_orig

