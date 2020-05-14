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
        dataset.append([seq[ind-p:ind+q], 1])
        i = np.random.randint(p, seq_len-q)
        if i == ind:
            i = i + 1
        dataset.append([seq[i-p:i+q], 0])
          
    return dataset

def to_ind_mat(seq, word_mat=None):
    '''
    Function used to convert the acid alphabet sequence to an index matrix.
    - Param(s):
        seq : str, The sequence of alphabets.
        word_mat : np.ndarray, (26, d), The word-2-vec matrix. Optional, default is None.
    - Return:
        mat : numpy.ndarray, (n, d), The index matrix.
    '''
    mat = []
    if word_mat is None:
        word_mat = np.diag(np.ones(26))
    
    for alp in seq:
        ind = ord(alp.lower()) - 97
        vec = word_mat[ind]
        mat.append(vec)

    return np.array(mat)
    
    
def to_ind_vec(seq, word_mat=None):
    '''
    Function used to convert the acid alphabet sequence to an index vector.
    - Param(s):
        seq : str, The sequence of alphabets.
    - Return:
        vec : numpy.ndarray, (26n,), The index vector.
    '''
    return to_ind_mat(seq, word_mat).reshape(-1)

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


