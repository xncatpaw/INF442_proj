'''
    The fonctions to train several svm model precised in different class.
'''

import numpy as np
import utils
import math
from sklearn import svm

HAS_TQDM = True
try:
    from tqdm import tqdm_notebook as tqdm
except ImportError:
    HAS_TQDM = False

    
    
class matrix_kernel:
    def __init__(self, p, q, matrix_name, dataset):
        '''
        Function init.
        - Param(s):
            p, q : neighborhood parametres
            matrix_name : name of the similarity matrix to use
            dataset : a sequence list created by function gen_data in utils.py
        '''
        self.p = p
        self.q = q
        self.lettre_list = 'ARNDCQEGHILKMFPSTWYVBZX*'
        self.matrix = utils.get_matrix(matrix_name)
        self.dataset = dataset
        
    def alpha_to_ind(x):
        x = x.upper()
        ind = LETTRE_LIST.find(x)
        if ind == -1:
            ind = 23
        return ind
    
    def S(self, a, b):
        '''
        Function used to calculate the score according to the given matrix.
        - Param(s):
            a : a sequence vector
            b : another sequence vector
            matrix : the matrix used to calculate the score
        - Return:
            score : double.
        '''
        score = 0
        assert (len(a) == len(b))
        for i in range(len(a)):
            score = score + self.matrix[alpha_to_ind(a[i])][alpha_to_ind(b[i])]
        return score

    def __call__(self, a, b):
        return self.S(a,b)

    
    
class prob_kernel:
    def __init__(self, p, q, dataset):
        self.dataset = dataset
        self.p = p
        self.q = q
        self.matrix_s = self.generate_matrix_s()
        
        self.debug=False

    def f(self, a, i):
        '''
        Function used to calculate the frequence of 'a' appeared at the position i in dataset.
        - Param(s):
            a : a caracter as symbol of an Amino acid
            i : Position to search in every sequence
            dataset : a sequence list created by function gen_data in utils.py
        - Return:
            double : the frequence of 'a' appeared at the position i in dataset.
        '''
        count = 0
        N = len(self.dataset)
        for k in range(N):
            if (self.dataset[k][0][i] == a):
                count = count + 1
        return count/N

    def g(self, a):
        '''
        Function used to calculate the observed general background frequency g(a) of amino acid a in the given set.
        - Param(s):
            a : a caracter as symbol of an Amino acid
            dataset : a sequence list created by function gen_data in utils.py
        - Return:
            double : the frequence of 'a' appeared over the whole length of given sequences.
        '''
        count = 0
        total_length = 0
        N = len(self.dataset)
        for k in range(N):
            count = count + self.dataset[k][0].count(a)
            total_length = total_length + len(self.dataset[k][0])
        return count/total_length

    def s(self,a, i):
        '''
        Function used to calculate the score of an Amino acid in one position.
        - Param(s):
            a : a caracter as symbol of an Amino acid
            i : Position to search in every sequence
            dataset : a sequence list created by function gen_data in utils.py
        - Return:
            double : score of an Amino acid 'a' in position i.
        '''
        return np.log10(self.f(a, i)) - np.log10(self.g(a))
    
    def score(self, w):
        '''
        Function used to calculate the score of a given sequence.
        - Param(s):
            w : given sequence of length (p+q)
            p : length of neighbors behind
            q : length of neighbors after
            a : a caracter as symbol of an Amino acid
            dataset : a sequence list created by function gen_data in utils.py
        - Return:
            double : the score of a given sequence w.
        '''
        sum = 0
        for i in range(self.p+self.q):
            sum = sum + s(w[i], i, self.dataset)
        return sum
    
    def generate_matrix_s(self):
        mat_s = np.zeros((26, self.p+self.q))
        for i in range(26):
            for j in range(self.p+self.q):
                mat_s[i,j] = self.s(chr(i+97).upper(),j)
        return mat_s

    def probability_log_kernel(self, a, b):
        '''
        Function used to calculate the log of a probability Kernel(a,b).
        - Param(s):
            a : an index vector, of shape(26n, ), or a list of index vectors, of shape (-1, 26n)
            b : an index vector, of shape(26n, ), or a list of index vectors, of shape (-1, 26n)
        - Return:
            log(K(a,b)) : double, or np.ndarray.
        '''
        n = self.p+self.q
        
        # Reshape a and b
        #a = np.reshape(a, (-1,n,26))
        #b = np.reshape(b, (-1,n,26))
        a = np.reshape(a, (n,26))
        b = np.reshape(b, (n,26))
        index_vec = np.arange(26)
        log_K = 0
        #assert(len(a) == len(b))
        for i in range(len(a)):
            a_ind = int(np.dot(index_vec, a[i]))
            b_ind = int(np.dot(index_vec, b[i]))
            #print(a_ind)
            if a_ind == b_ind:
                s_tmp = self.matrix_s[a_ind, i]
                log_K = log_K + s_tmp + np.log10(1+np.exp(s_tmp))
            else:
                log_K = log_K + self.matrix_s[a_ind, i] + self.matrix_s[b_ind, i]
            '''
            if (a[i]==b[i]):
                #a_ind = ord(a[i].lower())-97
                a_ind = 
                s_temp = self.matrix_s[a_ind,i]
                log_K = log_K + s_temp + np.log10(1+np.exp(s_temp))
            else:
                a_ind = ord(a[i].lower())-97
                b_ind = ord(b[i].lower())-97
                log_K = log_K + self.matrix_s[a_ind, i] + self.matrix_s[b_ind, i]
            '''
        return log_K
    
    
    def __call__(self, a, b):
        '''
        Used to calculate K(a,b)
        - Param(s):
            a, np.ndarray, of shape (26n, ) or of shape (-1, 26n)
            b, np.ndarray, of shape (26n, ) or of shape (-1, 26n)
        '''
        # Verify the shape.
        n = self.p+self.q
        a = np.array(a)
        b = np.array(b)
        if a.shape == (26*n, ):
            assert(a.shape == b.shape)
            log_K = self.probability_log_kernel(a, b)
            return log_K
        else :
            assert(a.shape[1]== 26*n and b.shape[1] == 26*n)
            len_a = a.shape[0]
            len_b = b.shape[0]
            print('Shape of kernel is (%d, %d)' % (len_a, len_b))
            kernel_mat = np.zeros((len_a, len_b))
            
            it_a = range(len_a)
            if self.debug and HAS_TQDM:
                it_a = tqdm(it_a)
            for i in it_a:
                for j in range(len_b):
                    kernel_mat[i,j] = self.probability_log_kernel(a[i], b[j])
                    
            return kernel_mat
            
        #log_K = self.probability_log_kernel(a, b)
        #return np.power(10, log_K)
        #return log_K
    
    
    
    
def train_model(data_orig, p, q, kernel='rbf'):
    '''
    Function used to train a svm model.
    - Param(s):
        data_orig : list, of formate [(seq, clv_ind), ...], the data used to train.
        p, 
        q,
        kernerl : str or callable, the kernel to be used in SVM.
                  Default is 'rbf', in which case rbf is used.
                  Can be callable, a function R^(p+q)*R^(p+q) -> R.
    - Return:
        model : a SVM model well trained.
    '''
    
    # Generate the dataset.
    dataset = utils.gen_dataset_v2(data_orig, p, q)
    # X and Y lists.
    X_list = [utils.to_ind_vec(data[0]) for data in dataset]
    Y_list = [data[1] for data in dataset]
    X_list = np.array(X_list)
    Y_list = np.array(Y_list)
    # Seperate train data from test data.
    msk = (np.random.rand(len(Y_list)) < 0.7)
    X_list_train = X_list[msk]
    Y_list_train = Y_list[msk]
    X_list_test = X_list[~msk]
    Y_list_test = Y_list[~msk]    
    # Define the model.
    model = svm.SVC(gamma='auto', kernel=kernel)
    # Train the model.
    model.fit(X_list_train, Y_list_train)
    # Test the model.
    Yp_list_train = model.predict(X_list_train)
    Yp_list_test = model.predict(X_list_test)
    Yp_list_train = np.array(Yp_list_train)
    Yp_list_test = np.array(Yp_list_test)
    err_train = np.sum(Y_list_train!=Yp_list_train) / len(Y_list_train)
    err_test = np.sum(Y_list_test!=Yp_list_test) / len(Y_list_test)
    print('The train err is %.3f.' % err_train)
    print('The test err is %.3f.' % err_test)
    
    return model

def model_evaluete():
    '''
    Function used to evaluate a svm model by its precision and callback value.
    - Param(s):
    - Return:
        double : type XXX.
    '''
    
    
    