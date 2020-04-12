'''
    The functions to be used in our project.
'''


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