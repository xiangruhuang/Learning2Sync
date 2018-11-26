import numpy as np 

def transform(R, X):
    '''
    # Apply the transformation to X, which if not
    # will be converted to homogenous coordinates.
    # R: [4,4]
    # X: [3,n] or [4,n]
    return: [3,n] or [4,n]
    '''
    if X.shape[0] == 3:
        homoX = np.ones([4, X.shape[1]])
        homoX[:3,:] = X 
        homoX = np.matmul(R, homoX)
        return homoX[:3,:]
    elif X.shape[0] == 4:
        return np.matmul(R, homoX)
    else:
        import ipdb; ipdb.set_trace()
        print(X.shape)

