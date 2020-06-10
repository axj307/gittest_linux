import numpy as np

epsilon = 1e-8
M = 100
K = 50
D = 30
N = 10
nCheck = 1000

def swish(x):
    return x / (1+np.exp(-x))

def der_swish(x):
    f = x / (1+np.exp(-x))
    sigma = 1 / (1+np.exp(-x))
    return f + sigma*(1 - f)

def reLu(x):
    return x * (x > 0)

def der_reLu(x):
    return 1*(x>0)

def forwardprop(x, t, A, B, C):
    # ---------- 
    x_t = np.vstack((x , 1))
    Ax = A@x_t
    
    y = swish(Ax)
    y_t = np.vstack((y , 1))
    By = B@y_t
    
    z = swish(By)
    z_t = np.vstack((z , 1))
    Cz = C@z_t

    h = reLu(Cz)  # Prediction
    E = np.linalg.norm(h - t)**2  # Error
    # -------------------------------------------------
    return y, z, h, E


def backprop(x, t, A, B, C):

    y, z, h, J = forwardprop(x, t, A, B, C)
    x_t = np.vstack((x , 1))  #x_tilde
    y_t = np.vstack((y , 1))  #y_tilde
    z_t = np.vstack((z , 1))  #z_tilde
    A_t = A[:,:-1] #A_tilde
    B_t = B[:,:-1] #B_tilde
    C_t = C[:,:-1] #C_tilde
        
    dEdh = (h - t).T  # t_hat = h  
    dhdp = der_reLu( np.diag((C@z_t).reshape(-1,)) )
    dEdp = dEdh@dhdp
    
    dpdz =  C_t
    dhdz = dhdp@dpdz
    dzdq = der_swish( np.diag( (B@y_t).reshape(-1,) ))
    dEdz = dEdh@dhdz
    dEdq = dEdz@dzdq
    
    dzdy = dzdq@B_t 
    dydr = der_swish( np.diag( (A@x_t).reshape(-1,) ))
    dEdr = dEdz@dzdy@dydr

    grad_C = np.zeros(C.shape)
    grad_B = np.zeros(B.shape)
    grad_A = np.zeros(A.shape)    
    for i in range(len(grad_C)):
        grad_C[i, :] = dEdp[:,i]@z_t.T
    for i in range(len(grad_B)):
        grad_B[i, :] =dEdq[:,i]@y_t.T
    for i in range(len(grad_A)):
        grad_A[i, :] =dEdr[:,i]@x_t.T        
    # -------------------------------------------------

    return grad_A, grad_B, grad_C

def gradient_check():

    A = np.random.rand(K, M+1)*0.1-0.05
    B = np.random.rand(D, K+1)*0.1-0.05
    C = np.random.rand(N, D+1)*0.1-0.05
    x = np.random.rand(M, 1)*0.1-0.05
    t = np.random.rand(N, 1)*0.2-0.1

    grad_A, grad_B, grad_C = backprop(x, t, A, B, C)
    errA, errB, errC = [], [], []
    
 
    for i in range(1000):
        Ap = np.copy(A)
        Am = np.copy(A)
        Bp = np.copy(B)
        Bm = np.copy(B)
        Cp = np.copy(C)
        Cm = np.copy(C)
        
        idx_x, idx_y = np.random.randint(C.shape)  ## random between (10,31)
        Cp[idx_x, idx_y] = Cp[idx_x, idx_y]+ epsilon
        Cm[idx_x, idx_y] = Cm[idx_x, idx_y]- epsilon
        _, _, _, Jp = forwardprop(x, t, A, B, Cp)
        _, _, _, Jm = forwardprop(x, t, A, B, Cm)
        numerical_grad_C = (Jp - Jm)/(2*epsilon)
        errC.append(np.abs(grad_C[idx_x, idx_y] - numerical_grad_C))

        idx_x, idx_y = idx_x+20, idx_y+20   ## random between (30,51)
        Bp[idx_x, idx_y] = Bp[idx_x, idx_y]+ epsilon
        Bm[idx_x, idx_y] = Bm[idx_x, idx_y]- epsilon
        _, _, _, Jp = forwardprop(x, t, A, Bp, C)
        _, _, _, Jm = forwardprop(x, t, A, Bm, C)
        numerical_grad_B = (Jp - Jm)/(2*epsilon)
        errB.append(np.abs(grad_B[idx_x, idx_y] - numerical_grad_B))

        idx_x, idx_y = idx_x+20, idx_y+50 ## random between (50,101)
        Ap[idx_x, idx_y] = Ap[idx_x, idx_y]+ epsilon
        Am[idx_x, idx_y] = Am[idx_x, idx_y]- epsilon
        _, _, _, Jp = forwardprop(x, t, Ap, B, C)
        _, _, _, Jm = forwardprop(x, t, Am, B, C)
        numerical_grad_A = (Jp - Jm)/(2*epsilon)
        errA.append(np.abs(grad_A[idx_x, idx_y] - numerical_grad_A))




    
    #-------------------------------------------------

    print('Gradient checking A, MAE: {0:0.8f}'.format(np.mean(errA)))
    print('Gradient checking B, MAE: {0:0.8f}'.format(np.mean(errB)))
    print('Gradient checking C, MAE: {0:0.8f}'.format(np.mean(errC)))

if __name__ == '__main__':
    gradient_check()
    

# Gradient checking A, MAE: 0.00086885
# Gradient checking B, MAE: 0.00046250
# Gradient checking C, MAE: 0.00162964