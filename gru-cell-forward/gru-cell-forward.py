import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def gru_cell_forward(x: list, h_prev: list, params: dict) -> np.ndarray:
    """
    Returns the updated hidden state as a NumPy array matching the shape of h_prev.
    Supports both single samples and batched inputs.
    """
    x = np.array(x)
    h_prev = np.array(h_prev)
    
    # Extract parameters
    Wz = np.array(params["Wz"])
    Uz = np.array(params["Uz"])
    bz = np.array(params["bz"])
    
    Wr = np.array(params["Wr"])
    Ur = np.array(params["Ur"])
    br = np.array(params["br"])
    
    Wh = np.array(params["Wh"])
    Uh = np.array(params["Uh"])
    bh = np.array(params["bh"])
    
    # 1. Update gate z_t
    z = sigmoid(np.matmul(x, Wz) + np.matmul(h_prev, Uz) + bz)
    
    # 2. Reset gate r_t
    r = sigmoid(np.matmul(x, Wr) + np.matmul(h_prev, Ur) + br)
    
    # 3. Candidate hidden state h~_t
    # r ⊙ h_prev uses element-wise multiplication (*)
    h_tilde = np.tanh(np.matmul(x, Wh) + np.matmul(r * h_prev, Uh) + bh)
    
    # 4. New hidden state h_t
    # (1 - z) ⊙ h_prev + z ⊙ h~_t uses element-wise multiplication (*)
    h_new = (1 - z) * h_prev + z * h_tilde
    
    return h_new