import numpy as np
import torch

# Data params
CONTINUOUS_TOKEN = 4
CATEGORY_TOKEN = 4

THETA_CLASS = 8
PHI_CLASS = 3
JOINTX_CLASS = 13
JOINTY_CLASS = 13
TOTAL_CATEGORY = THETA_CLASS + PHI_CLASS + JOINTX_CLASS + JOINTY_CLASS

BINARY_TOKEN = 5
# D_TOKEN = 14 # dimension of each limb
D_TOKEN = 14
N_TOKENS = 11 # maximum number of limbs (include head)

def str2arr(string):
    """Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]
    """
    return np.array([float(x) for x in string.split(" ")])

def arr2str(array):
    """Converts a np.array to a string.

    Examples:
        [0, 1, 2] => "0 1 2"
    """
    return " ".join([str(x) for x in array])

def check_dfs_validity(depth_list):
    """Check if the depth list is valid for DFS traversal.
    
    Args:
        depth_list (list): A list of depth values. 
    
    Examples:
        [0, 1, 2, 3, 1, 2, 1, 1, 2, 3, 1] => True
    """
    if depth_list[0] != 0:
        return False

    parent_candidates = dict() # depth -> node
    parent_candidates[0] = True
    for i in range(1, 10):
        parent_candidates[i] = False
    
    prev_depth = 0
    
    for depth in depth_list[1:]: # depth: [0 ... 10]
        diff_depth = depth - prev_depth
        if diff_depth > 1:
            return False
        elif diff_depth <= 0 and not parent_candidates.get(depth-1, False):
            return False
        parent_candidates[depth] = True
        prev_depth = depth
        for j in range(depth+1, 10):
            parent_candidates[j] = False

    return True

def validity_check(input_vector, train_depth_vector=None):
    """Check the validity of the limb vector.
    
    Args:
        input_vector (torch.Tensor): A tensor of limb vector. [max_limbs, D_TOKEN]
        train_depth_vector (torch.Tensor): A tensor of ground truth depth vector.
    """
    num_vector, depth_vector = torch.split(input_vector, [D_TOKEN-1, 1], dim=-1)

    # find EOS
    seq_len = -1
    # 
    
    try:
        for i in range(num_vector.shape[0]):
            if num_vector[i, -1] == 1:
                seq_len = i+1
                break
            
    except:
        raise ValueError("no EOS")

    if seq_len == -1:
        raise ValueError("no EOS")
    
    if seq_len < 2:
        raise ValueError("do not allow only a head")

    num_vector = num_vector[:seq_len, :]  # [seq_len, D_TOKEN-1]
    depth_vector = depth_vector[:seq_len, 0].long() # [seq_len]
    
    
    # joint bit validity check
    for i in range(seq_len):
        jointx_bit, jointy_bit = num_vector[i, -BINARY_TOKEN:-BINARY_TOKEN+2]
        
        if i > 0 and jointx_bit == 0 and jointy_bit == 0:
            raise ValueError(f"{i}th limb has no joint")

        depth = depth_vector[i]
        if train_depth_vector is not None:
            true_depth = train_depth_vector[i]
            if depth != int(true_depth):
                raise ValueError("invalid depth")
    
    # depth sequence validity check
    if not check_dfs_validity(depth_vector.tolist()):
        raise ValueError("invalid depth-first sequence")

    return seq_len

def denormalize(value, min_value, max_value):
    return value * (max_value - min_value) + min_value

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)
