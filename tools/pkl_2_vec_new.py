import sys
import os
import torch
import numpy as np
import sys

try:
    from derl.utils import file as fu
    from derl.config import cfg
except:
    import sys
    sys.path.append('..')
    from derl.derl.utils import file as fu
    from derl.derl.config import cfg

from tools.config import *
from tools.util import normalize, N_TOKENS

step_size = 45
# Theta is angle from x axis. 0 <= theta <= 360 - step_size
theta = np.radians(np.arange(360 / step_size) * step_size)
# Phi is angle from z axis. 90 <= phi <= 180
phi = np.radians(np.arange((90 / step_size) + 1) * step_size + 90)
joint_range_list = cfg.BODY.JOINT_ANGLE_LIST
print(f"Theta: {theta} \n Phi: {phi} \n Joint Range: {joint_range_list}")

def get_nearest_element_index(arr, val):
    return np.argmin(np.abs(arr - val))

def get_1d_list_index_from_2d_list(arr, val):
    for i, a in enumerate(arr):
        if a[0] == val[0] and a[1] == val[1]:
            return i
    raise ValueError(f"Value {val} not found in list {arr}")


def get_joint_features(limb_dict, axis='x'):
    joint_gear = limb_dict['gear'][axis]
    joint_range = limb_dict['joint_range'][axis]
    joint_gear = normalize(joint_gear, JOINT_GEAR_MIN, JOINT_GEAR_MAX)
    return joint_gear, get_1d_list_index_from_2d_list(joint_range_list, list(joint_range))


def convert_all_pkl_to_vec(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir, len(os.listdir(input_dir)))
    error_count = 0

    for filename in os.listdir(input_dir):
        # Check if the file is a .pkl file
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(input_dir, filename)
            try:
                a = fu.load_pickle(pkl_path)
            except EOFError:
                print(f"EOFError: {pkl_path}")
                continue

            try:
                input_vector, mask_vector = pkl_to_vec(a)
            except ValueError as e:
                print(f"seq_len should be between 3 and 11: {e} ({pkl_path})")
                error_count += 1
                continue
            combined_vector = torch.cat((input_vector, mask_vector), dim=1)
            # print(combined_vector.shape)

            output_path = os.path.join(output_dir, filename.replace('.pkl', '.pt'))

            fu.save_vector(combined_vector, output_path)  # Assuming you have a function to save vectors
    print(f"Error count: {error_count}")

def dict_to_vec(limb_dict, is_eos):
    """
    Convert a limb dictionary to a limb vector and a mask vector for training transformer model
    
    limb vector:
    [
        orientation_radius, 
        jointx_gear, 
        jointy_gear, 
        limb_density, 
        orientation_theta_class [0,THETA_CLASS)
        orientation_phi_class [0,PHI_CLASS)
        jointx_range_class [0,JOINTX_CLASS)
        jointy_range_class [0,JOINTY_CLASS)
        jointx_bit (0/1), 
        jointy_bit (0/1), 
        torso_mode (0: horizontal_y, 1: vertical), 
        attach_site_type (-1: torso, 0: btm, 1: mid), 
        EOS (0/1), 
        depth
    ]
    """
    # torso
    if limb_dict['depth'] == 0:
        # limb_radius = normalize(limb_dict['limb_radius'], HEAD_RADIUS_MIN, HEAD_RADIUS_MAX) # head radius
        limb_density = normalize(limb_dict['limb_density'], HEAD_DENSITY_MIN, HEAD_DENSITY_MAX) # head density
        torso_mode = limb_dict['torso_mode']
        mode = 0
        if torso_mode == 'horizontal_y':
            mode = 0
        elif torso_mode == 'vertical':
            mode = 1
        else:
            raise NotImplementedError
            
        return [
                0.,           # orientation_raidus (not used)
                0.,           # jointx_gear (not used)
                0.,           # jointy_gear (not used)
                limb_density, # limb density
                -1,           # orientation_theta (not used)
                -1,           # orientation_phi (not used)
                -1,           # jointx bit (not used)
                -1,           # jointy bit (not used)
                -1,           # jointx_range_class (not used)
                -1,           # jointy_range_class (not used)
                mode,         # torso mode
                -1,           # attach site type (not used)
                is_eos,       # EOS
                0,            # depth
            ], [
                False, 
                False, 
                False, 
                True,
                False, 
                False, 
                False, 
                False, 
                False, 
                False, 
                True, 
                False, 
                True, 
                True
            ]
        
    # limb
    ##################################
    ####### numerical features #######
    ##################################

    # orientation
    r, t, p = limb_dict['orient']
    vec = [
            normalize(r, LIMB_HEIGHT_MIN, LIMB_HEIGHT_MAX),
        ]

    # joint
    if limb_dict['joint_axis'] == 'xy':
        jointx_gear, jointx_class = get_joint_features(limb_dict, axis='x')
        jointy_gear, jointy_class = get_joint_features(limb_dict, axis='y')
        vec.extend([jointx_gear, jointy_gear])
        jointx_bit = 1
        jointy_bit = 1
    elif limb_dict['joint_axis'] == 'x':
        jointx_gear, jointx_class = get_joint_features(limb_dict, axis='x')
        vec.extend([jointx_gear, 0.])
        jointx_bit = 1
        jointy_bit = 0
        jointy_class = -1
    elif limb_dict['joint_axis'] == 'y':
        jointy_gear, jointy_class = get_joint_features(limb_dict, axis='y')
        vec.extend([0., jointy_gear]) 
        jointx_bit = 0
        jointy_bit = 1
        jointx_class = -1
    else:
        raise NotImplementedError

    # radius
    # vec.append(normalize(limb_dict['limb_radius'], LIMB_RADIUS_MIN, LIMB_RADIUS_MAX))

    # density
    vec.append(normalize(limb_dict['limb_density'], LIMB_DENSITY_MIN, LIMB_DENSITY_MAX))

    ##################################
    ###### categorical features ######
    ##################################
    # orientation theta
    theta_class = get_nearest_element_index(theta, t)
    vec.append(theta_class)
    # orientation phi
    phi_class = get_nearest_element_index(phi, p)
    vec.append(phi_class)

    # jointx range class
    vec.append(jointx_class)

    # jointy range class
    vec.append(jointy_class)

    # joint bit
    vec.append(jointx_bit)
    vec.append(jointy_bit)

    # torso_mode (not used)
    vec.append(-1)

    # attach site type
    if limb_dict['site'].split('/')[0] == 'torso':
        attach_mask = False
        vec.append(-1)
    else:
        attach_type = limb_dict['site'].split('/')[1]
        if attach_type == 'btm':
            vec.append(0)
        elif attach_type == 'mid':
            vec.append(1)
        else:
            raise NotImplementedError
        attach_mask = True
        
    # EOS
    vec.append(is_eos)

    # depth
    vec.append(limb_dict['depth'])

    ##################################

    return vec, [True,                    # orientation_radius
                 jointx_bit,              # jointx_gear
                 jointy_bit,              # jointy_gear
                 True,                    # limb_density
                 True,                    # orientation_theta
                 True,                    # orientation_phi
                 jointx_bit,              # jointx_range_class
                 jointy_bit,              # jointy_range_class            
                 True,                    # jointx_bit (0/1)
                 True,                    # jointy_bit (0/1)
                 False,                   # torso_mode (not used)
                 attach_mask,             # attach_site_type
                 True,                    # EOS
                 True,                    # depth
                ]


def dfs(i, visited, parents, res, limbs):
    if visited[i]:
        return
    else:
        res.append(limbs[i])
        visited[i] = True
        for j in range(len(parents)):
            if not visited[j] and parents[j] == limbs[i][0]:
                dfs(j, visited, parents, res, limbs)


def pkl_to_vec(pkl_dict):
    pkl_dict['limb_metadata'][-1]['torso_mode'] = pkl_dict['body_params']['torso_mode']
    limbs = list(pkl_dict['limb_metadata'].items()) # assume that order of this list means the order of when it is added into the graph
    num_limbs = len(limbs)

    if num_limbs < 3 or num_limbs > 11:
        raise ValueError(num_limbs)

    limbs.sort()
    parents = []

    for l in limbs:
        if l[1].get('parent_name') is None:
            parents.append(-2) # no parent
        else:
            parent_type, parent_idx = l[1]['parent_name'].split('/')
            if parent_type == 'torso':
                parents.append(-1)
            elif parent_type == 'limb':
                parents.append(int(parent_idx))
            else:
                raise NotImplementedError

    dfs_result = []
    # print(parents)
    visited = [False for _ in range(len(parents))]
    dfs(0, visited, parents, dfs_result, limbs)

    # print(dfs_result)
    input_vector = []
    mask_vector = []
    for i, (limb_idx, limb_dict) in enumerate(dfs_result):
        is_eos = 1 if i == len(dfs_result) - 1 else 0
        vec, mask = dict_to_vec(limb_dict, is_eos)
        input_vector.append(vec)
        mask_vector.append(mask)

    input_vector = torch.tensor(np.array(input_vector))
    mask_vector = torch.tensor(np.array(mask_vector))
    
    num_limbs = input_vector.shape[0]

    input_vector = torch.nn.functional.pad(input_vector, (0, 0, 0, N_TOKENS - num_limbs), mode='constant', value=0)
    mask_vector = torch.nn.functional.pad(mask_vector, (0, 0, 0, N_TOKENS - num_limbs), mode='constant', value=False)
    
    return input_vector, mask_vector


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Usage: python pkl_2_vec.py <data_dir>")
    
    data_dir = "./derl/" + sys.argv[1]
    ############################
    # convert single pkl file
    ############################
    # # get pkl path argument
    # if len(sys.argv) < 2:
    #     print("Usage: python vec_to_pkl.py <pkl_path>")
    #     sys.exit(1)
    # pkl_path = sys.argv[1]

    # # load pkl file
    # a = fu.load_pickle(pkl_path)
    # input_vector, mask_vector = pkl_to_vec(a)
    # print(input_vector, mask_vector, input_vector.shape)

    ###########################
    # convert all pkl file
    ############################
    
    # convert_all_pkl_to_vec('./output/ft/test', './output/ft/test_vec')
    # if from_sample == 'init_sample':
    #     convert_all_pkl_to_vec(f'{data_dir}/unimal_pkl_recons_sample/0/', f'{data_dir}/unimal_init_vec')
    # elif from_sample == 'sample':
    #     convert_all_pkl_to_vec(f'{data_dir}/unimal_pkl_recons_sample/', f'{data_dir}/unimal_final_vec')
    # else:
    #     convert_all_pkl_to_vec(f'{data_dir}/unimal_pkl_recons_sample_final', f'{data_dir}/unimal_pkl_recons_sample_final_vec')

    convert_all_pkl_to_vec(data_dir, f'{data_dir}/../unimal_init_vec')