import torch
import numpy as np
from lxml import etree
import os
from util import str2arr

def find_elem(etree_elem, tag, attr_type=None, attr_value=None, child_only=False):
    if child_only:
        xpath = "./"
    else:
        xpath = ".//"

    if attr_type:
        return [
            tag
            for tag in etree_elem.iterfind(
                '{}{}[@{}="{}"]'.format(xpath, tag, attr_type, attr_value)
            )
        ]
    else:
        return [tag for tag in etree_elem.iterfind("{}{}".format(xpath, tag))]

def etree_from_xml(xml, ispath=True):
    """Load xml as etree and return root and tree."""
    if ispath:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.parse(xml, parser).getroot()
    else:
        root = etree.fromstring(xml)
    tree = etree.ElementTree(root)
    return root, tree

def tree_treversal(order, depths, parent, reverse=False):
    """depth first traversal to find sequence of limbs"""
    children = find_elem(order[-1], "body", child_only=True)
    if reverse:
        children = children[::-1]
    for c in children:
        order.append(c)
        depths.append(parent)
        tree_treversal(order, depths, parent+1, reverse=reverse)

# normalization scheme
# density: [500, 1000] -> [0, 1]
# gear: [150, 300] -> [0, 1]
# joint_range1: [-90 0] -> abs -> [0 1]
# joint_range2: [0 90] -> [0 1]
# joint_axis: should be normalized
# joint bit for jointx, jointy (0/1) (idx: 3 (jointx), 13 (jointy)

# parse each limb
def encode_each_limb(limb, depth, default, actuator):
    input_vector = []
    first_joint = True
    pos = limb.get('pos')
    if pos is None:
        pos = "0 0 0"
    input_vector.extend(str2arr(pos))
    for idx, item in enumerate(limb):
        if item.tag == 'joint':
            joint_type = item.get('name').split("/")[0][-1]
            if first_joint and joint_type == "y":
                input_vector.extend([0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # only jointy
            input_vector.append(1) # joint
            name = item.get('name')
            # if name == 'root':
            #     continue
            cls = item.get('class')
            joint = find_elem(default, 'default', 'class', cls)[0]
            joint = find_elem(joint, 'joint')[0]
            limited = joint.get('limited')
            limited = True if limited == 'true' else False
            # input_vector.append(limited)
            # damping = float(joint.get('damping'))
            # armature = float(joint.get('armature'))
            # stiffness = float(joint.get('stiffness'))
            # input_vector.extend([damping, armature, stiffness]) # joint type default: hinge
            ########### dynamic parameters for joint ###########
            # if cls == 'normal_joint' or cls == 'walker_joint':
            #     solimplimit = str2arr(joint.get('solimplimit'))
            #     range = str2arr(joint.get('range'))
            #     input_vector.extend(solimplimit)
            #     input_vector.extend(range)
            # elif cls == 'stiff_joint':
            #     solimplimit = str2arr(joint.get('solimplimit'))
            #     input_vector.extend(solimplimit)
            #     input_vector.extend([0., 0.])
            # else:
            #     input_vector.extend([0., 0., 0.])
            #     input_vector.extend([0., 0.])
            ##########################################
            joint_range = str2arr(item.get('range'))
            joint_range[0] = abs(-joint_range[0]) / 90.
            joint_range[1] = joint_range[1] / 90.
            joint_pos = str2arr(item.get('pos'))
            joint_axis = str2arr(item.get('axis')) # rotation axis
            # joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-14) # normalization
            input_vector.extend(joint_range)
            input_vector.extend(joint_pos)
            input_vector.extend(joint_axis)
            actuator_config = find_elem(actuator, 'motor', 'name', name)
            if len(actuator_config) == 1:
                gear = actuator_config[0].get('gear')
                input_vector.append((float(gear) - 150.) / 150.)
            else:
                input_vector.append(0.) # for torso (but no need to consider)
            if first_joint and joint_type == "x" and idx+1 < len(limb) and limb[idx+1].tag != 'joint':
                    input_vector.extend([0, 0., 0., 0., 0., 0., 0., 0., 0., 0.]) # only jointx
            
            first_joint = False

        elif item.tag == 'site':
            name = item.get('name')
            if name == 'root':
                continue
            cls = item.get('class')
            site = find_elem(default, 'default', 'class', cls)[0]
            site = find_elem(site, 'site')[0]
            # if site.get('type') is None:
            #     site_type = 'sphere'
            # else:
            #     site_type = site.get('type')
            # input_vector.append(site_type)
            size = item.get('size')
            # if size is None:
            #     size = float(item.get('size'))
            # else:
            #     size = str2arr(size)[0]
            if size is not None:
                input_vector.append(float(size))
            # group = site.get('group')
            # if group is None:
            #     group = 0
            # else:
            #     group = str2arr(group)[0]
            # input_vector.append(group)
            # if cls == 'imu_vel' or cls == 'touch_site':
            #     rgba = str2arr(site.get('rgba'))
            # else:
            #     rgba = [0.5, 0.5, 0.5, 1]
            # input_vector.extend(rgba)

            site_pos = item.get('pos')
            if site_pos is None:
                input_vector.extend(str2arr(item.get('fromto')))
            else:
                input_vector.extend(str2arr(site_pos))


        elif item.tag == 'geom':
            geom_type = item.get('type')
            # input_vector.append(geom_type) # default: 'capsule'
            if geom_type == 'capsule':
                input_vector.extend(str2arr(item.get('fromto')))
            else:
                input_vector.extend([0.] * 6)
            input_vector.append(float(item.get('size')))
            input_vector.append((float(item.get('density')) - 500) / 500.)
            # geom_condim = item.get('condim')
            # if geom_condim is None:
            #     input_vector.append(3)
            # else:
            #     input_vector.append(int(geom_condim))

        # elif item.tag == 'camera':
        #     input_vector.extend(str2arr(item.get('pos')))
        #     input_vector.extend(str2intarr(item.get('xyaxes')))
        #     input_vector.append(item.get('mode'))
    input_vector.append(depth)
    # print(input_vector)
    return input_vector

def encode_limbs(base_dir):
    # tensor of training dataset
    max_limbs = 10
    data = []
    for xml_str in os.listdir(base_dir):

        root, tree = etree_from_xml(base_dir + xml_str, ispath=True)
    
        worldbody = root.findall("./worldbody")[0]
        default = root.findall("./default")[0]
        actuator = root.findall("./actuator")[0]
        root = find_elem(worldbody, "body", "name", "torso/0")[0]
        pos = str2arr(root.get("pos"))

        depths = [0] # torso/end of sequence: 0
        orig_order = [root]
        tree_treversal(orig_order, depths, parent=1)
        # print(depths)

        input = []
        for depth, limb in zip(depths[1:], orig_order[1:]):
            input_vector = encode_each_limb(limb, depth, default, actuator)
            input.append(input_vector)
        
        input = torch.tensor(np.array(input))

        num_limbs = input.shape[0]

        input = torch.nn.functional.pad(input, (0, 0, 0, max_limbs - num_limbs), mode='constant', value=0)
        data.append(input)

    data = torch.stack(data, dim=0)
    return data