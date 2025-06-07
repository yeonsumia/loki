import argparse
import math
import os
import numpy as np

try:
    from derl.utils import file as fu
    from derl.utils import geom as gu
    from derl.utils import xml as xu
except:
    import sys
    sys.path.append("..")
    from derl.derl.utils import file as fu
    from derl.derl.utils import geom as gu
    from derl.derl.utils import xml as xu



def convert_all_xml_to_pkl(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir, len(os.listdir(input_dir)))

    for filename in os.listdir(input_dir):
        # Check if the file is a .pkl file
        if filename.endswith('.xml'):
            xml_path = os.path.join(input_dir, filename)
            print(xml_path)
            pkl = xml_to_pkl(xml_path)

            output_path = os.path.join(output_dir, filename.replace('.xml', '.pkl'))

            fu.save_pickle(pkl, output_path)


def xml_to_pkl(xml_path):
    # load xml
    root, tree = xu.etree_from_xml(xml_path)
    worldbody = root.findall("./worldbody")[0]
    actuator = root.findall("./actuator")[0]
    contact = root.findall("./contact")[0]

    # get body
    body = worldbody.findall("./body")[0]

    # get dfs list of bodies
    limb_elem, parent_idx, depth = dfs(body, [], [], [0]) # must allocate initial lists for res, parent_idx, depth
    print(f"parent_idx: {parent_idx}")
    print(f"depth: {depth}")

    # calculate orient from relative coordinates
    result = {}
    id_to_idx = {}
    result['num_limbs'] = result['limb_idx'] = len(limb_elem)
    result['num_torso'] = 1
    result['torso_list'] = [0]
    result['growth_torso'] = [0]
    result['body_params'] = {
        'torso_mode': 'horizontal_y' if len(xu.find_elem(limb_elem[0], "site", "name", "torso/horizontal_y/0")) > 0 else 'vertical'
    }
    result['limb_metadata'] = dict()
    result['limb_list'] = []

    for i, e in enumerate(limb_elem):
        id = xu.name2id(e) if e.attrib['name'] != 'torso/0' else -1
        if id > -1:
            result['limb_list'].append(id)
        result['limb_metadata'][i-1] = dict()
        
        geom = e.findall("./geom")[0]

        result['limb_metadata'][i-1]['depth'] = depth[i]
        result['limb_metadata'][i-1]['limb_density'] = float(geom.attrib['density'])
        result['limb_metadata'][i-1]['limb_radius'] = float(geom.attrib['size'])

        if id == -1:
            print(f"{result['limb_metadata'][i-1]}")
            continue

        id_to_idx[id] = i-1
        x_f, y_f, z_f, x_t, y_t, z_t = xu.str2arr(geom.attrib['fromto'])
        result['limb_metadata'][i-1]['limb_height'] = round(math.sqrt((x_f - x_t) ** 2 + (y_f - y_t) ** 2 + (z_f - z_t) ** 2) - float(geom.attrib['size']), 2)
        
        joint = e.findall("./joint")

        result['limb_metadata'][i-1]['joint_range'] = dict()
        result['limb_metadata'][i-1]['joint_axis'] = ''
        result['limb_metadata'][i-1]['gear'] = dict()

        for j in joint:
            joint_range = xu.str2arr(j.attrib['range'])
            if j.attrib['name'] == f'limbx/{id}':
                result['limb_metadata'][i-1]['joint_range']['x'] = (float(joint_range[0]), float(joint_range[1]))
                result['limb_metadata'][i-1]['joint_axis'] += 'x'
                result['limb_metadata'][i-1]['gear']['x'] = float(xu.find_elem(actuator, "motor", "joint", f'limbx/{id}')[0].attrib['gear'])
            elif j.attrib['name'] == f'limby/{id}':
                result['limb_metadata'][i-1]['joint_range']['y'] = (float(joint_range[0]), float(joint_range[1]))
                result['limb_metadata'][i-1]['joint_axis'] += 'y'
                result['limb_metadata'][i-1]['gear']['y'] = float(xu.find_elem(actuator, "motor", "joint", f'limby/{id}')[0].attrib['gear'])
        

        result['limb_metadata'][i-1]['parent_name'] = 'torso/0' if parent_idx[i-1] == -1 else f"limb/{id_to_idx[parent_idx[i-1]]}"
        xml_parent_name = 'torso/0' if parent_idx[i-1] == -1 else f"limb/{parent_idx[i-1]}"
        # find parent
        parent = xu.find_elem(root, "body", "name", xml_parent_name)[0]
        
        # calculate possible position
        growth_sites = parent.findall("./site")
        pos = xu.str2arr(e.attrib['pos'])
        p_r = float(parent.findall("./geom")[0].attrib['size'])
        step_size = 45
        for site in growth_sites:
            if site.attrib['class'] == 'growth_site' or site.attrib['class'] == 'mirror_growth_site':
                parent_pos = xu.str2arr(site.attrib['pos'])
                # print(f"parent_pos: {parent_pos}, p_r: {p_r}")
                # Theta is angle from x axis. 0 <= theta <= 360 - step_size
                theta = np.radians(np.arange(360 / step_size) * step_size)
                # Phi is angle from z axis. 90 <= phi <= 180
                phi = np.radians(np.arange((90 / step_size) + 1) * step_size + 90)

                for t in theta:
                    for p in phi:
                        estimate_pos = xu.add_list(parent_pos, gu.sph2cart(p_r, t, p))
                        diff = np.linalg.norm(xu.sub_list(pos, estimate_pos)) 
                        # print(f"estimate_pos: {estimate_pos}, diff: {diff}")
                        if diff < 1e-3:
                            result['limb_metadata'][i-1]['orient'] = (result['limb_metadata'][i-1]['limb_height'], t, p)
                            result['limb_metadata'][i-1]['site'] = site.attrib['name']
                            if site.attrib['name'] != 'torso/0':
                                result['limb_metadata'][i-1]['site'] = "/".join(site.attrib['name'].split('/')[:-1] + [str(id_to_idx[parent_idx[i-1]])])


        print(f"{result['limb_metadata'][i-1]}")
    return result


def dfs(body, res=[], parent_idx=[], depth=[0]):
    res.append(body)
    print(body.attrib['name'])
    neighbors = body.findall("./body")
    for n in neighbors:
        if "torso" in body.get("name"):
            parent_idx.append(-1)
            depth.append(1)
        else:
            parent_idx.append(xu.name2id(body))
            depth.append(depth[parent_idx.index(parent_idx[-1])] + 1)
        dfs(n, res, parent_idx, depth)
    return res, parent_idx, depth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)

    args = parser.parse_args()

    input_dir = args.input_dir

    pkl_dir = os.path.join(input_dir, 'unimal_init')
    os.makedirs(pkl_dir, exist_ok=True)
    xml_dir = os.path.join(input_dir, 'xml')
    convert_all_xml_to_pkl(xml_dir, pkl_dir)

