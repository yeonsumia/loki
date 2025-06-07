import os, sys
import json, torch
sys.path.append("..")
from vae.webdata import data_decoder, TensorWebDataset, FilteredTensorWebDataset, get_webdataloader

if __name__ == "__main__":
    dir = sys.argv[1]
    tar_name = sys.argv[2]
    curr_dir = os.getcwd()
    # path = os.path.join(curr_dir, dir, "ft", "xml")
    # pkl_path = os.path.join(curr_dir, dir, "ft", "unimal_init")
    # vec_path = os.path.join(curr_dir, dir, "ft", "unimal_init_vec")
    tar_path = os.path.join(curr_dir, dir)
    pkl_path = os.path.join(curr_dir, dir)
    vec_path = os.path.join(curr_dir, dir)
    assert os.path.exists(tar_path), f"Path {tar_path} does not exist"
    assert os.path.exists(pkl_path), f"Path {pkl_path} does not exist"
    assert os.path.exists(vec_path), f"Path {vec_path} does not exist"

    save_path = os.path.join(curr_dir, tar_name, "ft")
    os.makedirs(save_path, exist_ok=True)

    D_TOKEN = 16

    TEST_PATH = '../new-webdataset.tar'

    key_list_save_path = os.path.join(curr_dir, f"new-webdataset_training_key_list.pt")
    if os.path.exists(key_list_save_path):
        # load key_list json
        with open(key_list_save_path, 'r') as f:
            key_list = json.load(f)
        key_list = set(key_list)
        print(f"Training data key list loaded from {key_list_save_path}")
    else:
        train_ratio = 0.9
        mod = int(1/(1-train_ratio))
        train_dataset = FilteredTensorWebDataset(TEST_PATH, input_dim=D_TOKEN, filter=lambda sample: int(sample['__key__']) % mod != 0).decode(data_decoder)
        train_dataloader = get_webdataloader(train_dataset, batch_size=1)

        # get training data key list
        key_list = []
        for i, (key, x, mask) in enumerate(train_dataloader):
            if i % 10000 == 0:
                print(f"Processing {i} / current key: {key[0]}")
            key_list.append(key[0])

        # save training data key list to json
        with open(key_list_save_path, 'w') as f:
            json.dump(key_list, f)
        print(f"Training data key list saved to {key_list_save_path}")

    visit_file_name = []
    for i, file in enumerate(os.listdir(tar_path)):
        # check if xml file key is in training data key list
        file_name = file.split(".")[0]
        if file_name not in key_list:
            print(f"key {file_name} not found in training data key list")
            continue
        # if file_name in visit_file_name:
        #     print(f"key {file_name} already visited")
        #     continue
        # check if pkl/vec file exists and not empty
        if not os.path.exists(os.path.join(pkl_path, file.replace('.xml', '.pkl'))):
            print(f"pkl file not found for {file}")
            continue
        if os.path.getsize(os.path.join(pkl_path, file.replace('.xml', '.pkl'))) == 0:
            print(f"pkl file empty for {file}")
            continue
        if not os.path.exists(os.path.join(vec_path, file.replace('.xml', '.pt'))):
            print(f"vec file not found for {file}")
            continue
        if os.path.getsize(os.path.join(vec_path, file.replace('.xml', '.pt'))) == 0:
            print(f"vec file empty for {file}")
            continue
        # copy xml to save_path
        os.system(f"cp {os.path.join(tar_path, file)} {save_path}/{i}.xml")
        # copy pkl to save_path
        os.system(f"cp {os.path.join(pkl_path, file.replace('.xml', '.pkl'))} {save_path}/{i}.pkl")
        # copy vec to save_path
        os.system(f"cp {os.path.join(vec_path, file.replace('.xml', '.pt'))} {save_path}/{i}.pt")

        visit_file_name.append(file_name)

    os.system(f"ls -1 {save_path} | sort > tmpt")
    os.system(f"tar -cvf {tar_name}.tar -C {save_path} -T tmpt")
    os.system("rm tmpt")