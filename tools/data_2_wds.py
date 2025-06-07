import os, sys

if __name__ == "__main__":
    dir = sys.argv[1]
    tar_name = sys.argv[2]
    curr_dir = os.getcwd()
    path = os.path.join(curr_dir, dir, "ft", "xml")
    pkl_path = os.path.join(curr_dir, dir, "ft", "unimal_init")
    vec_path = os.path.join(curr_dir, dir, "ft", "unimal_init_vec")
    assert os.path.exists(path), f"Path {path} does not exist"
    assert os.path.exists(pkl_path), f"Path {pkl_path} does not exist"
    assert os.path.exists(vec_path), f"Path {vec_path} does not exist"

    save_path = os.path.join(curr_dir, tar_name, "ft")
    os.makedirs(save_path, exist_ok=True)

    for i, file in enumerate(os.listdir(path)):
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
        os.system(f"mv {os.path.join(path, file)} {save_path}/{i}.xml")
        # copy pkl to save_path
        os.system(f"mv {os.path.join(pkl_path, file.replace('.xml', '.pkl'))} {save_path}/{i}.pkl")
        # copy vec to save_path
        os.system(f"mv {os.path.join(vec_path, file.replace('.xml', '.pt'))} {save_path}/{i}.pt")

    os.system(f"ls -1 {save_path} | sort > tmpt")
    os.system(f"tar -cvf {tar_name}.tar -C {save_path} -T tmpt")
    os.system("rm tmpt")