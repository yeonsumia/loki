import argparse
import torch
import os
import json
import tarfile

from sklearn.cluster import KMeans

from torch.nn.functional import one_hot
from vae.data import get_dataloader, KeyVectorDataset
from vae.model import Model_VAE, Encoder_model
from tools.util import D_TOKEN, N_TOKENS, BINARY_TOKEN, CONTINUOUS_TOKEN, CATEGORY_TOKEN, TOTAL_CATEGORY, THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS
from vae.train import H_DIM, D_DEPTH, N_HEAD, FACTOR, NUM_LAYERS
from tools.util import validity_check
from vae.webdata import get_webdataloader, data_decoder, TensorWebDataset, FilteredTensorWebDataset


D_DEPTH = 32
H_DIM = 32
N_HEAD = 4
FACTOR = 8
NUM_LAYERS = 4


def kmeans(args, latents):
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed).fit(latents)
    return kmeans

def load_latent_codes(args):
    BATCH_SIZE = 4096

    if args.make_cluster:
        TEST_PATH = 'webdataset.tar'
        dataset = FilteredTensorWebDataset(TEST_PATH, input_dim=D_TOKEN, filter=lambda sample: True).decode(data_decoder)
        dataloader = get_webdataloader(dataset, batch_size=BATCH_SIZE)
    elif args.assign_cluster_wds:
        TEST_PATH = 'webdataset.tar'
        train_ratio = 0.9
        mod = int(1/(1-train_ratio))
        dataset = FilteredTensorWebDataset(TEST_PATH, input_dim=D_TOKEN, filter=lambda sample: int(sample['__key__']) % mod == 0).decode(data_decoder)
        dataloader = get_webdataloader(dataset, batch_size=BATCH_SIZE)
    elif args.assign_cluster:
        TEST_PATH = os.path.join(args.data_dir, "unimal_init_vec")
        dataset = KeyVectorDataset(directory=TEST_PATH, input_dim=D_TOKEN)
        print(len(dataset))
        dataloader = get_dataloader(dataset, batch_size=BATCH_SIZE)
    else:
        # load cluster
        TEST_PATH = f'data/latent_cluster{args.n_clusters}_{args.cluster_label}.tar'
        dataset = FilteredTensorWebDataset(TEST_PATH, input_dim=D_TOKEN, filter=lambda sample: True).decode(data_decoder)
        dataloader = get_webdataloader(dataset, batch_size=BATCH_SIZE)
    
    # load model
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    LOAD_PATH = f'{curr_dir}/checkpoints/{args.ckpt_dir}/model.pt' # trained model
    device = args.device # model device
    model = Model_VAE(NUM_LAYERS, N_TOKENS, D_TOKEN, D_DEPTH, H_DIM, n_head = N_HEAD, factor = FACTOR)
    model.load_state_dict(torch.load(LOAD_PATH))

    model = model.to(device)

    # evaluate
    model.eval()

    seq_list = []
    result = []
    latent = []
    keys = []
    count_total = 0

    with torch.no_grad():
        for i, (key, x, mask) in enumerate(dataloader):
            x = x.float().to(device)
            mask = mask.float().to(device)
            # save xml strings in file
            # print(key, x.shape, mask.shape, len(xml), len(pkl))

            # Replace the dummy values with 0
            origin_x = x = x.masked_fill(mask == 0, 0)
            
            # Split the numerical and depth features
            x_con, x_cat, x_binary, x_depth = x.split([CONTINUOUS_TOKEN, CATEGORY_TOKEN, BINARY_TOKEN, 1], dim=-1)
            mask_num, mask_depth = mask.split([D_TOKEN - 1, 1], dim=-1)

            num_classes_list = [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS]

            one_hot_x_cat = torch.cat([one_hot(x_cat[:, :, i].long(), num_classes=num_classes_list[i]) for i in range(CATEGORY_TOKEN)], dim=-1)

            x_num = torch.cat([x_con, one_hot_x_cat, x_binary], dim=-1)
            
            # for raw-parameter-cluster
            # x = torch.cat([x_num, x_depth], dim=-1)

            Recon_X_con, Recon_X_cat, Recon_X_depth, mu_z, std_z = model(x_num, x_depth)

            latent_i = model.VAE.reparameterize(mu_z, std_z)
            
            Recon_X_category, Recon_X_binary = torch.split(Recon_X_cat, [TOTAL_CATEGORY, BINARY_TOKEN], dim=-1)
            Recon_X_binary = (Recon_X_binary > 0).int()
            
            Recon_X_category_theta, Recon_X_category_phi, Recon_X_category_jointx, Recon_X_category_jointy = \
                  torch.split(Recon_X_category, [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS], dim=-1)

            Recon_X_category_theta = torch.argmax(Recon_X_category_theta, dim=-1).unsqueeze(-1)
            Recon_X_category_phi = torch.argmax(Recon_X_category_phi, dim=-1).unsqueeze(-1)
            Recon_X_category_jointx = torch.argmax(Recon_X_category_jointx, dim=-1).unsqueeze(-1)
            Recon_X_category_jointy = torch.argmax(Recon_X_category_jointy, dim=-1).unsqueeze(-1)

            Recon_X_category = torch.cat((Recon_X_category_theta, Recon_X_category_phi, Recon_X_category_jointx, Recon_X_category_jointy), dim=-1)

            Recon_X_depth = torch.argmax(Recon_X_depth, dim=-1).unsqueeze(-1) # [batch_size, depth_categories(=max_limbs), max_limbs] -> [batch_size, max_limbs, 1]

            result_i = torch.cat((Recon_X_con, Recon_X_category, Recon_X_binary, Recon_X_depth), dim=-1)
            result_i_valid = []
            latent_i_valid = []
            key_i_valid = []
            

            for j in range(x_binary.shape[0]):
                try:
                    seq_len = validity_check(x[j], x_depth[j])
                    if seq_len < 4:
                        continue
                    seq_list.append(seq_len)
                    result_i_valid.append(result_i[j].unsqueeze(0))
                    latent_i_valid.append(latent_i[j].unsqueeze(0))
                    key_i_valid.append(key[j])
                except ValueError as e:
                    pass
                    print(e)
                    # return None, None
            if len(result_i_valid) > 0:
                result_i_valid = torch.cat(result_i_valid, dim=0)
                latent_i_valid = torch.cat(latent_i_valid, dim=0)
                if args.make_cluster or args.assign_cluster_wds or args.assign_cluster:
                    print(f"result_{i}_valid shape: {result_i_valid.shape}")
                    print(f"latent_{i}_valid shape: {latent_i_valid.shape}")
                    print(f"key_{i}_valid length: {len(key_i_valid)}")
                result.append(result_i_valid)
                latent.append(latent_i_valid)
                keys.extend(key_i_valid)
            
            ################################################
            # for raw-parameter-cluster
            # for j in range(x_binary.shape[0]):
            #     try:
            #         seq_len = validity_check(origin_x[j], x_depth[j])
            #         if seq_len < 4:
            #             continue
            #     except ValueError as e:
            #         pass
            #         print(e)
            #     seq_list.append(seq_len)
            #     # result_i_valid.append(result_i[j].unsqueeze(0))
            #     latent_i_valid.append(x[j].unsqueeze(0))
            #     key_i_valid.append(key[j])

            # if len(latent_i_valid) > 0:
            #     # result_i_valid = torch.cat(result_i_valid, dim=0)
            #     latent_i_valid = torch.cat(latent_i_valid, dim=0)
            #     if args.make_cluster or args.assign_cluster_wds or args.assign_cluster:
            #         # print(f"result_{i}_valid shape: {result_i_valid.shape}")
            #         print(f"latent_{i}_valid shape: {latent_i_valid.shape}")
            #         print(f"key_{i}_valid length: {len(key_i_valid)}")
            #     result.append(x)
            #     latent.append(latent_i_valid)
            #     keys.extend(key_i_valid)
                    
            # print("x shape:", x.shape)
            # latent.append(x)
            # keys.extend(key)
            # print("keys:", len(keys))
            ################################################
            if args.num_samples > 0:
                count_total += x.shape[0]
                if count_total >= args.num_samples:
                    break

    # result = torch.cat(result, dim=0)
    latent = torch.cat(latent, dim=0)

    # random permutation of len(latent)
    # indices = torch.randperm(latent.size(0))
    # latent = latent[indices[:50000]] # using only 50K samples for clustering
    # latent = latent[:50000] # using only 50K samples for clustering

    return keys, latent, seq_list


def assign_to_cluster(args, latents, key_list, xml_dir, pkl_dir, data_dir):
    # cluster_labels_path = os.path.join(args.save_dir, f"cluster{args.n_clusters}_labels.json")
    # if os.path.exists(cluster_labels_path):
    #     cluster_labels = fu.load_json(cluster_labels_path)
    #     print(cluster_labels['418519'])
    #     print(cluster_labels['34321'])
    #     print(cluster_labels['90770'])
    #     return
    # load cluster mean
    cluster_means_path = os.path.join(args.save_dir, f"cluster{args.n_clusters}_means.pt")
    cluster_means = torch.load(cluster_means_path)
    cluster_means = cluster_means.to(args.device)
    print(f"Cluster means shape: {cluster_means.shape}")

    # flatten the cluster_means
    cluster_means = cluster_means.view(cluster_means.size(0), -1)

    # find the closest cluster mean
    latents = latents.to(args.device)
    latents = latents.unsqueeze(1)
    cluster_means = cluster_means.unsqueeze(0)
    distances = torch.norm(latents - cluster_means, dim=-1)
    cluster_labels = torch.argmin(distances, dim=-1)
    print(cluster_labels.shape)

    # Save xmls to cluster directory
    move_count = 0
    for i, key in enumerate(key_list):
        label = cluster_labels[i].item()
        assert label < args.n_clusters and label >= 0
        xml_src = os.path.join(xml_dir, f"{key}.xml")
        pkl_src = os.path.join(pkl_dir, f"{key}.pkl")
        data_src = os.path.join(data_dir, f"{key}.pt")
        xml_dst = pkl_dst = data_dst = os.path.join(args.save_dir, str(label))
        if os.path.exists(xml_src):
            os.system(f"mv {xml_src} {xml_dst}")
            move_count += 1

        if os.path.exists(pkl_src):
            os.system(f"mv {pkl_src} {pkl_dst}")
            move_count += 1

        if os.path.exists(data_src):
            os.system(f"mv {data_src} {data_dst}")
            move_count += 1

        if (i + 1) % 50000 == 0:
            print(f"[{i+i}] Moved {move_count} files")


def _cluster(args):
    key_list, latents, seq_list = load_latent_codes(args)
    # print sequence length distribution
    if args.make_cluster or args.assign_cluster_wds or args.assign_cluster:
        print("Sequence length distribution:")
    else:
        print(f"Cluster {args.cluster_label} sequence length distribution:")
    seq_counts = {}
    for seq_len in seq_list:
        if str(seq_len) not in seq_counts:
            seq_counts[str(seq_len)] = 0
        seq_counts[str(seq_len)] += 1
    
    for seq_len, count in seq_counts.items():
        print(f"Sequence length {seq_len}: {count} samples")

    # save sequence length distribution to json file
    if args.make_cluster:
        with open(os.path.join(args.save_dir, f"cluster{args.n_clusters}_sequence_length_distribution.json"), "w") as f:
            json.dump(seq_counts, f)
    elif args.assign_cluster or args.assign_cluster_wds:
        pass
    else:
        with open(os.path.join(args.cluster_dir, f"cluster{args.cluster_label}_sequence_length_distribution.json"), "w") as f:
            json.dump(seq_counts, f)

    print(f"Latents shape: {latents.shape}")
    seq_len, latent_dim = latents.shape[1], latents.shape[2]
    # flatten the latents
    latents = latents.view(latents.size(0), -1)
    print(f"Flattened latents shape: {latents.shape}")

    if not args.make_cluster and not args.assign_cluster and not args.assign_cluster_wds:
        return

    if args.assign_cluster_wds:
        assign_to_cluster(args, 
                          latents, 
                          key_list, 
                          xml_dir="webdataset/ft", 
                          pkl_dir="webdataset/ft", 
                          data_dir="webdataset/ft")
        return
    elif args.assign_cluster:
        assign_to_cluster(args, 
                          latents, 
                          key_list, 
                          xml_dir=os.path.join(args.data_dir, "xml"), 
                          pkl_dir=os.path.join(args.data_dir, "unimal_init"), 
                          data_dir=os.path.join(args.data_dir, "unimal_init_vec"))
        return
    
    assert args.make_cluster
    # make clusters
    # latents to cpu
    latents = latents.cpu().numpy()

    kmeans_model = kmeans(args, latents)
    print(len(kmeans_model.labels_))
    print(f"Cluster labels range: {min(kmeans_model.labels_)}-{max(kmeans_model.labels_)}")
    cluster_means = torch.tensor(kmeans_model.cluster_centers_).view(kmeans_model.cluster_centers_.shape[0], seq_len, latent_dim)
    print(f"Cluster means shape: {cluster_means.shape}")

    # print cluster label distribution
    cluster_counts = {}
    for label in kmeans_model.labels_:
        if str(label) not in cluster_counts:
            cluster_counts[str(label)] = 0
        cluster_counts[str(label)] += 1

    print("Cluster label distribution:")
    for label, count in cluster_counts.items():
        print(f"Cluster {label}: {count} samples")

    # save cluster label distribution to json file
    with open(os.path.join(args.save_dir, f"cluster{args.n_clusters}_label_distribution.json"), "w") as f:
        json.dump(cluster_counts, f)

    # Save cluster means
    cluster_means_path = os.path.join(args.save_dir, f"cluster{args.n_clusters}_means.pt")
    torch.save(cluster_means, cluster_means_path)

    # Save cluster labels to json file
    cluster_labels_path = os.path.join(args.save_dir, f"cluster{args.n_clusters}_labels.json")
    cluster_labels = {key: int(label) for key, label in zip(key_list, kmeans_model.labels_)}
    with open(cluster_labels_path, "w") as f:
        json.dump(cluster_labels, f)

    # return
    # Save the cluster xmls
    move_count = 0
    cluster_tar_set = {}
    tar_set = {}
    for i in range(args.n_clusters):
        cluster_tar_set[i] = []
    
    with tarfile.open('webdataset.tar', 'r') as tar:
        members = tar.getmembers()  # Get all files and directories in the tar
        print(f"Total files: {len(members)}")

        members_key = [os.path.splitext(os.path.basename(member.name))[0] for member in members]

        for k in members_key:
            tar_set[k] = []
        for member in members:
            key = os.path.splitext(os.path.basename(member.name))[0]
            tar_set[key].append(member)
        
        print(f"Total keys: {len(tar_set)}")
        
        for i, key in enumerate(key_list):
            label = kmeans_model.labels_[i]
            assert label < args.n_clusters and label >= 0
            if key in members_key:
                if len(tar_set[key]) == 3:
                    cluster_tar_set[label].extend(tar_set[key])
                move_count += 1
            if (i + 1) % 50000 == 0:
                print(f"[{i+i}] Moved {move_count} files")
    
        # create tar files for each cluster
        os.makedirs("data/", exist_ok=True)
        for i in range(args.n_clusters):
            with tarfile.open(f'data/latent_cluster{args.n_clusters}_{i}.tar', 'w') as cluster_i_tar:
                for file in cluster_tar_set[i]:
                    cluster_i_tar.addfile(file, tar.extractfile(file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--base_dir', type=str, help='Directory of the morphologies.')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters.')
    parser.add_argument('--ckpt_dir', type=str, help='The path to the trained vae model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to use for clustering.')
    parser.add_argument('--make_cluster', action='store_true', help='Make cluster from new webdataset.')
    parser.add_argument('--assign_cluster', action='store_true', help='Assign cluster to existing dataset.')
    parser.add_argument('--assign_cluster_wds', action='store_true', help='Assign cluster to existing webdataset.')
    parser.add_argument('--cluster_label', type=int, default=0, help='Cluster label.')
    parser.add_argument('--data_dir', type=str, help='Directory of the data to assign cluster.')
    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    # make save directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(cur_dir, f"latent_cluster{args.n_clusters}_new_webdataset")
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    
    if not args.make_cluster and not args.assign_cluster and not args.assign_cluster_wds:
        args.cluster_dir = os.path.join(save_dir, str(args.cluster_label))
        os.makedirs(args.cluster_dir, exist_ok=True)
    
    with torch.no_grad():
        _cluster(args)
