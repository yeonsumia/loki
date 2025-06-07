import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
import time
import wandb

from tools.util import D_TOKEN, N_TOKENS, BINARY_TOKEN, CONTINUOUS_TOKEN, CATEGORY_TOKEN, TOTAL_CATEGORY, THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS
from vae.model import Model_VAE, Encoder_model, Decoder_model
from vae.data import get_dataloader, get_dataloaders, VectorDataset, save_train_val_dataset
from vae.webdata import get_webdataloader, data_decoder, FilteredTensorWebDataset
from torch.nn.functional import one_hot


warnings.filterwarnings('ignore')

# Model params
N_HEAD = 4
FACTOR = 8
NUM_LAYERS = 4
D_DEPTH = 32
H_DIM = 32


def plot_loss(train_losses, val_losses, save_path=""):
    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path)


def compute_loss(X_continuous, X_cat, X_binary, X_depth, Recon_X_continuous, Recon_X_cat, Recon_X_depth, mu_z, logvar_z, mask):
    criterion_num = nn.MSELoss(reduction='none')
    criterion_depth = nn.CrossEntropyLoss(reduction='none')
    criterion_binary = nn.BCEWithLogitsLoss(reduction='none')
    criterion_categorical = nn.CrossEntropyLoss(reduction='none')

    # X_num_binary = [jointx, jointy, site, mode, EOS]
    # Separate the binary and non-binary features
    mask_continuous, mask_category, mask_binary = torch.split(mask, [CONTINUOUS_TOKEN, CATEGORY_TOKEN, BINARY_TOKEN], dim=-1)
    Recon_X_num_category, Recon_X_num_binary = torch.split(Recon_X_cat, [TOTAL_CATEGORY, BINARY_TOKEN], dim=-1)

    Recon_X_num_category_theta, Recon_X_num_category_phi, Recon_X_num_category_jointx, Recon_X_num_category_jointy = \
        torch.split(Recon_X_num_category, [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS], dim=-1)
    X_num_category_theta = X_cat[:, :, 0].long()
    X_num_category_phi = X_cat[:, :, 1].long()
    X_num_category_jointx = X_cat[:, :, 2].long()
    X_num_category_jointy = X_cat[:, :, 3].long()
    mask_category_theta = mask_category[:, :, 0]
    mask_category_phi = mask_category[:, :, 1]
    mask_category_jointx = mask_category[:, :, 2]
    mask_category_jointy = mask_category[:, :, 3]

    loss_continuous = criterion_num(Recon_X_continuous, X_continuous)
    loss_category_theta = criterion_categorical(Recon_X_num_category_theta.permute(0, 2, 1), X_num_category_theta) # CrossEntropyLoss
    loss_category_phi = criterion_categorical(Recon_X_num_category_phi.permute(0, 2, 1), X_num_category_phi) # CrossEntropyLoss
    loss_category_jointx = criterion_categorical(Recon_X_num_category_jointx.permute(0, 2, 1), X_num_category_jointx) # CrossEntropyLoss
    loss_category_jointy = criterion_categorical(Recon_X_num_category_jointy.permute(0, 2, 1), X_num_category_jointy) # CrossEntropyLoss
    
    loss_binary = criterion_binary(Recon_X_num_binary, X_binary)
    loss_depth = criterion_depth(Recon_X_depth.permute(0, 2, 1), X_depth.long().squeeze(-1)) # CrossEntropyLoss

    # Apply the mask to the losses
    loss_continuous = (loss_continuous * mask_continuous).sum() / mask_continuous.sum()
    loss_category_theta = (loss_category_theta * mask_category_theta).sum() / mask_category_theta.sum()
    loss_category_phi = (loss_category_phi * mask_category_phi).sum() / mask_category_phi.sum()
    loss_category_jointx = (loss_category_jointx * mask_category_jointx).sum() / mask_category_jointx.sum()
    loss_category_jointy = (loss_category_jointy * mask_category_jointy).sum() / mask_category_jointy.sum()
    loss_category = loss_category_theta + loss_category_phi + loss_category_jointx + loss_category_jointy / 4
    loss_binary = (loss_binary * mask_binary).sum() / mask_binary.sum()
    loss_depth = (loss_depth).mean()
    
    # print(mu_z.shape, logvar_z.shape)

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    
    # print(loss_continuous, loss_category, loss_binary, loss_depth, loss_kld)
    
    return loss_continuous, loss_category, loss_binary, loss_depth, loss_kld


def train(args):
    N_HEAD = args.n_head
    FACTOR = args.factor
    NUM_LAYERS = args.n_layer
    D_DEPTH = args.d_depth
    H_DIM = args.h_dim
    print(f"D_DEPTH: {D_DEPTH}")
    print(f"H_DIM: {H_DIM}")
    print(f"N_HEAD: {N_HEAD}")
    print(f"FACTOR: {FACTOR}")
    print(f"NUM_LAYERS: {NUM_LAYERS}")

    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    WD = args.wd
    if args.wds:
        DATA_PATH = 'webdataset.tar'
    else:
        DATA_PATH = f'derl/{args.data_dir}/ft/unimal_init_vec'
    XML_PATH = f'derl/{args.data_dir}/ft/xml'
    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    device =  args.device

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/checkpoints/{args.ckpt_dir}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    print(model_save_path)
    
    # Create the dataset
    train_ratio = 0.9
    if args.wds:
        mod = int(1/(1-train_ratio))
        train_dataset = FilteredTensorWebDataset(DATA_PATH, input_dim=D_TOKEN, filter=lambda sample: int(sample['__key__']) % mod != 0).decode(data_decoder)
        train_dataloader = get_webdataloader(train_dataset, batch_size=BATCH_SIZE)
        val_dataset = FilteredTensorWebDataset(DATA_PATH, input_dim=D_TOKEN, filter=lambda sample: int(sample['__key__']) % mod == 0).decode(data_decoder)
        val_dataloader = get_webdataloader(val_dataset, batch_size=BATCH_SIZE)
    else:
        dataset = VectorDataset(directory=DATA_PATH, input_dim=D_TOKEN, xml_directory=XML_PATH)
        train_dataloader, val_dataloader = get_dataloaders(dataset, batch_size=BATCH_SIZE, shuffle=True, seed=args.seed, train_ratio=train_ratio, save_dataset=args.save_data)

    # Load VAE model
    model = Model_VAE(NUM_LAYERS, N_TOKENS, D_TOKEN, D_DEPTH, H_DIM, n_head = N_HEAD, factor = FACTOR, attention_dropout=args.attention_dropout, ffn_dropout=args.ffn_dropout).to(device)
    model = model.to(device)

    pre_encoder = Encoder_model(NUM_LAYERS, D_DEPTH, H_DIM, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, D_DEPTH, H_DIM, n_head = N_HEAD, factor = FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    # Define your optimizer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    train_losses = []
    train_num_losses = []
    train_binary_losses = []
    train_category_losses = []
    train_depth_losses = []
    val_losses = []
    val_binary_losses = []
    val_category_losses = []
    val_num_losses = []
    val_depth_losses = []

    start_time = time.time()
    cur_iteration = 0
    for epoch in range(N_EPOCHS):
        
        print(f"Epoch {epoch+1}/{N_EPOCHS}")

        curr_loss_recons = 0.0
        curr_loss_binary = 0.0
        curr_loss_category = 0.0
        curr_loss_num = 0.0
        curr_loss_depth = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        # for x, mask in train_dataloader:
        for iteration, (_, x, mask) in enumerate(train_dataloader):
            # print(iteration)
            # x_num: (batch_size, N_TOKENS, D_TOKEN)
            x = x.float().to(device)
            mask = mask.float().to(device)
            
            # Replace the dummy values with 0
            x = x.masked_fill(mask == 0, 0)
            
            # Split the numerical and depth features
            x_con, x_cat, x_binary, x_depth = x.split([CONTINUOUS_TOKEN, CATEGORY_TOKEN, BINARY_TOKEN, 1], dim=-1)
            mask_num, mask_depth = mask.split([D_TOKEN - 1, 1], dim=-1)
            
            model.train()
            optimizer.zero_grad()

            # make x_cat one-hot encoding for each feature
            num_classes_list = [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS]

            one_hot_x_cat = torch.cat([one_hot(x_cat[:, :, i].long(), num_classes=num_classes_list[i]) for i in range(CATEGORY_TOKEN)], dim=-1)

            x_num = torch.cat([x_con, one_hot_x_cat, x_binary], dim=-1)

            Recon_X_con, Recon_X_cat, Recon_X_depth, mu_z, std_z = model(x_num, x_depth)
        
            loss_num, loss_category, loss_binary, loss_depth, loss_kld = compute_loss(x_con, x_cat, x_binary, x_depth, Recon_X_con, Recon_X_cat, Recon_X_depth, mu_z, std_z, mask_num)

            loss_recons = loss_num + loss_category + loss_binary + loss_depth

            loss = loss_recons + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = x_con.shape[0]
            curr_count += batch_length
            curr_loss_binary += loss_binary.item() * batch_length
            curr_loss_category += loss_category.item() * batch_length
            curr_loss_depth += loss_depth.item() * batch_length
            curr_loss_num += loss_num.item() * batch_length
            curr_loss_recons += loss_recons.item() * batch_length
            curr_loss_kl    += loss_kld.item() * batch_length

            # Log metrics every 100 iterations
            if iteration % 10 == 0:
                wandb.log({"Iteration Loss Binary": loss_binary.item(),
                        "Iteration Loss Depth": loss_depth.item(),
                        "Iteration Loss Num": loss_num.item(),
                        "Iteration Loss Category": loss_category.item(),
                        "Iteration Loss Reconstruction": loss_recons.item(),
                        "Iteration Loss KL": loss_kld.item()})
            

        recons_loss = curr_loss_recons / curr_count
        binary_loss = curr_loss_binary / curr_count
        category_loss = curr_loss_category / curr_count
        depth_loss = curr_loss_depth / curr_count
        num_loss = curr_loss_num / curr_count
        kl_loss = curr_loss_kl / curr_count
        
        if epoch % 10 != 0:
            continue

        '''
            Evaluation
        '''
        model.eval()
        total_val_loss = 0
        total_val_binary_loss = 0
        total_val_num_loss = 0
        total_val_depth_loss = 0
        total_val_category_loss = 0
        total_val_kl_loss = 0
        val_samples = 0

        print("Validation")
        with torch.no_grad():
            for _, X_test, mask_test in val_dataloader:
                X_test = X_test.float().to(device)
                mask_test = mask_test.float().to(device)
                
                X_test = X_test.masked_fill(mask_test == 0, 0)

                X_test_con, X_test_cat, X_test_binary, X_test_depth = X_test.split([CONTINUOUS_TOKEN, CATEGORY_TOKEN, BINARY_TOKEN, 1], dim=-1)
                mask_test_num, mask_test_depth = mask_test.split([D_TOKEN - 1, 1], dim=-1)

                # make x_cat one-hot encoding for each feature
                num_classes_list = [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS]

                one_hot_X_test_cat = torch.cat([one_hot(X_test_cat[:, :, i].long(), num_classes=num_classes_list[i]) for i in range(CATEGORY_TOKEN)], dim=-1)

                X_test_num = torch.cat([X_test_con, one_hot_X_test_cat, X_test_binary], dim=-1)


                Recon_X_con, Recon_X_cat, Recon_X_depth, mu_z, std_z = model(X_test_num, X_test_depth)
                

                val_num_loss, val_category_loss, val_binary_loss, val_depth_loss, val_kl_loss = compute_loss(X_test_con, X_test_cat, X_test_binary, X_test_depth, Recon_X_con, Recon_X_cat, Recon_X_depth, mu_z, std_z, mask_test_num)
                val_recons_loss = val_num_loss + val_category_loss + val_binary_loss + val_depth_loss
                val_loss = val_recons_loss.item()

                total_val_loss += val_loss * X_test_con.size(0)
                total_val_binary_loss += val_binary_loss.item() * X_test_con.size(0)
                total_val_category_loss += val_category_loss.item() * X_test_con.size(0)
                total_val_num_loss += val_num_loss.item() * X_test_con.size(0)
                total_val_depth_loss += val_depth_loss.item() * X_test_con.size(0)
                val_samples += X_test_con.size(0)

        average_val_loss = total_val_loss / val_samples
        average_val_binary_loss = total_val_binary_loss / val_samples
        average_val_category_loss = total_val_category_loss / val_samples
        average_val_depth_loss = total_val_depth_loss / val_samples
        average_val_num_loss = total_val_num_loss / val_samples

        scheduler.step(average_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != current_lr:
            current_lr = new_lr
            print(f"Learning rate updated: {current_lr}")

        if average_val_loss < best_train_loss:
            best_train_loss = average_val_loss
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience == 10:
                if beta > min_beta:
                    beta = beta * lambd


        print('epoch: {}, beta = {:.6f}, Train Num: {:.6f}, Train Category: {:.6f}, Train Binary: {:.6f}, Train Depth: {:.6f}, Train Reconstruction: {:.6f}, Train KL:{:.6f}, Val Reconstruction:{:.6f}'.format(epoch, beta, num_loss, category_loss, binary_loss, depth_loss, recons_loss, kl_loss, val_recons_loss.item()), flush=True)
        train_losses.append(recons_loss)
        train_binary_losses.append(binary_loss)
        train_category_losses.append(category_loss)
        train_depth_losses.append(depth_loss)
        train_num_losses.append(num_loss)
        val_losses.append(average_val_loss)
        val_binary_losses.append(average_val_binary_loss)
        val_category_losses.append(average_val_category_loss)
        val_num_losses.append(average_val_num_loss)
        val_depth_losses.append(average_val_depth_loss)
        
        
        wandb.log({
                #    "Train Numerical Loss:": num_loss, 
                #    "Train Binary Loss": binary_loss,
                #    "Train Depth Loss": depth_loss,
                #    "Train Reconstruction Loss": recons_loss,
                #    "Train KL Loss": kl_loss,
                   "Val Binary Loss": average_val_binary_loss,
                   "Val Numerical Loss": average_val_num_loss,
                   "Val Depth Loss": average_val_depth_loss,
                   "Val Category Loss": average_val_category_loss,
                   "Val Reconstruction Loss": average_val_loss,})
        
        if epoch % 100 == 0:
            plot_loss(train_losses, val_losses, save_path=f'{ckpt_dir}/recons_loss_plot.png')
            plot_loss(train_binary_losses, val_binary_losses, save_path=f'{ckpt_dir}/bianry_loss_plot.png')
            plot_loss(train_category_losses, val_category_losses, save_path=f'{ckpt_dir}/category_loss_plot.png')
            plot_loss(train_depth_losses, val_depth_losses, save_path=f'{ckpt_dir}/depth_loss_plot.png')
            plot_loss(train_num_losses, val_num_losses, save_path=f'{ckpt_dir}/num_loss_plot.png')
            model_tmp_save_path = f'{ckpt_dir}/model_{epoch}.pt'
            torch.save(model.state_dict(), model_tmp_save_path)

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    
    # Saving latent embeddings
    print('Saving latent embeddings...')
    save_latent_embeddings(args)

    
def save_latent_embeddings(args):
    N_HEAD = args.n_head
    FACTOR = args.factor
    NUM_LAYERS = args.n_layer
    D_DEPTH = args.d_depth
    H_DIM = args.h_dim
    print(f"D_DEPTH: {D_DEPTH}")
    print(f"H_DIM: {H_DIM}")
    print(f"N_HEAD: {N_HEAD}")
    print(f"FACTOR: {FACTOR}")
    print(f"NUM_LAYERS: {NUM_LAYERS}")

    train_ratio = 0.9
    BATCH_SIZE = args.batch_size
    if args.wds:
        DATA_PATH = 'webdataset.tar'
        mod = int(1/(1-train_ratio))
        train_dataset = FilteredTensorWebDataset(DATA_PATH, input_dim=D_TOKEN, filter=lambda sample: int(sample['__key__']) % mod != 0).decode(data_decoder)
        train_dataloader = get_webdataloader(train_dataset, batch_size=BATCH_SIZE)
    else:
        DATA_PATH = f'derl/{args.data_dir}/ft/unimal_init_vec'
        XML_PATH = f'derl/{args.data_dir}/ft/xml'
        dataset = VectorDataset(directory=DATA_PATH, input_dim=D_TOKEN, xml_directory=XML_PATH)
        train_dataloader, _ = get_dataloaders(dataset, batch_size=BATCH_SIZE, shuffle=True, seed=args.seed, train_ratio=0.9, save_dataset=args.save_data)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/checkpoints/{args.ckpt_dir}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'
    print(model_save_path)

    device = args.device

    with torch.no_grad():

        model = Model_VAE(NUM_LAYERS, N_TOKENS, D_TOKEN, D_DEPTH, H_DIM, n_head = N_HEAD, factor = FACTOR, attention_dropout=args.attention_dropout, ffn_dropout=args.ffn_dropout).to(device)
        model = model.to(device)
        model.eval()
        model.load_state_dict(torch.load(model_save_path))
        
        pre_encoder = Encoder_model(NUM_LAYERS, D_DEPTH, H_DIM, n_head = N_HEAD, factor = FACTOR).to(device)
        pre_decoder = Decoder_model(NUM_LAYERS, D_DEPTH, H_DIM, n_head = N_HEAD, factor = FACTOR).to(device)

        pre_encoder.eval()
        pre_decoder.eval()
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        print('Successfully load and save the model!')

        all_train_z_mu = []
        all_train_z_sigma = []
        all_train_z = []

        for _, X_train, mask_train in train_dataloader:
            X_train = X_train.float().to(device)
            mask_train = mask_train.float().to(device)

            X_train = X_train.masked_fill(mask_train == 0, 0)
            
            X_con, X_cat, X_binary, X_depth = X_train.split([CONTINUOUS_TOKEN, CATEGORY_TOKEN, BINARY_TOKEN, 1], dim=-1)

            mask_num, mask_depth = mask_train.split([D_TOKEN - 1, 1], dim=-1)

            # train_z = pre_encoder(X_train_num, X_train_depth).detach().cpu().numpy()
            num_classes_list = [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS]

            one_hot_X_cat = torch.cat([one_hot(X_cat[:, :, i].long(), num_classes=num_classes_list[i]) for i in range(CATEGORY_TOKEN)], dim=-1)

            X_num = torch.cat([X_con, one_hot_X_cat, X_binary], dim=-1)

            Recon_X_con, Recon_X_cat, Recon_X_depth, mu_z, std_z = model(X_num, X_depth)
            z = model.VAE.reparameterize(mu_z, std_z)
            all_train_z_mu.append(mu_z.detach().cpu().numpy())
            all_train_z_sigma.append(std_z.detach().cpu().numpy())
            all_train_z.append(z.detach().cpu().numpy())
            
            # double-check loss
            loss_num, loss_category, loss_binary, loss_depth, loss_kld = compute_loss(X_con, X_cat, X_binary, X_depth, Recon_X_con, Recon_X_cat, Recon_X_depth, mu_z, std_z, mask_num)
            print("loss_num: ", loss_num.item(), ", loss_category: ", loss_category.item(), ", loss_binary: ", loss_binary.item(), ", loss_depth: ", loss_depth.item(), ", loss_kld: ", loss_kld.item())
        
        all_train_z = np.concatenate(all_train_z, axis=0)
        all_train_z_sigma = np.concatenate(all_train_z_sigma, axis=0)
        all_train_z_mu = np.concatenate(all_train_z_mu, axis=0)

        np.save(f'{ckpt_dir}/train_z.npy', all_train_z)
        np.save(f'{ckpt_dir}/train_z_mu.npy', all_train_z_mu)
        np.save(f'{ckpt_dir}/train_z_sigma.npy', all_train_z_sigma)

        print('Successfully save pretrained embeddings in disk!')


def main():
    parser = argparse.ArgumentParser(description='VAE training configuration')
    
    parser.add_argument('--wds', action='store_true', help='Use webdataset.')
    parser.add_argument('--save_latent', action='store_true', help='Save the latent vectors.')
    parser.add_argument('--gpu', type=int, default=1, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')
    parser.add_argument('--save_data', action='store_true', help='Save the train/validation pickle/xml data.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight Decay.')
    parser.add_argument('--attention_dropout', type=float, default=0., help='Attention dropout.')
    parser.add_argument('--ffn_dropout', type=float, default=0., help='FFN dropout.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to split dataset.')
    parser.add_argument('--data_dir', type=str, default='output', help='Directory of the data.')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='Directory of the data.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of heads in the multiheadattention models.')
    parser.add_argument('--factor', type=int, default=8, help='Factor of the model dimension.')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers in the model.')
    parser.add_argument('--d_depth', type=int, default=16, help='Dimension of depth.')
    parser.add_argument('--h_dim', type=int, default=16, help='Dimension of hidden layer.')

    args = parser.parse_args()
    
    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    
    args.ckpt_dir = f"VAE_500k_hdim{args.h_dim}_depth{args.d_depth}_LR_{args.lr}_WD_{args.wd}_L{args.n_layer}_H{args.n_head}_F{args.factor}_beta{args.max_beta}_bsize{args.batch_size}_epochs{args.epochs}"
    wandb.init(project="VAE", name=args.ckpt_dir)

    if args.save_latent:
        save_latent_embeddings(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
