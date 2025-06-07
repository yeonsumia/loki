import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch.nn.functional as F
import itertools
import random

class VectorDataset(Dataset):
    def __init__(self, directory, input_dim=17, xml_directory=None):
        self.directory = directory
        self.file_list = [file for file in os.listdir(directory) if file.endswith('.pt')]
        self.input_dim = input_dim
        self.xml_directory = xml_directory

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # print(f"Loading {self.file_list[idx]}")
        file_path = os.path.join(self.directory, self.file_list[idx])
        data = torch.load(file_path, weights_only=True)
        
        features = data[:, :self.input_dim]
        mask = data[:, self.input_dim:]
        return features, mask

class KeyVectorDataset(Dataset):
    def __init__(self, directory, input_dim=17):
        self.directory = directory
        self.file_list = [file for file in os.listdir(directory) if file.endswith('.pt')]
        self.input_dim = input_dim
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_list[idx])
        data = torch.load(file_path, weights_only=True)
        
        features = data[:, :self.input_dim]
        mask = data[:, self.input_dim:]
        return self.file_list[idx], features, mask
    
class RewardVectorDataset(VectorDataset):
    def __init__(self, reward_directory, vector_directory, input_dim, xml_directory=None):
        super(RewardVectorDataset, self).__init__(vector_directory, input_dim, xml_directory)
        self.file_list = [file for file in os.listdir(reward_directory) if file.endswith('.pt')] # Only load files that have rewards

        self.reward_directory = reward_directory

    def __getitem__(self, idx):
        features, mask = super(RewardVectorDataset, self).__getitem__(idx)

        reward_file_path = os.path.join(self.reward_directory, self.file_list[idx])
        reward = torch.load(reward_file_path)

        return features, mask, reward.mean()

class VectorSubset(Subset):
    def __init__(self, dataset, indices):
        super(VectorSubset, self).__init__(dataset, indices)

        self.directory = dataset.directory
        self.file_list = [dataset.file_list[i] for i in indices]
        self.input_dim = dataset.input_dim
        self.xml_directory = dataset.xml_directory

def get_dataloader(dataset, batch_size=32, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def random_split(dataset, lengths, generator=None):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(len(dataset), generator=generator).tolist()
    return [VectorSubset(dataset, indices[offset - length:offset]) for offset, length in zip(itertools.accumulate(lengths), lengths)]

def get_dataloaders(dataset, train_ratio=0.8, batch_size=32, shuffle=True, seed=42, save_dataset=True):

    # Determine the lengths of the splits
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len

    # Create the splits
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    # Save the train and val datasets
    if save_dataset:
        save_train_val_dataset(train_dataset, val_dataset)
    
    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader

def save_train_val_dataset(train_dataset, val_dataset):
    
    train_pt_dir = f'{train_dataset.directory}/train'
    val_pt_dir = f'{val_dataset.directory}/val'
    train_xml_dir = f'{train_dataset.xml_directory}/train'
    val_xml_dir = f'{val_dataset.xml_directory}/val'

    if not os.path.exists(train_pt_dir):
        os.makedirs(train_pt_dir)
    if not os.path.exists(val_pt_dir):
        os.makedirs(val_pt_dir)
    if not os.path.exists(train_xml_dir):
        os.makedirs(train_xml_dir)
    if not os.path.exists(val_xml_dir):
        os.makedirs(val_xml_dir)

    # Save train/val tensors
    for i, train_data in enumerate(train_dataset):
        torch.save(torch.cat(train_data, dim=-1), f'{train_pt_dir}/{train_dataset.file_list[i]}')

    for i, val_data in enumerate(val_dataset):
        torch.save(torch.cat(val_data, dim=-1), f'{val_pt_dir}/{val_dataset.file_list[i]}')
    
    # Save train/val xmls
    if train_dataset.xml_directory is not None:
        for i, file in enumerate(train_dataset.file_list):
            os.system(f'cp {train_dataset.xml_directory}/{file.replace(".pt", ".xml")} {train_xml_dir}/{file.replace(".pt", ".xml")}')
        for i, file in enumerate(val_dataset.file_list):
            os.system(f'cp {val_dataset.xml_directory}/{file.replace(".pt", ".xml")} {val_xml_dir}/{file.replace(".pt", ".xml")}')