import os, sys
import webdataset as wds
import torch
import io
import pickle

from tools.util import D_TOKEN

def data_decoder(key, value):
    if key.endswith(".pt"):
        return torch.load(io.BytesIO(value), weights_only=True)
    elif key.endswith(".xml"):
        return value.decode("utf-8")
    else:
        return value
    
class FilteredTensorWebDataset(wds.WebDataset):
    def __init__(self, *args, input_dim, filter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.filter = filter
        
    def __iter__(self):
        # sample: dict_keys(['__key__', '__url__', 'pkl', 'pt', 'xml'])
        for sample in super().__iter__():
            if '__key__' not in sample or 'pkl' not in sample or 'pt' not in sample or 'xml' not in sample:
                continue
            if self.filter(sample):
                xml = sample["xml"]
                features = sample["pt"][:, :self.input_dim]
                mask = sample["pt"][:, self.input_dim:]
                yield sample["__key__"], features, mask

class TensorWebDataset(wds.WebDataset):
    def __init__(self, *args, input_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        
    def __iter__(self):
        # sample: dict_keys(['__key__', '__url__', 'pkl', 'pt', 'xml'])
        for sample in super().__iter__():
            xml = sample["xml"]
            features = sample["pt"][:, :self.input_dim]
            mask = sample["pt"][:, self.input_dim:]
            pkl = sample["pkl"]
            yield sample["__key__"], xml, features, mask, pkl

def get_webdataloader(dataset, batch_size=32):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Create webdataset from directory
# tar -cvf webdataset.tar -C webdataset/ft $(ls -1 webdataset/ft | sort)

if __name__ == "__main__":
    url = sys.argv[1]
    dataset = FilteredTensorWebDataset(url, input_dim=D_TOKEN, filter=lambda sample: int(sample['__key__']) % 10 == 0).decode(data_decoder)
    dataloader = get_webdataloader(dataset, batch_size=64)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    
    for sample in dataloader:
        x, mask = sample
        print(x.shape, mask.shape)
        break
