import h5py
import numpy as np
import torch

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, file_path='datasets/t.h5py', device='cpu') -> None:
        super().__init__()
        self.file = h5py.File(file_path, 'r+')
        self.device = device
    
    def __len__(self):
        return len(self.file['accension'])

    def __getitem__(self, idx):
        return (
            torch.Tensor(self.file['AA_seq'][idx]).to(self.device), 
            torch.Tensor(self.file['reaction'][idx]).to(self.device),
            # np.array([self.file['accension'][idx], self.file['smile_equation'][idx], self.file['ecId'][idx], self.file['Rhea_id']]) # text info
            )

