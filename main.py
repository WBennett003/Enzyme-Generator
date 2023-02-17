import torch

from Networks.UNet import UNetModel
from Networks.Transformer import Transformer
from datahandler import dataset_h5
from diffusion import DenoiseDiffusion
from enzyme import EnzymeGenerator
from tokeniser import Amino_Acid_Tokeniser

import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train():
    aa_tokeniser = Amino_Acid_Tokeniser()
    ds = dataset_h5(file_path='datasets/1M_sample_ds.h5', device=device)


    params = {
        "EPOCHS" : 10,
        "BATCH_SIZE" : 15,
        "LEARN_RATE" : 1e-3,
        "AA_SIZE" : len(aa_tokeniser),
        "MAX_SEQUENCE_LENGTH" : 1000, #4th value in ds is AA_seq
        "DMODEL" : 24,
        "dff" : 128,
        "N_STEPS" : 200,
        "DROPOUT" : 0.01,
        "N_HEADS" : 4,
        "N_BLOCKS" : 2,
        "CHEM_HASHSIZE" : 256
    }
    model = Transformer(params['MAX_SEQUENCE_LENGTH'], params['CHEM_HASHSIZE'], params['DROPOUT'], params['DMODEL'], params['dff'], params['N_HEADS'], params['N_BLOCKS'], params['AA_SIZE'])
    # model = UNetModel(in_channels=params['AA_SIZE'], out_channels=params['AA_SIZE'], channels=params['CHANNELS'], n_res_blocks=params['N_RES_BLOCKS'], attention_levels=params['ATTN_LVLS'], channel_multipliers=params['CHANNEL_MULTIPLIERS'], n_heads=params['N_HEADS'], d_cond=params['CHEM_HASHSIZE'], device=device)
    model.to(device)
    denoise = DenoiseDiffusion(model, params['N_STEPS'], device)
    enzyme_generator = EnzymeGenerator(denoise, params['MAX_SEQUENCE_LENGTH'], aa_tokeniser, params['N_STEPS'], MODEL_PARAMS=params, WANDB=True, device=device)

    enzyme_generator.train(ds, params['EPOCHS'], params['BATCH_SIZE'], params['LEARN_RATE'])


if __name__ == '__main__':
    train()