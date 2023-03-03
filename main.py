import torch
import os 
import collections

from Networks.UNet import UNetModel
from Networks.Transformer import Transformer
from datahandler import dataset_h5
from diffusion import DenoiseDiffusion
from enzyme import EnzymeGenerator
from tokeniser import Amino_Acid_Tokeniser

from omegafold import omegaplm
from omegafold.model import OmegaFold
import argparse


import sys

def _make_config(input_dict: dict) -> argparse.Namespace:
    """Recursively go through dictionary"""
    new_dict = {}
    for k, v in input_dict.items():
        if type(v) == dict:
            new_dict[k] = _make_config(v)
        else:
            new_dict[k] = v
    return argparse.Namespace(**new_dict)

device = torch.device("cuda:0" if not torch.cuda.is_available() else "cpu")
print(device)

def train():
    aa_tokeniser = Amino_Acid_Tokeniser()
    ds = dataset_h5(file_path='datasets/1M_sample_ds.h5', device=device)


    params = {
        "EPOCHS" : 10,
        "BATCH_SIZE" : 5,
        "LEARN_RATE" : 1e-3,
        "AA_SIZE" : len(aa_tokeniser),
        "MAX_SEQUENCE_LENGTH" : 1000, #4th value in ds is AA_seq
        "DMODEL" : 24,
        "dff" : 256,
        "N_STEPS" : 200,
        "DROPOUT" : 0.01,
        "N_HEADS" : 6,
        "N_BLOCKS" : 6,
        "CHEM_HASHSIZE" : 256
    }
    model = Transformer(params['MAX_SEQUENCE_LENGTH'], params['CHEM_HASHSIZE'], params['DROPOUT'], params['DMODEL'], params['dff'], params['N_HEADS'], params['N_BLOCKS'], params['AA_SIZE'])
    # model = UNetModel(in_channels=params['AA_SIZE'], out_channels=params['AA_SIZE'], channels=params['CHANNELS'], n_res_blocks=params['N_RES_BLOCKS'], attention_levels=params['ATTN_LVLS'], channel_multipliers=params['CHANNEL_MULTIPLIERS'], n_heads=params['N_HEADS'], d_cond=params['CHEM_HASHSIZE'], device=device)
    pytorch_total_params = sum(p.numel() for p in model.parameters()) 
    print(pytorch_total_params)
    model.to(device)

    denoise = DenoiseDiffusion(model, params['N_STEPS'], device)
    enzyme_generator = EnzymeGenerator(denoise, params['MAX_SEQUENCE_LENGTH'], aa_tokeniser, params['N_STEPS'], MODEL_PARAMS=params, WANDB=True, device=device)

    enzyme_generator.train(ds, params['EPOCHS'], params['BATCH_SIZE'], params['LEARN_RATE'])

def test():
    aa_tokeniser = Amino_Acid_Tokeniser()
    ds = dataset_h5(file_path='datasets/mini_ds.h5', device=device)

    params = {
        "EPOCHS" : 2,
        "BATCH_SIZE" : 10,
        "LEARN_RATE" : 1e-3,
        "AA_SIZE" : len(aa_tokeniser),
        "MAX_SEQUENCE_LENGTH" : 1000, #4th value in ds is AA_seq
        "DMODEL" : 24,
        "dff" : 128,
        "N_STEPS" : 200,
        "DROPOUT" : 0.01,
        "N_HEADS" : 2,
        "N_BLOCKS" : 2,
        "CHEM_HASHSIZE" : 256
    }
    model = Transformer(params['MAX_SEQUENCE_LENGTH'], params['CHEM_HASHSIZE'], params['DROPOUT'], params['DMODEL'], params['dff'], params['N_HEADS'], params['N_BLOCKS'], params['AA_SIZE'])
    pytorch_total_params = sum(p.numel() for p in model.parameters()) 
    print(pytorch_total_params)
    model.to(device)
    denoise = DenoiseDiffusion(model, params['N_STEPS'], device)
    enzyme_generator = EnzymeGenerator(denoise, params['MAX_SEQUENCE_LENGTH'], aa_tokeniser, params['N_STEPS'], MODEL_PARAMS=params, WANDB=False, device=device)

    enzyme_generator.train(ds, params['EPOCHS'], params['BATCH_SIZE'], params['LEARN_RATE'])

def get_cfg():
    cfg = dict(
        alphabet_size=21,
        plm=dict(
            chem_size=256,
            dff=2056,
            nlayers=4,
            alphabet_size=23,
            node=1280,
            padding_idx=21,
            edge=66,
            proj_dim=1280 * 2,
            attn_dim=256,
            num_head=1,
            num_relpos=129,
            masked_ratio=0.12,
        ),
        node_dim=256,
        edge_dim=128,
        relpos_len=32,
        prev_pos=dict(
            first_break=3.25,
            last_break=20.75,
            num_bins=16,
            ignore_index=0,
        ),
        rough_dist_bin=dict(
            x_min=3.25,
            x_max=20.75,
            x_bins=16,
        ),
        dist_bin=dict(
            x_bins=64,
            x_min=2,
            x_max=65,
        ),
        pos_bin=dict(
            x_bins=64,
            x_min=-32,
            x_max=32,
        ),
        c=16,
        geo_num_blocks=50,
        gating=True,
        attn_c=32,
        attn_n_head=8,
        transition_multiplier=4,
        activation="ReLU",
        opm_dim=32,
        geom_count=2,
        geom_c=32,
        geom_head=4,
        struct=dict(
            node_dim=384,
            edge_dim=128,
            num_cycle=8,
            num_transition=3,
            num_head=12,
            num_point_qk=4,
            num_point_v=8,
            num_scalar_qk=16,
            num_scalar_v=16,
            num_channel=128,
            num_residual_block=2,
            hidden_dim=128,
            num_bins=50,
        ),
        struct_embedder = 2
    )
    return _make_config(cfg)

def _load_weights(
        weights_url: str, weights_file: str,
) -> collections.OrderedDict:
    """
    Loads the weights from either a url or a local file. If from url,

    Args:
        weights_url: a url for the weights
        weights_file: a local file

    Returns:
        state_dict: the state dict for the model

    """

    weights_file = os.path.expanduser(weights_file)
    use_cache = os.path.exists(weights_file)

    return torch.load(weights_file, map_location='cpu')

def omegafold():
    cfg = get_cfg()
    # model = omegaplm.OmegaPLM(cfg.plm)
    model = OmegaFold(cfg)
    
    state_dict = 'omegafold_ckpt\model2.pt'
    weights = _load_weights('', state_dict)
    weights['omega_plm.input_embedding.bias'] = torch.zeros(cfg.plm.alphabet_size) 
    weights['input_embedder.proj_i.bias'] = torch.zeros(cfg.alphabet_size) 
    weights['input_embedder.proj_j.bias'] = torch.zeros(cfg.alphabet_size) 
    
    model.load_state_dict(weights)
    model.to(device)

    aa_tokeniser = Amino_Acid_Tokeniser()
    ds = dataset_h5(file_path='datasets/mini_ds.h5', device=device)
    X_test = ds[:1][0]
    conds = ds[:1][1]
    seq = torch.where(X_test > 0, X_test, 0)
    seq = torch.nn.functional.one_hot(torch.round(seq).long(), cfg.alphabet_size).float()
    y = model(seq, conds)
    pass

def omegaPLM():
    cfg = get_cfg()

    fwd_cfg = argparse.Namespace(
        subbatch_size=1000,
        num_recycle=1,
    )

    model = omegaplm.OmegaPLM(cfg.plm)
    model.to(device)
    state_dict = 'omegafold_ckpt\model2.pt'
    weights = _load_weights('', state_dict)
    weights['omega_plm.input_embedding.bias'] = torch.zeros(cfg.plm.node)
    weights['omega_plm.input_embedding.weight'] = weights['omega_plm.input_embedding.weight'].T #transpose coz the embbeding is MxN instead of NXM?

    w = {}

    state_dict = model.state_dict()
    for k in state_dict:
        w[k] = state_dict[k]

    for k in weights.keys():
        if k[:10] == 'omega_plm.':
            w[k[10:]] = weights[k]
    





    model.load_state_dict(w)
    torch.save(model.state_dict, 'weights/omegaPLM.pt')
    aa_tokeniser = Amino_Acid_Tokeniser()
    ds = dataset_h5(file_path='datasets/mini_ds.h5', device=device)
    X_test = ds[:1][0]
    conds = ds[:1][1]
    seq = torch.zeros((1, 1280))
    seq[:, :1000] = X_test
    seq = torch.where(seq > 0, seq, 0)
    seq = torch.nn.functional.one_hot(torch.round(seq).long(), cfg.plm.alphabet_size).float().to(device)
    mask = torch.ones((1, 1280)).to(device)
    y = model(seq, mask, conds, torch.tensor([10]), fwd_cfg)

    pass

def train_omegaplm():
    aa_tokeniser = Amino_Acid_Tokeniser()
    ds = dataset_h5(file_path='datasets/mini_ds.h5', device=device)


    params = {
        "EPOCHS" : 10,
        "BATCH_SIZE" : 1,
        "LEARN_RATE" : 1e-3,
        "AA_SIZE" : len(aa_tokeniser),
        "MAX_SEQUENCE_LENGTH" : 1000, #4th value in ds is AA_seq
        "DMODEL" : 24,
        "dff" : 256,
        "N_STEPS" : 200,
        "DROPOUT" : 0.01,
        "N_HEADS" : 6,
        "N_BLOCKS" : 6,
        "CHEM_HASHSIZE" : 256
    }
    
    cfg = get_cfg()

    fwd_cfg = argparse.Namespace(
        subbatch_size=1000,
        num_recycle=1,
    )

    model = omegaplm.OmegaPLM(cfg.plm)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters()) 
    print(pytorch_total_params)
    model.to(device)
    model.load_state_dict(torch.load('weights/openPLM.pt'))

    denoise = DenoiseDiffusion(model, params['N_STEPS'], device)

    enzyme_generator = EnzymeGenerator(denoise, params['MAX_SEQUENCE_LENGTH'], aa_tokeniser, params['N_STEPS'], MODEL_PARAMS=params, WANDB=True, device=device)

    enzyme_generator.train(ds, params['EPOCHS'], params['BATCH_SIZE'], params['LEARN_RATE'])

if __name__ == '__main__':
    # train()
    # test()
    omegaPLM()
    