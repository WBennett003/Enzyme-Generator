import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import confusion_matrix
from rdkit.Chem import rdChemReactions
from rdkit.Chem import Draw


def plot_prediction(pred, true): #takes true and predicted input
    conf_fig = plot_AA_confusion(true.argmax(-1), pred.argmax(-1), get_figure=True)
    seq_fig = plot_Seq_heat(true, pred, get_figure=True)
    return conf_fig, seq_fig

def plot_reaction(rxn):
    reaction = rdChemReactions.ReactionFromSmarts(rxn)
    img = Draw.ReactionToImage(reaction)
    return img

def visualise_denoising(xt, xt_hat):
    pass

def plot_AA_confusion(Y_true, Y_pred, labels='datasets/Amino Acids.json', annot=False, n_AAs=21, get_figure=False):
    with open(labels, 'r') as f:
        labels = json.load(f)
        labels = {i: key for i, key in enumerate(labels.keys())}
        labels = [labels[k] for k in sorted(labels)]

    fig, ax = plt.subplots(3, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(8,16))
    
    uniques = np.arange(n_AAs)
    blank = np.zeros(n_AAs)
    True_uniques, True_count = np.unique(Y_true, return_counts=True)
    blank[np.intc(True_uniques)] = True_count    
    True_count = blank

    ax[1].bar(uniques, True_count)
    ax[1].set_xticks(uniques, labels)
    ax[1].set_title("True AA Distrubution")

    blank = np.zeros(n_AAs)
    Pred_uniques, Pred_count = np.unique(Y_pred, return_counts=True)
    blank[np.intc(Pred_uniques)] = Pred_count    
    Pred_count = blank

    ax[2].bar(uniques, Pred_count, label=labels)
    ax[2].set_title("Pred AA Distrubution")
    ax[2].set_xticks(uniques, labels)


    if len(Y_true.shape) == 1:
        Conf_matrix = confusion_matrix(Y_true, Y_pred, labels=uniques)
    else:
        confs = np.zeros((n_AAs, n_AAs))
        for batch in range(Y_true.shape[0]):
            Conf_matrix = confusion_matrix(Y_true[batch], Y_pred[batch], labels=uniques)
            confs += Conf_matrix
        Conf_matrix = confs

    heatmap = sns.heatmap(Conf_matrix, xticklabels=labels, yticklabels=labels, annot=annot, ax=ax[0])
    ax[0].set(xlabel='Pred', ylabel='True')

    if get_figure:
        plt.close()
        return fig
    else:
        plt.show()

def plot_Seq_heat(TrueY, PredY, n_AAs=21, labels='datasets/Amino Acids.json', get_figure=False):
    with open(labels, 'r') as f:
        labels = json.load(f)
        labels = {i: key for i, key in enumerate(labels.keys())}
        labels = [labels[k] for k in sorted(labels)]

    fig, ax = plt.subplots(2, figsize=(16,8), sharex=True)

    sns.heatmap(TrueY.T, yticklabels=labels, ax=ax[0])
    ax[0].set_title('True')
    sns.heatmap(PredY.T, yticklabels=labels, ax=ax[1])
    ax[1].set_title('Pred')

    if get_figure:
        plt.close()
        return fig
    else:
        plt.show()
    
def plot_Step_heat(TrueY, PredY, noise, n_AAs=21, labels='datasets/Amino Acids.json', get_figure=False):
    with open(labels, 'r') as f:
        labels = json.load(f)
        labels = {i+1: key for i, key in enumerate(labels.keys())}
        labels = [labels[k] for k in sorted(labels)[:n_AAs-1]]

    fig, ax = plt.subplots(3, figsize=(16,8), sharex=True)

    sns.heatmap(TrueY.T, yticklabels=labels, ax=ax[0])
    ax[0].set_title('True')
    sns.heatmap(PredY.T, yticklabels=labels, ax=ax[1])
    ax[1].set_title('X(t-1)')
    sns.heatmap(noise.T, yticklabels=labels, ax=ax[2])
    ax[1].set_title('noise')
    if get_figure:
        plt.close()
        return fig
    else:
        plt.show()

def plot_reactions_input(rxn): #plots the conditioning reaction
    pass