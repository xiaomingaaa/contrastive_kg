'''
Author: your name
Date: 2021-08-09 10:34:15
LastEditTime: 2021-10-20 14:16:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /molecular_representation/utils.py
'''
import os
from dgllife.utils import ScaffoldSplitter, RandomSplitter
import dgl
import torch.nn as nn
import numpy as np
import torch
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, precision_recall_curve, cohen_kappa_score

def mkdir_path(pathname):
    path = os.path.join(os.getcwd(), pathname)
    if not os.path.exists(path):
        os.makedirs()


def init_featurizer(args):
    """Initialize node/edge featurizer
    Parameters
    ----------
    args : dict
        Settings
    Returns
    -------
    args : dict
        Settings with featurizers updated
    """
    if args['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                         'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
        args['featurizer_type'] = 'pre_train'
        args['node_featurizer'] = PretrainAtomFeaturizer()
        args['edge_featurizer'] = PretrainBondFeaturizer()
        return args

    if args['featurizer_type'] == 'canonical':
        from dgllife.utils import CanonicalAtomFeaturizer
        args['node_featurizer'] = CanonicalAtomFeaturizer()
    elif args['featurizer_type'] == 'attentivefp':
        from dgllife.utils import AttentiveFPAtomFeaturizer
        args['node_featurizer'] = AttentiveFPAtomFeaturizer()
    else:
        return ValueError(
            "Expect featurizer_type to be in ['canonical', 'attentivefp'], "
            "got {}".format(args['featurizer_type']))

    if args['model'] in ['Weave', 'MPNN', 'AttentiveFP']:
        if args['featurizer_type'] == 'canonical':
            from dgllife.utils import CanonicalBondFeaturizer
            args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
        elif args['featurizer_type'] == 'attentivefp':
            from dgllife.utils import AttentiveFPBondFeaturizer
            args['edge_featurizer'] = AttentiveFPBondFeaturizer(self_loop=True)
    else:
        args['edge_featurizer'] = None

    return args



def split_dataset(args, dataset):
    """Split the dataset
    Parameters
    ----------
    args : dict
        Settings
    dataset
        Dataset instance
    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    train_ratio, val_ratio, test_ratio = map(float, args['split_ratio'].split(','))
    if args['split'] == 'scaffold':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args['split'] == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args['split']))
    
    return train_set, val_set, test_set

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def collate_pretext(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    if len(data[0]) == 4:
        d_encoder, d_input_masks, graphs, labels = map(list, zip(*data))
    else:
        d_encoder, d_input_masks, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    d_encoders = torch.stack(d_encoder, dim=0)
    d_input_masks = torch.stack(d_input_masks, dim=0)
    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return d_encoders, d_input_masks, bg, labels, masks

def model_config():
    config = {}
    config['input_dim_drug'] = 23532
    config['max_drug_seq'] = 50
    config['dropout_rate'] = 0.1
    
    #DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3
    
    # Encoder
    config['intermediate_size'] = 256
    config['num_attention_heads'] = 8
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['flat_dim'] = 256
    return config

def get_configure(model, featurizer_type, dataset):
    """Query for configuration

    Parameters
    ----------
    model : str
        Model type
    featurizer_type : str
        The featurization performed
    dataset : str
        Dataset for modeling

    Returns
    -------
    dict
        Returns the manually specified configuration
    """
    if featurizer_type == 'pre_train':
        with open('configures/{}/{}.json'.format(dataset, model), 'r') as f:
            config = json.load(f)
    else:
        file_path = 'configures/{}/{}_{}.json'.format(dataset, model, featurizer_type)
        if not os.path.isfile(file_path):
            return NotImplementedError('Model {} on dataset {} with featurization {} has not been '
                                       'supported'.format(model, dataset, featurizer_type))
        with open(file_path, 'r') as f:
            config = json.load(f)
    return config

def graph_training(args, model, bg, device):
    bg = bg.to(device)
    if args['edge_featurizer'] is None:
        node_feats = bg.ndata.pop('h').to(device)
        return model(bg, node_feats)
    elif args['featurizer_type'] == 'pre_train':
        node_feats = [
            bg.ndata.pop('atomic_number').to(device),
            bg.ndata.pop('chirality_type').to(device)
        ]
        edge_feats = [
            bg.edata.pop('bond_type').to(device),
            bg.edata.pop('bond_direction_type').to(device)
        ]
        return model(bg, node_feats, edge_feats)
    else:
        node_feats = bg.ndata.pop('h').to(device)
        edge_feats = bg.edata.pop('e').to(device)
        return model(bg, node_feats, edge_feats)

def concordance_index(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ind = np.argsort(y_true)
    y_true = y_true[ind]
    y_pred = y_pred[ind]
    i = len(y_true)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y_true[i] > y_true[j]:
                z = z+1
                u = y_pred[i] - y_pred[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

'''
description: contrastive loss
param {*}
return {*}
'''
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0, distance_type: str = 'cosine', reduction: str = 'mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.dis_type = distance_type
        self.reduction = reduction

    def forward(self, output1, output2, target):
        if self.dis_type == 'Euclidean':
            dist2 = (output2 - output1).pow(2).sum(1).reshape(-1, 1)
            dist = dist2.sqrt()
        elif self.dis_type == 'cosine':
            dist = 1-F.cosine_similarity(output1, output2)
            dist2 = dist.pow(2).reshape(-1, 1)
        else:
            print("Distance Type Error!")

        losses = 0.5 * (target.float() * dist2 +
                        (1 + -1 * target).float() * F.relu(self.margin - (dist2 + self.eps).sqrt()).pow(2))
        if self.reduction == 'mean':
            return dist, losses.mean()
        if self.reduction == 'sum':
            return dist, losses.sum()
        if self.reduction == 'none':
            return dist, losses

def eval_multi_class(pred, labels):
    pred_labels = np.argmax(pred.detach().numpy(), axis=1)
    pre_score = [pred.detach().numpy()[i][s] for i, s in enumerate(pred_labels)]
    #labels = [x - 1 for x in labels]
    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels, average='macro')
    kappa = cohen_kappa_score(labels, pred_labels)
    return acc, f1, kappa

def get_device(gpu):
    if gpu==-1:
        return torch.device('cpu')
    else:
        return torch.device('cuda:{}'.format(gpu))