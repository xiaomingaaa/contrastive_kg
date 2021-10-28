'''
Author: your name
Date: 2021-08-06 16:41:59
LastEditTime: 2021-08-09 20:16:52
LastEditors: Please set LastEditors
Description: training of pretrained multi modal
FilePath: /molecular_representation/main.py
'''

from argparse import ArgumentParser
import torch
from model import MT
from utils import init_featurizer, mkdir_path, model_config, get_configure
from preprocess import Pretext_DataLoader


def main(args, data):
    model = MT(args)
    model.to(args['device'])
    for b_id, b_data in enumerate(data.pretrain_datas):
        d_encoder, d_input_masks, bg, labels, masks = b_data
        d_encoder, d_input_masks, labels, masks = d_encoder.to(args['device']), d_input_masks.to(args['device']), labels.to(args['device']), masks.to(args['device'])
        model(d_encoder, d_input_masks, labels, masks, bg)

        

if __name__ == '__main__':
    parser = ArgumentParser('Multi-modal pretrain')
    parser.add_argument('-d', '--dataset', default='Tox21', choices=['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                                     'ToxCast', 'HIV', 'PCBA', 'Tox21'], help='Dataset to use')
    parser.add_argument('-mo', '--model', default='MPNN', choices=['GCN', 'GAT', 'Weave', 'MPNN', 'AttentiveFP',
                                                                  'gin_supervised_contextpred',
                                                                  'gin_supervised_infomax',
                                                                  'gin_supervised_edgepred',
                                                                  'gin_supervised_masking',
                                                                  'NF'], help='model to use')
    parser.add_argument('-f', '--featurizer-type', default='canonical', choices=['canonical', 'attentivefp'],
                        help='Featurization for atoms (and bonds). This is required for models '
                             'other than gin_supervised_**.')
    parser.add_argument('-s', '--split', choices=['scaffold', 'random'], default='scaffold',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-sr', '--split-ratio', default='0.8,0.1,0.1', type=str,
                        help='Proportion of the dataset to use for training, validation and test, '
                             '(default: 1.0,0.0,0.0)')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-nw', '--num-workers', type=int, default=2,
                        help='Number of processes for data loading (default: 0)')
    parser.add_argument('-rp', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    parser.add_argument('-bs', '--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('-eds', '--embed_size', type=int,
                        default=128, help='embedding dim of word and drug graph')
    parser.add_argument('-dr', '--dropout_rate', type=float,
                        default=0.1, help='dropout rate of embedding layers.')
    parser.add_argument('-ml', '--n-mlp-layers', type=int,
                        default=3, help='layers of mlp for classification')
    args = parser.parse_args().__dict__
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')
    args = init_featurizer(args)
    mkdir_path(args['result_path'])
    config = model_config()
    config['in_node_feats'] = args['node_featurizer'].feat_size()
    config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    data = Pretext_DataLoader(args)
    args = data.args
    args.update(config)
    config = get_configure(args['model'], args['featurizer_type'], args['dataset'])
    args.update(config)
    main(args, data)
