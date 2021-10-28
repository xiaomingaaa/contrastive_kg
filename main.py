'''
Author: your name
Date: 2021-10-16 14:23:54
LastEditTime: 2021-10-26 18:00:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /molecular_representation/main.py
'''

from models.model import CrossView, Trainer
from data_loader import CrossViewData
from dgl.dataloading.pytorch import GraphDataLoader
from argparse import ArgumentParser
from utils import init_featurizer, mkdir_path, model_config, get_configure
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = ArgumentParser('Multi-modal Cross View')
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
    parser.add_argument('-ef', '--edge_featurizer', default='canonical', choices=['canonical', 'attentivefp'],
                        help='Featurization for atoms (and bonds). This is required for models '
                             'other than gin_supervised_**.')
    parser.add_argument('-s', '--split', choices=['scaffold', 'random'], default='scaffold',
                        help='Dataset splitting method (default: scaffold)')
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score'],
                        default='roc_auc_score',
                        help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-n', '--num-epochs', type=int, default=1000,
                        help='Maximum number of epochs for training. '
                             'We set a large number by default as early stopping '
                             'will be performed. (default: 1000)')
    parser.add_argument('-rp', '--result-path', type=str, default='classification_results',
                        help='Path to save training results (default: classification_results)')
    parser.add_argument('-bs', '--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('-eds', '--embed_size', type=int,
                        default=128, help='embedding dim of word and drug graph')
    parser.add_argument('-dr', '--dropout_rate', type=float,
                        default=0.1, help='dropout rate of embedding layers.')
    parser.add_argument('-ml', '--n_mlp_layers', type=int,
                        default=3, help='layers of mlp for classification')
    parser.add_argument('-if', '--in_feats', type=int, default=200, help='the size of node features for GNN')
    parser.add_argument('-hf', '--hidden_feats', type=int, default=128, help='hidden features for GNN or MLP')
    parser.add_argument('-of', '--out_feats', type=int, default=128, help='size of output features for GNN')
    parser.add_argument('-ehf', '--edge_hidden_feats', type=int, default=128, help='size of edge hidden features for MPNN')
    parser.add_argument('-lr', '--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-gpu', '--gpu', type=int, default=1, help='gpu id')
    parser.add_argument('-ep', '--epoch', type=int, default=1000, help='epoch')
    parser.add_argument('-nc', '--num_classes', type=int, default=86, help='epoch')
    parser.add_argument('-l1', '--lambda1', type=float, default=0.1, help='lamdb')
    parser.add_argument('-l2', '--lambda2', type=float, default=0.1, help='lambda2 param')
    parser.add_argument('-ei', '--eval_interval', type=int, default=2, help='the interval of evaluation')
    parser.add_argument('--hop', type=int, default=1, help='the hop of neighboors')

    params = parser.parse_args().__dict__
    params = init_featurizer(params)                              
    config = model_config()
    config['in_node_feats'] = params['node_featurizer'].feat_size()
    config['in_edge_feats'] = params['edge_featurizer'].feat_size()
    params.update(config)
    dataset = CrossViewData(params, hop=params['hop'])
    val_dataset = CrossViewData(params, mode='val', hop=params['hop'])
    test_dataset = CrossViewData(params, mode='test', hop=params['hop'])
    dataloader = GraphDataLoader(dataset, batch_size=params['batch_size'])
    val_dataloader = GraphDataLoader(val_dataset, batch_size=params['batch_size'])
    test_dataloader = GraphDataLoader(test_dataset, batch_size=params['batch_size'])
    Trainer(params, dataloader, test_dataloader, val_dataloader)

if __name__=='__main__':
     main()