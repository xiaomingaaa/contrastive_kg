'''
Author: your name
Date: 2021-08-08 14:43:11
LastEditTime: 2021-10-13 15:05:57
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /molecular_representation/preprocess.py
'''
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from dgllife.data import Tox21
from torch.utils import data
from functools import partial
from utils import split_dataset, collate_molgraphs, collate_pretext

from sklearn.preprocessing import OneHotEncoder

from subword_nmt.apply_bpe import BPE
import codecs
import pandas as pd
import numpy as np
vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

max_d = 205


def drug2emb_encoder(x):
    max_d = 50
    #max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class Pretext_DataLoader():
    def __init__(self, args):
        self.args = args
        self._data_process()

    def _data_process(self):
        if self.args['dataset'] == 'Tox21':
            datas = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                          node_featurizer=self.args['node_featurizer'], edge_featurizer=self.args['edge_featurizer'], n_jobs=1 if self.args['num_workers'] == 0 else self.args['num_workers'], cache_file_path='./dataset/tox21_dglgraph.bin')
            self.args['n_tasks'] = datas.n_tasks
            train_data, val_data, test_data = split_dataset(self.args, datas)
            all_datapoints = list()
            for d in datas:
                smiles = d[0]
                d_encoder, d_input_mask = drug2emb_encoder(smiles)
                all_datapoints.append(
                    (torch.from_numpy(d_encoder), torch.from_numpy(d_input_mask), d[1], d[2], d[3]))
            self.pretrain_datas = data.DataLoader(
                dataset=all_datapoints, batch_size=self.args['batch_size'], collate_fn=collate_pretext)
            
            ''' classification to use '''
            self.train_loader = data.DataLoader(
                dataset=train_data, batch_size=self.args['batch_size'], collate_fn=collate_molgraphs, num_workers=self.args['num_workers'])
            self.val_loader = data.DataLoader(
                dataset=val_data, batch_size=self.args['batch_size'], collate_fn=collate_molgraphs)
            self.test_loader = data.DataLoader(
                dataset=test_data, batch_size=self.args['batch_size'])

def sample_with_sim():