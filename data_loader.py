'''
Author: your name
Date: 2021-10-12 14:53:36
LastEditTime: 2021-10-26 16:59:37
LastEditors: Please set LastEditors
Description: load datasets with a high-level api
FilePath: /molecular_representation/data_loader.py
'''

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from dgllife.utils import smiles_to_bigraph
from dgl.dataloading.pytorch import GraphDataLoader
import os

#### load smiles of drug pair and subgraph of kg
class CrossViewData(Dataset):
    def __init__(self, params, dataname='DrugBank', hop=2, mode='train'):
        print('Loading {} dataset'.format(mode))
        self.dataname = dataname
        self.hop = hop
        self.mode = mode
        self.params = params
        ### initialize datas
        self._init()

        
    def _init(self):
        if self.dataname=='DrugBank':
            if os.path.isfile('dataset/drugbank/drugbank_{}hop_{}_format.pkl'.format(self.hop,self.mode)):
                self.data = pickle.load(open('dataset/drugbank/drugbank_{}hop_{}_format.pkl'.format(self.hop, self.mode),'rb'))
            
            else:
                data = pickle.load(open('dataset/drugbank/drugbank_{}hop_{}.pkl'.format(self.hop, self.mode),'rb'))
                temp = []
                for d in data:
                    d1,d2 = d[1]
                    d1_graph = smiles_to_bigraph(d1, add_self_loop=True, node_featurizer=self.params['node_featurizer'], edge_featurizer=self.params['edge_featurizer'])
                    d2_graph = smiles_to_bigraph(d2, add_self_loop=True, node_featurizer=self.params['node_featurizer'], edge_featurizer=self.params['edge_featurizer'])
                    if d1_graph and d2_graph:
                        temp.append([[d1_graph, d2_graph, d[2]], int(d[3])-1])
                self.data = temp
                ## save data
                pickle.dump(temp, open('dataset/drugbank/drugbank_{}hop_{}_format.pkl'.format(self.hop, self.mode), 'wb'))
        else:
            pass

    def sample_neg(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

if __name__=='__main__':
    a = CrossViewData()
    d = GraphDataLoader(a, batch_size=12)
    for i_batch, dd in d:
        print(i_batch)
        print(dd)