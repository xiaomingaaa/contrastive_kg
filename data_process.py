'''
Author: your name
Date: 2021-10-14 12:42:54
LastEditTime: 2021-10-26 16:50:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /molecular_representation/data_process.py
'''
import dgl
import gzip
import torch as th
import numpy as np
import scipy.sparse as ssp
import pickle
import random
from sklearn.model_selection import train_test_split
# 传入了train中的所有三元组

def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    ### 稀疏矩阵乘法
    sp_neighbors = sp_nodes.dot(adj) # current_node)id列值为1[0,0,0,1,0,0,0...,1,...].dot(adj)，结果会得到1*adj.shape[1]向量
    # 如果某个node是current_nodes中的某个或某几个的邻接节点，那对应的列就是>0的数
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    # 返回稀疏矩阵A中的非零元的位置以及数值
    return neighbors

# 创建一个1*adj.dim矩阵，0行node_id列为1
def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs. #广度优限搜索
    Modified from dgl. contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        # 返回current_nodes的邻居节点集合，set类型
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            # 随机采样节点邻居的固定大小
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)
        # 返回两个集合的并集，即包含了所有集合的元素，重复的元素只会出现一次


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        # 对于普通的生成器，第一个next调用，相当于启动生成器，会从生成器函数的第一行代码开始执行，直到第一次执行完yield语句（第4行）后，跳出生成器函数。
        # 然后第二个next调用，进入生成器函数后，从yield语句的下一句语句（第5行）开始执行，然后重新运行到yield语句，执行后，跳出生成器函数
        except StopIteration:
            pass
    return set().union(*lvls)  # 返回并集 *lvls 表明传入的不止一个元祖,根据h返回h次list，都放在lvls中，*lvls取出这些子列表进行合并



def extract_subgraph(g, drugA, drugB, adj, embedding, hop=2):
    drugA_n = get_neighbor_nodes(set([drugA]), adj, h=hop, max_nodes_per_hop=500)
    drugB_n = get_neighbor_nodes(set([drugB]), adj, h=hop, max_nodes_per_hop=500)
    subgraph_nodes_int = drugA_n.intersection(drugB_n)
    subgraph_nodes_un = drugA_n.union(drugB_n)

    if enclosing_sub_graph:
        if drugA in subgraph_nodes_int:
            subgraph_nodes_int.remove(drugA)
        if drugB in subgraph_nodes_int:
            subgraph_nodes_int.remove(drugB)
        subgraph_nodes = list(subgraph_nodes_int) + [drugA, drugB]
    else:
        if drugA in subgraph_nodes_un:
            subgraph_nodes_un.remove(drugA)
        if drugB in subgraph_nodes_un:
            subgraph_nodes_un.remove(drugB)
        subgraph_nodes = list(subgraph_nodes_un) + [drugA, drugB]

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes]]


    if len(subgraph_nodes)==0:
        global missing
        missing += 1
        return None
    subgraph_nodes = list(subgraph_nodes)+[drugA, drugB]
    subgraph = dgl.node_subgraph(g, subgraph_nodes, store_ids=True)
    nids = subgraph.ndata[dgl.NID]
    subgraph.ndata['feat'] = embedding[nids,:]
    # eids = subgraph.edata[dgl.EID]
    # original_edge = g.edges()
    return subgraph

def process_drugbank():
    edges_u, edges_v = [], []
    rows, cols, dats = [], [], []
    edges_dict = dict()
    node_triples = dict()
    with open('dataset/hetionet/triples.tsv', 'r') as f:
        for line in f:
            e1, r, e2 = line.strip().split('\t')
            rows.append(int(e1))
            cols.append(int(e2))
            dats.append(1)
            edges_u.append(int(e1))
            edges_v.append(int(e2))
            e1_id = int(e1)
            e2_id = int(e2)
            if e1_id not in node_triples:
                # 使用集合避免重复
                node_triples[e1_id] = set()
                node_triples[e1_id].add((e1_id, int(r), e2_id))
            else:
                node_triples[e1_id].add((e1_id, int(r), e2_id))
            if e2_id not in node_triples:
                # 使用集合避免重复
                node_triples[e2_id] = set()
                node_triples[e2_id].add((e1_id, int(r), e2_id))
            else:
                node_triples[e2_id].add((e1_id, int(r), e2_id))

            if (int(e1), int(e2)) not in edges_dict:
                edges_dict[(int(e1), int(e2))] = list()
                # !!!
                edges_dict[(int(e1), int(e2))].append(int(r))
            else:
                edges_dict[(int(e1), int(e2))].append(int(r))  # 构建全图
    # 预处理全图，这一块可以放到这个方法外面提前处理好，就不用每次提取一次子图就处理一下
    
    # DGLgraph
    g = dgl.DGLGraph((th.tensor(edges_u), th.tensor(edges_v)))

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(dats)
    # 邻接矩阵稀疏存储
    adj = ssp.csr_matrix((data, (rows, cols)), shape=(len(rows), len(cols)))
    adj += adj.T
    # 转置是因为这里考虑边是无向的
    entity2id = dict()
    smiles_dict = dict()
    with open('dataset/drugbank/drug_smiles.tsv','r') as f:
        for line in f:
            did, smile = line.strip().split('\t')
            smiles_dict[did] = smile
    with open('dataset/hetionet/entity2id.tsv','r') as f:
        for line in f:
            e1, e_id = line.strip().split('\t')
            entity2id[e1] = int(e_id)

    data = []
    with open('dataset/drugbank/Drugbank.txt','r') as f:
        f.readline() #去掉表头
        missing_set = []
        embedding = np.load('ckpts/TransE_Hetionet_9/Hetionet_TransE_entity.npy', allow_pickle=True)
        embedding = th.from_numpy(embedding)
        for line in f:
            d1, d2, label = line.strip().split('\t')
            if d1 not in smiles_dict or d2 not in smiles_dict:
                continue
            if 'Compound::{}'.format(d1) not in entity2id or 'Compound::{}'.format(d2) not in entity2id:
                missing_set.append([(d1, d2), (smiles_dict[d1], smiles_dict[d2]), None, label]) ### missing data
                #

                global missing
                missing += 1
                continue
            d1_id = entity2id['Compound::{}'.format(d1)]
            d2_id = entity2id['Compound::{}'.format(d2)]
            subgraph = extract_subgraph(g, int(d1_id), int(d2_id), adj, embedding=embedding, hop=hop)
            
            print(subgraph)
            if subgraph:
                subgraph = dgl.add_self_loop(subgraph)
                data.append([(d1, d2), (smiles_dict[d1], smiles_dict[d2]), subgraph, label])
            else:
                missing_set.append([(d1, d2), (smiles_dict[d1], smiles_dict[d2]), subgraph, label])
    train_file = open('dataset/drugbank/drugbank_{}hop_train.pkl'.format(hop), 'wb')
    test_file = open('dataset/drugbank/drugbank_{}hop_test.pkl'.format(hop), 'wb')
    val_file = open('dataset/drugbank/drugbank_{}hop_val.pkl'.format(hop), 'wb')
    missing_file = open('dataset/drugbank/drugbank_{}hop_missing.pkl'.format(hop), 'wb')
    whole_file = open('dataset/drugbank/drugbank_{}hop_whole.pkl'.format(hop), 'wb')
    print('total data: ', len(data))
    train_set, val_set = train_test_split(data, test_size=0.2)
    test_set, val_set = train_test_split(val_set, test_size=0.5)
    pickle.dump(test_set, test_file)
    pickle.dump(val_set, val_file)
    pickle.dump(train_set, train_file)
    pickle.dump(missing_set, missing_file)
    pickle.dump(data, whole_file)
    
def process_hetionet():
    entity2id = dict()
    # with open('dataset/hetionet/hetionet-v1.0-nodes.tsv', 'r') as f:
    #     for line in f:
    #         entity, info,_ = line.strip().split('\t')
    #         if not entity in entity2id:
    #             entity2id[entity] = len(entity2id)
    
    
    relation2id = dict()
    triples_writer = open('dataset/hetionet/triples.tsv','w')
    triples = set()
    with gzip.open('dataset/hetionet/hetionet-v1.0-edges.sif.gz', 'rb') as f:
        f.readline()
        for line in f:
            line = line.decode('utf-8').strip()
            e1, r, e2 = line.strip().split('\t')
            if e1 not in entity2id:
                entity2id[e1] = len(entity2id)
            if e2 not in entity2id:
                entity2id[e2] = len(entity2id)
            triples.add((e1,r,e2))
            if r not in relation2id:
                relation2id[r] = len(relation2id)
            triples_writer.write('{}\t{}\t{}\n'.format(entity2id[e1],relation2id[r],entity2id[e2]))
    print('number of triples: ', len(triples))
    triples_writer.close()
    with open('dataset/hetionet/entity2id.tsv', 'w') as f:
        for entity in entity2id:
            f.write('{}\t{}\n'.format(entity, entity2id[entity]))
    print('number of entities: ', len(entity2id))
    relation_writer = open('dataset/hetionet/relation2id.tsv','w')
    for r in relation2id:
        relation_writer.write('{}\t{}\n'.format(r, relation2id[r]))
    
    relation_writer.close()

def process_drug_smiles():
    import pandas as pd
    data = pd.read_csv('dataset/drugbank/drugs_info_5_1_8.csv')
    data = data[['drugbank_id', 'smiles']]
    data = data.dropna(axis=0)
    data.to_csv('dataset/drugbank/drug_smiles.tsv', sep='\t', index=False, columns=None)

def save_triples(triples, mode='train'):
    data = open('dataset/hetionet/{}.tsv'.format(mode), 'w')
    for t in triples:
        data.write('{}\t{}\t{}\n'.format(t[0],t[1],t[2]))
    data.close()

def split_hetionet():
    triples = []
    with open('dataset/hetionet/triples.tsv', 'r') as f:
        for line in f:
            infos = line.strip().split('\t')
            triples.append([int(infos[0]), int(infos[1]), int(infos[2])])
    
    train = triples[:int(len(triples)*0.9)]
    val = triples[int(len(triples)*0.9):int(len(triples)*0.95)]
    test = triples[int(len(triples)*0.95):]
    save_triples(train)
    save_triples(val, mode='valid')
    save_triples(test, mode='test')

#process_hetionet()
hop = 1
missing = 0
process_drugbank()
print('missing data: ', missing)
## split kg into train/valid/test
# split_hetionet()
# process_drug_smiles()