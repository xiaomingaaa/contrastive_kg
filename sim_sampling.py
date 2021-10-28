'''
Author: your name
Date: 2021-10-05 15:18:03
LastEditTime: 2021-10-05 15:18:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /molecular_representation/sim_sampling.py
'''
import json, os, time
import multiprocessing as mp
import random
from json import JSONEncoder
import argparse

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from tqdm import tqdm

from chem import *


class NumpyArrayEncoder(JSONEncoder):
    """Rewrite the class
    To serialize the ndarray
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def cos_sim(vector_a, vector_b):
    """
    Cosine similarity
    Args:
        - vector_a
        - vector_b: has same dimension of v_a
    Return:
        - sim in float type
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def prepare_similar_pairs(smiles, vocab, task='fingerprint, onehot', threshold=0.5, n_samples=3000000) -> set:
    """Unified threshold.

    Can be paralleled.
    Returns:
        - out_set: set of indices corresponding to smiles list
    """
    #print('Rank[{}/{}] entering subroutine.'.format(rank, num_procs))
    # border of space to be searched
    #left = rank * (len(smiles) // num_procs)
    #right = left + (len(smiles) // num_procs)

    index_sets = []
    if 'fingerprint' in task:
        id_set = []
        # upper triangle similarity matrix
        # print('Creating the pairs for fingerprint task.')
        stop_flag = False
        for i in tqdm(range(len(smiles))):
            for j in tqdm(range(i+1, len(smiles))):
                fps_i = AllChem.GetMorganFingerprintAsBitVect(get_mol(smiles[i]), 2, nBits=1024)
                fps_j = AllChem.GetMorganFingerprintAsBitVect(get_mol(smiles[j]), 2, nBits=1024)
                sim = DataStructs.FingerprintSimilarity(fps_i,fps_j)
                # sim_array[i][j] = sim     # sim(i,j)(j>i)
                if sim > threshold:
                    id_set.append((i,j))
                if len(id_set) > n_samples:
                    stop_flag = True
            print("Obtained pair n: ", len(id_set))
            if stop_flag:
                break
        index_sets.append(set(id_set))
    if 'onehot' in task:
        id_set = []
        stop_flag = False
        # upper triangle similarity matrix
        # print('Creating the pairs for substructure task.')
        for i in tqdm(range(len(smiles))):
            for j in tqdm(range(i+1, len(smiles))):
                oh_i = mol2onehot(get_mol(smiles[i]), vocab)
                oh_j = mol2onehot(get_mol(smiles[j]), vocab)
                sim = cos_sim(oh_i, oh_j)
                # sim_array[i][j] = sim     # sim(i,j)(j>i)
                if sim > threshold:
                    id_set.append((i,j))
                if len(id_set) > n_samples:
                    stop_flag = True
            print("Obtained pair n: ", len(id_set))
            if stop_flag:
                break
        index_sets.append(set(id_set))
    if len(index_sets) > 1:
        out_set = index_sets[0]
        for id_set in index_sets[1:]:
            out_set = out_set.intersection(id_set)
    else:
        out_set = index_sets[0]
    print('Positive samples n: ', len(out_set))
    return out_set


def negative_sampling(smiles, vocab, thres_inf=0.2, thres_sup=0.5, n_samples=1500000) -> set:
    """Unified threshold.
    Intersection of all task
    
    Returns:
        - out_set: set of indices corresponding to smiles list
    """
    negative_id = []
    stop_flag = False
    # upper triangle similarity matrix
    print('Creating the pairs for negative sampling.')
    for i in tqdm(range(len(smiles))):
        for j in tqdm(range(i+1, len(smiles))):
            fps_i = AllChem.GetMorganFingerprintAsBitVect(get_mol(smiles[i]), 2, nBits=1024)
            fps_j = AllChem.GetMorganFingerprintAsBitVect(get_mol(smiles[j]), 2, nBits=1024)
            fps_sim = DataStructs.FingerprintSimilarity(fps_i,fps_j)
            oh_i = mol2onehot(get_mol(smiles[i]), vocab)
            oh_j = mol2onehot(get_mol(smiles[j]), vocab)
            oh_sim = cos_sim(oh_i,oh_j)
            if (thres_inf < oh_sim < thres_sup) and (thres_inf < fps_sim < thres_sup):
                negative_id.append((i,j))
            if len(negative_id) > n_samples:
                    stop_flag = True
        print("Obtained pair n: ", len(negative_id))
        if stop_flag:
            break
    print('Negative sample n: ', len(negative_id))
    return negative_id


# neg_id = negative_sampling(smiles,onehot_dict)

# For siameseNet / pretraining
# length = 100 - 500 smiles for training
# under pretrain_smiles_52537 as pool
# neg_indices: negative samples
# cand_indices: filtered by morgan_fp and one-hot similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling',   type=str, default='positive')
    args = parser.parse_args()

    os.chdir('./data/pretrain')
    with open('smiles_100k.json','r') as f:
        smiles = json.load(f)
    with open('vocab_0117.json','r') as f:
        vocab = json.load(f)
    print('Loaded smiles n: ', len(smiles))
    print('Loaded vocab length: ', len(vocab))

    if args.sampling is 'positive':
        pos_set = prepare_similar_pairs(smiles, vocab)
        with open('positive_{}.json'.format(len(pos_set)), 'w') as f:
            json.dump(pos_set, f)
    else:
        neg_set = negative_sampling(smiles, vocab)
        with open('negative_{}.json'.format(len(neg_set)), 'w') as f:
            json.dump(neg_set, f)