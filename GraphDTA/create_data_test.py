import argparse
import gzip
import sys
import time
import global_data
import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from joblib import Parallel, delayed
from torch_geometric.data import InMemoryDataset, Data

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1500


def handle_fasta(data_path):
    seq = ''
    with open(data_path) as reader:
        lines = reader.readlines()
    for line in lines:
        if line[0] == '>':
            continue
        else:
            seq = seq + line.strip()
    return seq


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
  
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


# def create_data(dataset_name, fasta_path):
#     print('start')
#
#     compound_iso_smiles = []
#
#     opts = ['test_drop']
#     for opt in opts:
#         df = pd.read_csv(self.root + '/' + opt + '.csv', header=0)
#         compound_iso_smiles += list(df['compound_iso_smiles'])
#     compound_iso_smiles = set(compound_iso_smiles)
#     smile_graph = {}
#     if os.path.exists(self.root + '/graph_dict.pkl.gz'):
#         print('load graph dict from ', self.root + '/graph_dict.pkl.gz')
#         # with open(self.root + '/graph_dict.pkl', 'rb') as file:
#         #     smile_graph = pickle.load(file)
#         with open(self.root + '/graph_dict.pkl.gz', 'rb') as reader:
#             smile_graph = pickle.loads(gzip.decompress(reader.read()))
#     else:
#         print('generate graph dict to ', self.root + '/graph_dict.pkl.gz')
#         for smile in compound_iso_smiles:
#             g = smile_to_graph(smile)
#             smile_graph[smile] = g
#         # with open(self.root + '/graph_dict.pkl', 'wb') as file:
#         #     pickle.dump(smile_graph, file)
#         with open(self.root + '/graph_dict.pkl.gz', 'wb') as writer:
#             writer.write((gzip.compress(pickle.dumps(smile_graph))))
#     print('smiles graph generate sucess!')
#     # convert to PyTorch data format
#
#     processed_data_file_test = 'data/processed/' + dataset_name + '_test_drop.pt'
#
#     if ((not os.path.isfile(processed_data_file_test))):
#
#         df = pd.read_csv(self.root + '/' + 'test_drop.csv')
#         test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
#             df['affinity'])
#         XT = [seq_cat(t) for t in test_prots]
#         test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
#         print('csv read sucess!')
#
#         # make data PyTorch Geometric ready
#         print('preparing ', dataset_name + '_test.pt in pytorch format!')
#         # test_data = TestbedDataset(root=self.root, dataset='test_drop', xd=test_drugs, xt=test_prots,
#         #                            y=test_Y,
#         #                            smile_graph=smile_graph)
#         print(processed_data_file_test, ' have been created')
#     else:
#         print(processed_data_file_test, ' are already created')
#         return

class TestbedDataset(InMemoryDataset):
    def __init__(self, root, phase='test', transform=None, pre_transform=None, pre_filter=None):

        super().__init__(root, transform, pre_transform, pre_filter)
        # if phase == 'test':
        #     self.data, self.slices = torch.load(self.processed_paths[0])
        if phase == 'test':
            self.process()

    @property
    def raw_file_names(self):
        return ['test_drop.csv', ]
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        pass

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # def _get_graph(self, smile):
    #     g = smile_to_graph(smile)
    #     smile_graph[smile] = g

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self):
        print('start')
        time_start = time.time()
        compound_iso_smiles = []

        df = pd.read_csv(self.raw_paths[0], header=0)
        compound_iso_smiles += list(df['compound_iso_smiles'])
        compound_iso_smiles = list(set(compound_iso_smiles))
        if os.path.exists(self.root + '/graph_dict.pkl.gz') and os.path.getsize(
                self.root + '/graph_dict.pkl.gz') != 0:
            print('load graph dict from ', self.root + '/graph_dict.pkl.gz')
            # with open(self.root + '/graph_dict.pkl', 'rb') as file:
            #     smile_graph = pickle.load(file)
            with open(self.root + '/graph_dict.pkl.gz', 'rb') as reader:
                # file_gz = reader.read()
                # print('get gz success')
                # file_pkl = gzip.decompress(file_gz)
                # print('gz decompress success')
                # smile_graph = pickle.loads(file_pkl)
                # print('generate graph dict success')

                smile_graph = pickle.loads(gzip.decompress(reader.read()))
        else:
            print('generate graph dict to ', self.root + '/graph_dict.pkl.gz')
            smile_graph = {}
            for smile in compound_iso_smiles:
                g = smile_to_graph(smile)
                smile_graph[smile] = g
            # for smile in compound_iso_smiles:
            #
            #     smile_graph[smile] = delayed(smile_to_graph)(smile)
            # multi_work = Parallel(n_jobs=-1, backend='threading')
            # res = multi_work(smile_graph)
            # smile_graph = {
            #     zip(compound_iso_smiles, Parallel(n_jobs=-1, backend="threading")(
            #         delayed(smile_to_graph)(smile) for smile in compound_iso_smiles))}
            # with open(self.root + '/graph_dict.pkl', 'wb') as file:
            #     pickle.dump(smile_graph, file)
            with open(self.root + '/graph_dict.pkl.gz', 'wb') as writer:
                writer.write((gzip.compress(pickle.dumps(smile_graph))))
        print('smiles graph generate sucess!')
        # convert to PyTorch data format
        target = seq_cat(handle_fasta(global_data.get_fasta_path()))

        test_drugs = list(df['compound_iso_smiles'])

        test_drugs = np.asarray(test_drugs)
        print('csv read sucess!')

        data_list = []
        data_len = len(test_drugs)

        for i in range(data_len):
            smiles = test_drugs[i]
            # labels = test_Y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            try:
                GCNData = Data(x=torch.Tensor(np.array(features)),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    # y=torch.FloatTensor([labels]),
                                    target=torch.LongTensor(np.array([target])))
            except:
                print(smiles)

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)
        # print('smile_graph ', sys.getsizeof(smile_graph) / 1024 / 1024 / 1024, 'GB')
        del smile_graph
        del test_drugs, compound_iso_smiles, df
        # print('data_list   ', sys.getsizeof(data_list) / 1024 / 1024 / 1024, 'GB')

        # keys = dir()
        # for variable in keys:
        #     print(variable, sys.getsizeof(eval(variable)) / 1024 / 1024 / 1024, 'GB', '\n')
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        self.data, self.slices = self.collate(data_list)
        # print('self.data   ', sys.getsizeof(self.data) / 1024 / 1024 / 1024, 'GB')
        # print('self.slices   ', sys.getsizeof(self.slices) / 1024 / 1024 / 1024, 'GB')

        # save preprocessed data:
        # torch.save((data, slices), self.processed_paths[0])
        time_end = time.time()
        print('generate data time use: ', (time_end - time_start) / 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph data generate arguments')
    parser.add_argument(
        '--data_root',
        type=str,
        help='the root of test dataset')
    parser.add_argument(
        '--fasta_path',
        type=str,
        help='the path of protein fasta')

    args = parser.parse_args()
    fasta_path = args.fasta_path

    args = parser.parse_args()
    TestbedDataset(args.data_root, 'test')
