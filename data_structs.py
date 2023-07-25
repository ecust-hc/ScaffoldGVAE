import torch.utils.data as data
from utils import canonicalize_smiles_from_file,replace_halogen,construct_vocabulary,create_var,one_hot
import re
import  numpy as np
import torch
from rdkit import Chem

class MolData(data.Dataset):
    def __init__(self, root_path, voc ):
        self.voc = voc
        self.root_path = root_path
        self.mols, self.scas = canonicalize_smiles_from_file(self.root_path)

    def __getitem__(self, index):
        mol = self.mols[index]
        sca = self.scas[index]

        tokenized = self.voc.tokenize(sca)
        encoded = self.voc.encode(tokenized)
        return create_var(encoded),mol,sca

    def __len__(self):
        return len(self.mols)

    @classmethod
    def collate_fn(self, batch):
        max_length = max([seq.size(0) for seq, _, _ in batch])
        collated_arr = create_var(torch.zeros(len(batch), max_length))
        mol_batch = []
        sca_batch = []

        i =0
        for  seq, mol, sca in batch:
            collated_arr[i, :seq.size(0)] = seq
            mol_batch.append(mol)
            sca_batch.append(sca)

            i+=1


        return mol_batch,sca_batch,collated_arr

class Vocabulary(object):
    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ["EOS","GO"]
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('EOS')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

if __name__ == "__main__":
    data_path = "D:\Python\ProjectOne\data\data.txt"
    voc_chars , max_len = construct_vocabulary(data_path)
    voc = Vocabulary(init_from_file='data/Voc',max_length=max_len)
    print(voc_chars)
    print(max_len)
    moldata = MolData(data_path,voc)
    test = moldata.__getitem__(0)
    print(test)

