import numpy as np
import scaffoldgraph as sg
import networkx as nx
import os
from rdkit.Chem import Draw
from rdkit import Chem
from utils import get_mol
import matplotlib.pyplot as plt
import random
import argparse

def filter_sca(sca):
    mol = Chem.MolFromSmiles(sca)
    if mol == None:
        return None
        print("WUXIAO")
    ri = mol.GetRingInfo()
    benzene_smarts = Chem.MolFromSmiles("c1ccccc1")
    if ri.NumRings() == 1 and mol.HasSubstructMatch(benzene_smarts) == True:
        return None
    elif mol.GetNumHeavyAtoms() > 20:
        return None
    elif Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) > 3:
        return None
    else:
        return True

def mol_if_equal_sca (mol,sca):
    S_sca = []
    #Chem.Kekulize(mol)
    smile = get_mol(mol)
    sca_mol = get_mol(sca)
    # smile =  Chem.MolFromSmiles(mol)
    # sca_mol = Chem.MolFromSmarts(sca)
    if smile == None or sca_mol == None :
        return False
    else:
        n_atoms = smile.GetNumAtoms()
        index = smile.GetSubstructMatch(sca_mol)
        for i in range(n_atoms):
            if i in index:
                S_sca.append(1)
            else:
                S_sca.append(0)
        arr = np.array(S_sca)
        if (arr == 1).all() == True or (arr == 0).all() == True:
            return False
        else:
            return True

def main(args):

    network = sg.ScaffoldNetwork.from_smiles_file(args.data)
    n_scaffolds = network.num_scaffold_nodes
    n_molecules = network.num_molecule_nodes
    print('\nGenerated scaffold network from {} molecules with {} scaffolds\n'.format(n_molecules, n_scaffolds))
    scaffolds = list(network.get_scaffold_nodes())
    molecules = list(network.get_molecule_nodes())
    counts = network.get_hierarchy_sizes()  # returns a collections Counter object
    lists = sorted(counts.items())
    x, y = zip(*lists)
    plt.figure(figsize=(8, 6))
    plt.bar(x, y)
    plt.xlabel('Hierarchy')
    plt.ylabel('Scaffold Count')
    plt.title('Number of Scaffolds per Hierarchy (Network)')
    plt.show()

    with open(args.save_dir, "w") as f:
        total = 0
        for pubchem_id in molecules:
            predecessors = list(nx.bfs_tree(network, pubchem_id, reverse=True))
            smile = network.nodes[predecessors[0]]['smiles']
            # 获取smiles_list中元素为smile的索引号

            sca = []
            for i in range(1, len(predecessors)):
                if filter_sca(predecessors[i]) is not None:
                    sca.append(predecessors[i])

            if len(sca) != 0:
                random_sca = random.choice(sca)
                # for i in range(len(sca)):
                #     if mol_if_equal_sca(smile,sca[i]) is True:
                #             f.write(smile + " " + sca[i] +'\n')
                #             total +=1
                if mol_if_equal_sca(smile, random_sca) is True:
                    f.write(smile + " " + random_sca + '\n')
                    total += 1
            # index +=1
    print('\nthe pairs of molecula and scaffold have {} \n'.format(total))

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Neural message passing and rnn')
    parser.add_argument('--data', default='./data/chembl31.smi', help='dataset path')
    parser.add_argument('--save_dir', default='./data/chembl31_sca_randone.smi', help='save model path')
    args = parser.parse_args()
    main(args)
