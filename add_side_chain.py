from rdkit import Chem
from rdkit.Chem import AllChem
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.DataStructs import cDataStructs
from rdkit.DataStructs import BulkTanimotoSimilarity, BulkDiceSimilarity
from tqdm import tqdm


def calc_fingerprint(smiles, fingerprint_type='ECFP'):
#     mol = Chem.MolFromSmiles(smiles)
    mol = smiles
    if fingerprint_type == 'Topological':
        return Chem.RDKFingerprint(mol)
    elif fingerprint_type == 'MACCS':
        return MACCSkeys.GenMACCSKeys(mol)
    elif fingerprint_type == 'AtomPairs':
        return Pairs.GetAtomPairFingerprintAsBitVect(mol)
    elif fingerprint_type == 'TopologicalTorsions':
        int_vect = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
        int_array = np.zeros((1,), dtype=np.int32)
        cDataStructs.ConvertToNumpyArray(int_vect, int_array)
        bit_vect = np.unpackbits(int_array.view(np.uint8))
        return bit_vect
    elif fingerprint_type == 'ECFP':
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True)
    elif fingerprint_type == 'FCFP':
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False)
    else:
        raise ValueError(f'Unsupported fingerprint type: {fingerprint_type}')

def calc_similarity_matrix(ref_mol, input_list, fingerprint_type='ECFP', similarity_metric='Tanimoto', k=2):
    # Generate molecular fingerprints for each SMILES string
    #print('Computing fingerprint...')
    ref_fp = calc_fingerprint(ref_mol, fingerprint_type)
    fingerprints = [calc_fingerprint(smiles, fingerprint_type) for smiles in input_list]
    # Initialize an empty similarity matrix
    similarity_matrix = np.zeros((len(input_list), len(input_list)))
    # Choose the similarity metric function
    if similarity_metric == 'Tanimoto':
        similarity_func = BulkTanimotoSimilarity
    elif similarity_metric == 'Dice':
        similarity_func = BulkDiceSimilarity
    else:
        raise ValueError(f'Unsupported similarity metric: {similarity_metric}')
    # Compute the similarity coefficients using bulk operations
    #print('Computing similarity...')
    similarity_coeffs = similarity_func(ref_fp, fingerprints)
    return similarity_coeffs


def get_neiid_bysymbol(mol, marker):
    try:
        if type(marker) == str:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == marker:
                    neighbors = atom.GetNeighbors()
                    if len(neighbors) > 1:
                        print('Cannot process more than one neighbor, will only return one of them')
                    atom_nb = neighbors[0]
                    return atom_nb.GetIdx()
        elif type(marker) == int:
            for atom in mol.GetAtoms():
                if atom.GetIntProp('id_oc') == marker:
                    neighbors = atom.GetNeighbors()
                    if len(neighbors) > 1:
                        print('Cannot process more than one neighbor, will only return one of them')
                    atom_nb = neighbors[0]
                    return atom_nb.GetIdx()
    except Exception as e:
        print(e)
        return None


def get_id_bysymbol(mol, marker):
    for atom in mol.GetAtoms():
        if type(marker) == str:
            if atom.GetSymbol() == marker:
                return atom.GetIdx()
        elif type(marker) == int:
            for atom in mol.GetAtoms():
                if atom.GetIntProp('id_oc') == marker:
                    return atom.GetIdx()


def combine2frags(mol_a, mol_b, maker_a='H', maker_b='*'):
    # 将两个待连接分子置于同一个对象中
    merged_mol = Chem.CombineMols(mol_a, mol_b)
    bind_pos_a = get_neiid_bysymbol(merged_mol, maker_a)
    bind_pos_b = get_neiid_bysymbol(merged_mol, maker_b)
    # 转换成可编辑分子，在两个待连接位点之间加入单键连接，特殊情形需要其他键类型的情况较少，需要时再修改
    ed_merged_mol = Chem.EditableMol(merged_mol)
    ed_merged_mol.AddBond(bind_pos_a, bind_pos_b, order=Chem.rdchem.BondType.SINGLE)
    # 将图中多余的marker原子逐个移除，先移除marker a
    marker_a_idx = get_id_bysymbol(merged_mol, maker_a)
    ed_merged_mol.RemoveAtom(marker_a_idx)
    # marker a移除后原子序号变化了，所以又转换为普通分子后再次编辑，移除marker b
    temp_mol = ed_merged_mol.GetMol()
    marker_b_idx = get_id_bysymbol(temp_mol, maker_b)
    ed_merged_mol = Chem.EditableMol(temp_mol)
    ed_merged_mol.RemoveAtom(marker_b_idx)
    final_mol = ed_merged_mol.GetMol()
    return final_mol


def scaffold_hop(smiles, core_smiles, new_core_smiles):
    # 创建化合物分子
    compound = Chem.MolFromSmiles(smiles)
    # Chem.rdmolops.KekulizeIfPossible(compound, clearAromaticFlags=True)
    for id_o, atom in enumerate(compound.GetAtoms()):
        atom.SetIntProp('id_o', id_o)

    # 原始母核
    core = Chem.MolFromSmiles(core_smiles)
    # Chem.rdmolops.KekulizeIfPossible(core, clearAromaticFlags=True)
    # 新母核
    new_core = Chem.MolFromSmiles(new_core_smiles)
    new_core_with_h = Chem.AddHs(new_core)

    for id_o, atom in enumerate(new_core_with_h.GetAtoms()):
        atom.SetIntProp('id_oc', id_o)

    #print('New Core: ', Chem.MolToSmiles(new_core_with_h))

    id_h = [atom.GetIdx() for atom in new_core_with_h.GetAtoms() if atom.GetAtomicNum() == 1]

    # 切掉原始母核并获取所有侧链R-group
    r = AllChem.ReplaceCore(compound, core)
    try:
        side_mols = Chem.GetMolFrags(r, asMols=True)
    except:
        Chem.Kekulize(compound, clearAromaticFlags=True)
        Chem.Kekulize(core, clearAromaticFlags=True)

        for id_o, atom in enumerate(compound.GetAtoms()):
            atom.SetIntProp('id_o', id_o)

        new_core = Chem.MolFromSmiles(new_core_smiles)
        new_core_with_h = Chem.AddHs(new_core)

        for id_o, atom in enumerate(new_core_with_h.GetAtoms()):
            atom.SetIntProp('id_oc', id_o)

        # print('New Core: ', Chem.MolToSmiles(new_core_with_h))

        id_h = [atom.GetIdx() for atom in new_core_with_h.GetAtoms() if atom.GetAtomicNum() == 1]

        # 切掉原始母核并获取所有侧链R-group
        r = AllChem.ReplaceCore(compound, core)
        side_mols = Chem.GetMolFrags(r, asMols=True)

        side_mols_tmp = []
        for m in side_mols:
            if Chem.MolToSmiles(m).count('*') > 1:
                star_dict = {}
                for atom in m.GetAtoms():
                    if '*' in atom.GetSmarts():
                        star_dict[atom.GetSmarts()] = atom.GetIdx()
                new_dict = dict(sorted(star_dict.items()))
                last_key = list(new_dict)[0]
                del new_dict[last_key]

                remove_idx = [data for key, data in (new_dict.items())]

                ed_m = Chem.EditableMol(m)
                for atom_idx in sorted(remove_idx, reverse=True):
                    ed_m.RemoveAtom(atom_idx)
                side_mols_tmp.append(ed_m.GetMol())
            else:
                side_mols_tmp.append(m)

        side_mols = side_mols_tmp

    id_h_combinations = itertools.permutations(id_h, len(side_mols))
    id_h_combinations_list = list(id_h_combinations)

    side_mols_dict = {}
    side_mols_connect_atom_dict = {}
    for m in side_mols:
        side_id = int(Chem.MolToSmiles(m).split('*')[0][1:])
        side_mols_dict[side_id] = m

        for atom in m.GetAtoms():
            if (atom.GetAtomicNum() == 0):
                side_mols_connect_atom_dict[side_id] = atom.GetNeighbors()[0].GetIntProp('id_o')


    # 在新母核上依次装上R-group
    new_mol = []
    rmsd = []
    for list_tmp in id_h_combinations_list:
        new_core_with_h_tmp = deepcopy(new_core_with_h)
        for m in side_mols:
            side_id = int(Chem.MolToSmiles(m).split('*')[0][1:])
            new_core_with_h_tmp = combine2frags(new_core_with_h_tmp, m, maker_a=list_tmp[side_id - 1], maker_b='*')
            
        new_mol.append(new_core_with_h_tmp)
#         rmsd.append(mol_similarity(new_core_with_h_tmp,compound))
    
    sim = calc_similarity_matrix(compound, new_mol, fingerprint_type='Topological', similarity_metric='Tanimoto', k=2)
    sim_max_id = np.argmax(sim)

        # print('R-group: ', Chem.MolToSmiles(m))
        #print('Update Mol: ', Chem.MolToSmiles(new_core_with_h))

    return Chem.MolToSmiles(new_mol[sim_max_id])

def main():
    compound = "N#C[C@@H]1CCCC[C@@H]1n1cc(C(N)=O)c(Nc2ccc(F)nc2)n1"
    core = "NC(=O)c1cn([H])nc1N[H]"
    add_side = []
    scaff = pd.read_csv("D:\Python\Project_VAE\data\JAK1\\test_sample_JAK1.csv")
    scaff_list = scaff['SMILES'].tolist()
#    scaff_list = ['c1cccnc1']
    for i in range(len(scaff_list)):
        print(i)
        sca_mol = Chem.MolFromSmiles(scaff_list[i])
        if sca_mol is None:
            add_side.append(scaff_list[i])
        else:
            new = scaffold_hop(compound, core, scaff_list[i])
            try:
                new = scaffold_hop(compound, core, scaff_list[i])
                add_side.append(new)
            except:
                add_side.append(scaff_list[i])
            

    #将结果保存到csv文件中
    print("add side end")
    df_smiles = pd.DataFrame()
    df_smiles['SMILES'] = add_side
    print(add_side)
    df_smiles.to_csv("D:\Python\Project_VAE\data\JAK1\\test_sample_JAK1_side.csv", index=None)

if __name__ == "__main__":
    main()