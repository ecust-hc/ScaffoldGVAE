from rdkit import Chem
import re
from torch.autograd import Variable
import torch
import numpy as np
from multiprocessing import Pool
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from Metrics.SA_Score import sascorer
from Metrics.NP_Score import npscorer
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors
import scaffoldgraph as sg
import networkx as nx
import requests
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import scipy.sparse
import os
#from use_pretrain_model import encoder
#encoder = encoder()
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn',
             'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6
def fingerprint(smiles_or_mol, fp_type='morgan', dtype=None, morgan__r=2, morgan__n=1024):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits
    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=True, **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)

    fps = mapper(n_jobs)(fingerprint, smiles_mols_array)

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps

_base_dir = "D:\PythonProject\ScaffoldGVAE\Metrics"
_mcf = pd.read_csv(os.path.join(_base_dir, 'mcf.csv'))
_pains = pd.read_csv(os.path.join(_base_dir, 'wehi_pains.csv'),
                     names=['smarts', 'names'])
_filters = [Chem.MolFromSmarts(x) for x in
            _mcf.append(_pains, sort=True)['smarts'].values]
def mol_passes_filters(mol,
                       allowed=None,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    mol = get_mol(mol)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True

def fraction_passes_filters(gen, n_jobs=1):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)

def calc_self_tanimoto(gen_vecs, agg='max', device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules
    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"

    # Initialize output array and total count for mean aggregation
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))

    # Convert input vectors to PyTorch tensors and move to the specified device
    x_gen = torch.tensor(gen_vecs).to(device).half()
    y_gen = torch.tensor(gen_vecs).to(device).half()

    # Transpose x_stock tensor
    y_gen = y_gen.transpose(0, 1)

    # Calculate Tanimoto similarity using matrix multiplication
    tp = torch.mm(x_gen, y_gen)
    jac = (tp / (x_gen.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp))

    # Handle NaN values in the Tanimoto similarity matrix
    jac = jac.masked_fill(torch.isnan(jac), 1)

    # Delete the elements on the eye (self-self similarity)
    jac = jac[~np.eye(jac.shape[0], dtype=bool)]
    jac = jac.reshape(jac.shape[0], -1)

    if p != 1:
        jac = jac ** p

    # Aggregate scores from this batch
    if agg == 'max':
        # Aggregate using max
        agg_tanimoto = jac.max(1)[0].cpu().numpy()
    elif agg == 'mean':
        # Aggregate using mean
        agg_tanimoto = jac.mean(1).cpu().numpy()

    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)

    return agg_tanimoto

def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        scas_list = []

        for i, line in enumerate(f):
            smiles = line.split(" ")[0]

            scas = line.split(" ")[1].strip("\n")

            mol = Chem.MolFromSmiles(smiles)
            sca = Chem.MolFromSmiles(scas)

            if mol:
                smiles_list.append(Chem.MolToSmiles(mol))
            else:
                raise ValueError(f'Cannot be rdkit analysis "{mol}" rdkit analysis.')
            if sca:
                scas_list.append(Chem.MolToSmiles(sca))
            else:
                raise ValueError(f'Cannot be rdkit analysis "{sca}" rdkit analysis.')
        print("{} SMILES retrieved".format(len(smiles_list)))
        print("{} scas retrieved".format(len(scas_list)))

        if len(smiles_list) != len(scas_list)  :
            raise ValueError('The length of smiles_list is not match with the length of scas_list and cluster_list.')
        return smiles_list, scas_list

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string

def construct_vocabulary(path,voc_path):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    smiles_list, _ = canonicalize_smiles_from_file(path)
    add_chars = set()

    # max_len = 0
    for i, smiles in enumerate(smiles_list):
        chars_list = []
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
                chars_list.append(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]
                [chars_list.append(unit) for unit in chars]
        # if max_len < len(chars_list):
        #     max_len = len(chars_list)

    print("Number of characters: {}".format(len(add_chars)))
    with open(voc_path, 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

def get_mol(smiles):
    if type(smiles) == float:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Chem.Kekulize(mol)
    return mol

def create_var(tensor, requires_grad=None):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if requires_grad is None:
        return Variable(tensor).cuda()
    else:
        return Variable(tensor, requires_grad=requires_grad).cuda()

def one_hot(mol,size):
    mol_one_hot = torch.zeros(len(mol), size)
    for i in range(len(mol)):
        mol_one_hot[i,int(mol[i])] = 1
    return mol_one_hot

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                        + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
                        + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    return torch.Tensor(fbond + fstereo)

def atom_if_sca(mol_batch,sca_batch):
    S_sca = []
    scope = []
    total_atoms = 0
    for i in range(len(mol_batch)):
        smile = mol_batch[i]
        # smile_mol = Chem.MolFromSmiles(smile)
        smile_mol = get_mol(smile)
        sca = sca_batch[i]
        # sca_mol = Chem.MolFromSmarts(sca)
        sca_mol = get_mol(sca)
        n_atoms = smile_mol.GetNumAtoms()
        index = smile_mol.GetSubstructMatch(sca_mol)
        for i in range(n_atoms):
            if i in index:
                S_sca.append(1)
            else:
                S_sca.append(0)
        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms
    return S_sca,scope

def mol2graph(mol_batch):
    padding = torch.zeros(BOND_FDIM)
    fatoms, fbonds = [], [padding]  # Ensure bond is 1-indexed
    out_bonds,in_bonds, all_bonds = [], [], [(-1, -1)]  # Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    i = 0
    for smiles in mol_batch:
        mol = get_mol(smiles)
        # mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])
            out_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds)
            all_bonds.append((x, y))
            #fbonds.append(torch.cat([fatoms[y], bond_features(bond)], 0))
            fbonds.append(bond_features(bond))
            in_bonds[y].append(b)
            out_bonds[x].append(b)

            b = len(all_bonds)
            all_bonds.append((y, x))
            fbonds.append(bond_features(bond))
            in_bonds[x].append(b)
            out_bonds[y].append(b)

        scope.append((total_atoms, n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    aoutgraph = torch.zeros(total_atoms, MAX_NB).long()
    aingraph = torch.zeros(total_atoms, MAX_NB).long()
    bgraph = torch.zeros(total_bonds, MAX_NB).long()

    for a in range(total_atoms):
        for i, b in enumerate(out_bonds[a]):
            aoutgraph[a, i] = b
        for i, b in enumerate(in_bonds[a]):
            aingraph[a, i] = b

    for b1 in range(1, total_bonds):
        x, y = all_bonds[b1]
        for i, b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1, i] = b2

    return fatoms, fbonds, aoutgraph, bgraph, aingraph, scope, all_bonds
def valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:          # check validity
        return False
    try:                     # check valence, aromaticity, conjugation and hybridization
        Chem.SanitizeMol(mol)
    except:
        return False
    return True

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

class KLAnnealer:
    def __init__(self, n_epoch, kl_w_end, kl_w_start, kl_start =0 ):
        self.i_start = kl_start
        self.w_start = kl_w_start
        self.w_max = kl_w_end
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map

def read_smiles_csv(path, sep=','):
    return pd.read_csv(path, usecols=['SMILES'], sep=sep).squeeze('columns').astype(str).tolist()

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return smiles_or_mol

def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()

def compute_scaffold(mol, min_rings=2):
    """
    Extracts a scafold from a molecule in a form of a canonic SMILES
    """
    mols = get_mol(mol)
    if mols is None:
        print(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mols)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles

def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    scaffolds = []
    for mol_scaf in mapper(n_jobs)(compute_scaffold, mol_list):
        if mol_scaf is not None:
            scaffolds.append(mol_scaf)
    return list(set(scaffolds))

def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)

def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)

def NP(mol):
    """
    Computes RDKit's Natural Product-likeness score
    """
    return npscorer.scoreMol(mol)

def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)

def Weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))

def filter_sca(sca):
    mol = Chem.MolFromSmiles(sca)
    ri = mol.GetRingInfo()
    if ri.NumRings() <2:
        return None
    elif mol.GetNumHeavyAtoms() >20:
        return None
    elif Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) >3:
        return None
    else:
        return True

#loss.forward bug:mol equal sca,index tensor of side is null ,default float32 not int so error add_index need int but get float
def mol_if_equal_sca (mol,sca):
    S_sca = []
    smile = get_mol(mol)
    sca = get_mol(sca)
    if mol == None or sca == None :
        return False
    else:
        n_atoms = smile.GetNumAtoms()
        index = smile.GetSubstructMatch(sca)
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

def ext_sca(seqs, agent_likelihood ,seq_batch,pic50,protein):
    # m = Chem.MolFromSmiles(mol[0])
    # patt = Chem.MolFromSmarts(sca[0])
    # rm = Chem.DeleteSubstructs(m, patt)
    # frag = rm
    # # frag_mol = Chem.MolFromSmart(frag)
    # mol_batch = []
    # sca_batch = []
    # likelihood = []
    # pic50_batch = []
    # protein_batch = []
    # encode_batch = []
    # for i in range(len(seqs)):
    #     mol_i = Chem.MolFromSmiles(seqs[i])
    #     sca_i = Chem.DeleteSubstructs(mol_i, frag)
    #     mol_batch.append(seqs[i])
    #     sca_batch.append(Chem.MolToSmiles(sca_i))
    #     likelihood.append(agent_likelihood[i])
    #     pic50_batch.append(pic50[0])
    #     protein_batch.append(protein[0])
    #     encode_batch.append(seq_batch[i])
    #
    # return mol_batch, sca_batch, likelihood, encode_batch, pic50_batch, protein_batch

    with open("data\\reinforce_transition.smi", "w") as f:
        for i in range(len(seqs)):
            f.write(seqs[i] + "\n" )
    aeq_lik = dict(zip(seqs, agent_likelihood))
    encode_batch = []
    network = sg.ScaffoldNetwork.from_smiles_file("data\\reinforce_transition.smi")
    scaffolds = list(network.get_scaffold_nodes())
    molecules = list(network.get_molecule_nodes())
    mol_batch = []
    sca_batch =[]
    likelihood = []
    pic50_batch = []
    protein_batch = []
    index =0
    for pubchem_id in molecules:
        predecessors = list(nx.bfs_tree(network, pubchem_id, reverse=True))
        smile = network.nodes[predecessors[0]]['smiles']
        sca = 0
        for i in range(1, len(predecessors)):
            if filter_sca(predecessors[i]) is not None:
                sca = predecessors[i]
                break
        if sca != 0 and mol_if_equal_sca(smile, sca) and smile == seqs[index]:
            mol_batch.append(smile)
            sca_batch.append(sca)
            likelihood.append(agent_likelihood[index])
            encode_batch.append(seq_batch[index])
            pic50_batch.append(pic50[0])
            protein_batch.append(protein[0])
        index +=1
    return mol_batch,sca_batch,likelihood,encode_batch,pic50_batch,protein_batch

def side_no_sca_change(smile,mol,sca):
    m = Chem.MolFromSmiles(mol)
    patt = Chem.MolFromSmiles(sca)
    rm = Chem.DeleteSubstructs(m, patt)
    frag = Chem.MolToSmiles(rm)
    mol = Chem.MolFromSmiles(smile)
    if mol.HasSubstructMatch(rm):
        return True
    else:
        return False


def download_fasta_from_uniprot(protein_name):
    protein_name = protein_name
    base_url = "https://www.uniprot.org/uniprot/"
    query = f"{protein_name}.fasta"
    url = base_url + query
    response = requests.get(url)

    if response.status_code == 200:
        fasta_sequence = response.text
        return fasta_sequence
    else:
        print(f"Error: Unable to download FASTA sequence for protein '{protein_name}'.")
        return None

if __name__ == "__main__":
    seqs = ["CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1","C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1","COc1ccc(N2CCn3c2nn(CC(N)=O)c(=O)c3=O)cc1"]
    lik = [2.1,3.2,1]
    out = ext_sca(seqs,lik)
    kl_annealer = KLAnnealer(10)
    kl_weight = kl_annealer(10)
    data_path = "D:\Python\ProjectOne\data\data.txt"
    data = np.array([ 7.,  7.,  7.,  1.,  9.,  7.,  1.,  6., 10.,  2., 12.,  4., 15.,
       12., 13., 12.,  4.,  7.,  4.,  7.,  7.,  4.,  2.,  7.,  1.,  6.,
       10.,  2.,  9.,  4.,  7.,  7., 10.,  7.,  7.,  4., 16., 16., 16.])
    result = one_hot(data,17)
    print(result)
    smiles_list , scas_list = canonicalize_smiles_from_file(data_path)
    print(smiles_list)
    print(scas_list)