from utils import construct_vocabulary
from data_structs import Vocabulary,MolData
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DMPN
import argparse
import torch
from utils import valid_smiles,side_no_sca_change,canonicalize_smiles_from_file
import pandas as pd
import time
from time import strftime
from time import gmtime
import sys, os, time
from rdkit import Chem
from add_side_new import scaffold_hop
from rdkit.Chem import AllChem
from functools import partial
import numpy as np
# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing and rnn')
parser.add_argument('--datasetPath', default='D:\Python\Project_VAE\data\LRRK2\\ref_LRRK2_new.smi', help='dataset path')
parser.add_argument('--vocPath', default='D:\Python\Project_VAE\data\Voc_chembl_all_che', help='voc path')
parser.add_argument('--modelPath', default='D:\Python\Project_VAE\models\model_no_node_lrrk2.ckpt', help='model path')
parser.add_argument('--save_dir', default='D:\Python\Project_VAE\data\LRRK2\our\sample_our_newe_2.csv', help='save sample path')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',help='Input batch size for training ')
parser.add_argument('--epochs', type=int, default=400, metavar='N',help='Number of epochs to sample')
parser.add_argument('--molecule_num', type=int, default=5000, metavar='N',help='sample number')
parser.add_argument('--d_z', type=int, default=128, metavar='N',help='z  size')

parser.add_argument('--d_hid', type=int, default=256, metavar='N',help='DMPN model hidden size')
parser.add_argument('--hidden-size', type=int, default=200, metavar='N',help='NMPN , EMPN model hidden size')
parser.add_argument('--depth', type=int, default=3, metavar='N',help='NMPN , EMPN model Hidden vector update times')
parser.add_argument('--out', type=int, default=100, metavar='N',help='EMPN model the size of output')
parser.add_argument('--atten_size', type=int, default=128, metavar='N',help='DMPN model the size of graph attention readout')
parser.add_argument('--r', type=int, default=3, metavar='N',help=' r different insights of node importance')
args = parser.parse_args()
print(args)


from multiprocessing import Pool

def mapper(n_jobs):
    """
    pool多进程并行计算
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    """
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

# log recorder
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "./data/"  # folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        log_name_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        log_name = log_name_time + ".txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def scaffold_hop_one(seqs, mol, sca,  voc):
    totalsmiles_one = []
    for i, seq in enumerate(seqs):
        smile = voc.decode(seq)
        sca_mol = Chem.MolFromSmiles(smile)
        if sca_mol is None:
            totalsmiles_one.append(smile)
        else:

            try:
                new = scaffold_hop(mol[0], sca[0], smile)
                totalsmiles_one.append(new)
            except:
                totalsmiles_one.append(smile)
    return totalsmiles_one

def main(args):

    voc = Vocabulary(init_from_file= args.vocPath)

    #sys.stdout = Logger(sys.stdout)
    # define model
    dmpn = DMPN(args.hidden_size, args.depth, args.out, args.atten_size, args.r, args.d_hid, args.d_z, voc)
    #dmpn = DMPN(args.hidden_size, args.depth, args.out, args.atten_size, args.r, args.d_hid, args.d_z, voc, protein_dict, ver=True)
    dmpn = dmpn.cuda()
    if torch.cuda.is_available():
        dmpn.load_state_dict(torch.load(args.modelPath))

    smiles_list = []
    sca_list = []
    with open(args.datasetPath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            smiles, sca = line.split()
            smiles_list.append(smiles)
            sca_list.append(sca)

    smiles_all = []
    start = time.time()
    for j in range(len(smiles_list)):
        totalsmiles = []
        mol = [smiles_list[j]]
        sca = [sca_list[j]]
        all_seq = 0

        for epoch in range(args.epochs):
            seqs = dmpn.sample(args.batch_size,mol,sca)
            #seqs,_ = dmpn.sample_ver(args.batch_size)
            seq_numpy = seqs.cpu().numpy()
            splitted_tensors = np.array_split(seq_numpy, 2, 0)
            scaffold_hop_one_mol = partial(scaffold_hop_one, mol=mol, sca=sca, voc=voc)

            result = mapper(2)(scaffold_hop_one_mol,splitted_tensors)
            for i in range(len(result)):
                totalsmiles += result[i]
            #totalsmiles = totalsmiles + scaffold_hop_one(mol, sca, seqs, voc)

            molecules_total = len(totalsmiles)

            # print("Epoch {}: {} ({:>4.1f}%) molecules were valid. [{} / {}]".format(epoch + 1, valid,
            #                                                                      100 * valid / len(seqs),
            #                                                                      filter_total, args.molecule_num))
            all_seq += len(seqs)
            if molecules_total > args.molecule_num:
               break
        print("{} {} {} sample end".format(j, smiles_list[j], sca_list[j]))
        smiles_all += totalsmiles[:5000]

    print('Sampling completed')
    end = time.time()
    time_spent = strftime("%H:%M:%S", gmtime(end - start))
    print("sample time spent {time}".format(time=time_spent))
    df_smiles = pd.DataFrame()
    df_smiles['SMILES'] = smiles_all
    df_smiles.to_csv(args.save_dir, index=None)
    # return molecules_total

if __name__ == "__main__":
    main(args)
