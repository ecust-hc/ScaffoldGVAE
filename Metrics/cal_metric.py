from multiprocessing import Pool
from utils import mapper, valid_smiles, read_smiles_csv, get_mol, canonic_smiles, compute_scaffolds, fingerprints, calc_self_tanimoto, side_no_sca_change, mol_passes_filters
import argparse
import pandas as pd
from utils import canonicalize_smiles_from_file
import numpy as np

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

def internal_diversity(gen, n_jobs=1, device='cuda:0', fp_type='morgan', gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
#     sim = calc_agg_tanimoto(gen_fps, gen_fps, agg='mean', device=device, p=p)
    sim = calc_self_tanimoto(gen_fps, agg='mean', device=device, p=p)
    return 1 - np.mean(sim)

def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(valid_smiles, gen)
    if len(gen) == 0:
        return 0
    return gen.count(True) / len(gen)

def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]

def fraction_unique(gen, k =None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        canonic.remove(None)
    return len(canonic) / len(gen)

def scaffold_novelty(gen, train, n_jobs=1):
    # Create the set to store the unique scaffolds
    gen_scaffolds = set(compute_scaffolds(gen, n_jobs=n_jobs))
    #train_scaffolds = set(compute_scaffolds(train, n_jobs=n_jobs))
    train_scaffolds = train
    # Calculate the Scaffold Novelty Score
    if len(gen) != 0:
        scaffold_novelty_score = len(gen_scaffolds - train_scaffolds) / len(gen)
    else:
        scaffold_novelty_score = 0
    return scaffold_novelty_score

def mol_novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    if len(gen_smiles_set) !=0:
        return len(gen_smiles_set - train_set) / len(gen_smiles_set)
    else:
        return 0

def scaffold_diversity(gen, n_jobs=1):
    # Create a set to store the unique scaffolds
    scaffolds = compute_scaffolds(gen, n_jobs=n_jobs)

    # Calculate the Scaffold Diversity Score
    if len(gen) != 0:
        scaffold_diversity_score = len(scaffolds) / len(gen)
    else:
        scaffold_diversity_score = 0

    return scaffold_diversity_score

def sca_sucess(gen,mol,sca):
    filter = []
    for i in range(len(gen)):
        if side_no_sca_change(gen[i],mol,sca):
           filter.append(gen[i])
    if len(gen) != 0:
        return len(filter)/len(gen)
    else:
        return 0

def get_all_metrics(gen, n_jobs=1,
                    device='cpu', batch_size=512, pool=None, train=None,fine_tune=None, ref_mol=None, ref_scas=None, train_scaffolds=None):
    metrics = {}
    # Start the process at the beginning and avoid repeating the process
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    metrics['Validity'] = fraction_valid(gen, n_jobs=pool)
    gen_valid = remove_invalid(gen, canonize=True)
    # if len(gen_valid) == 0:
    #     return {1: 0}
    #合并两个集合
    train = train + fine_tune
    metrics['Uniqueness@1k'] = fraction_unique(gen, k=1000, n_jobs=pool)
    metrics['Uniqueness@5k'] = fraction_unique(gen, k=5000, n_jobs=pool)
    metrics['Diversity_sca'] = scaffold_diversity(gen_valid, n_jobs=pool)

    if train is not None:
        metrics['Novelty_sca'] = scaffold_novelty(gen_valid, train_scaffolds, n_jobs=pool)
        metrics['Novelty'] = mol_novelty(gen_valid, train)

    if len(gen_valid) != 0:
        metrics['Internal_diversity1'] = internal_diversity(gen_valid, n_jobs=pool, p=1)
        metrics['Internal_diversity2'] = internal_diversity(gen_valid, n_jobs=pool, p=2)
        metrics['Filter'] = fraction_passes_filters(gen_valid, n_jobs=pool)
    else:
        metrics['Internal_diversity1'] = 0
        metrics['Internal_diversity2'] = 0
        metrics['Filter'] = 0
    if ref_mol is not None:
        metrics['Sca_sucess'] = sca_sucess(gen_valid,ref_mol,ref_scas)

    if close_pool:
        pool.close()
        pool.join()
    print(metrics)
    return metrics  # , df_distribution

def main(args):

    gen = read_smiles_csv(args.gen_path)
    # gen = []
    # with open(args.gen_path , "r") as f:
    #     for i in range(50000):
    #         gen.append(f.readline().strip())
    #gen = pd.read_csv(args.gen_path)["SMILES"].tolist()
    train = None
    fine_tune = None
    ref_mol = None
    ref_scas = None
    if args.train_path is not None:
        train = pd.read_csv(args.train_path)["SMILES"].tolist()
        # train, _ = canonicalize_smiles_from_file(args.train_path)
        train_scaffolds = set(compute_scaffolds(train, n_jobs=args.n_jobs))

    if args.fine_tune_path is not None:
        # fine_tune, _ = canonicalize_smiles_from_file(args.fine_tune_path)
        with open(args.fine_tune_path, 'r') as f:
            smiles_list = []
            for i, line in enumerate(f):
                smiles = line.split(" ")[0]
                smiles_list.append(smiles)
        fine_tune = smiles_list

    if args.ref_smiles is not None:
        with open(args.ref_smiles, 'r') as f:
            mol = []
            sca = []
            for i, line in enumerate(f):
                mol.append(line.split(" ")[0])
                sca.append(line.split(" ")[1].strip("\n"))
        ref_mol = mol
        ref_scas = sca

    Validity = []
    Uniqueness1k = []
    Uniqueness5k = []
    Diversity_sca = []
    Novelty_sca = []
    Novelty = []
    Internal_diversity1 = []
    Internal_diversity2 = []
    Sca_sucess = []
    Filter = []
    for i in range(20):
        if args.ref_smiles is not None:
            ref_mol_one = ref_mol[i]
            ref_scas_one = ref_scas[i]
        else:
            ref_mol_one = None
            ref_scas_one = None
        gen_5000 = gen[i*5000:(i+1)*5000]
        metrics = get_all_metrics(gen=gen_5000, n_jobs=args.n_jobs,
                              device=args.device, train=train,fine_tune=fine_tune, ref_mol=ref_mol_one, ref_scas =ref_scas_one, train_scaffolds=train_scaffolds)
        for key, value in metrics.items():
            if key == "Validity":
                Validity.append(value)
            elif key == "Uniqueness@1k":
                Uniqueness1k.append(value)
            elif key == "Uniqueness@5k":
                Uniqueness5k.append(value)
            elif key == "Diversity_sca":
                Diversity_sca.append(value)
            elif key == "Novelty_sca":
                Novelty_sca.append(value)
            elif key == "Novelty":
                Novelty.append(value)
            elif key == "Internal_diversity1":
                Internal_diversity1.append(value)
            elif key == "Internal_diversity2":
                Internal_diversity2.append(value)
            elif key == "Sca_sucess":
                Sca_sucess.append(value)
            elif key == "Filter":
                Filter.append(value)
        print("{} metrics is end".format(i))
    metrics_all = {}
    metrics_all["Validity"] = sum(Validity) / len(Validity)
    metrics_all["Uniqueness@1k"] = sum(Uniqueness1k) / len(Uniqueness1k)
    metrics_all["Uniqueness@5k"] = sum(Uniqueness5k) / len(Uniqueness5k)
    metrics_all["Filter"] = sum(Filter) / len(Filter)
    metrics_all["Diversity_sca"] = sum(Diversity_sca) / len(Diversity_sca)
    metrics_all["Novelty_sca"] = sum(Novelty_sca) / len(Novelty_sca)
    metrics_all["Novelty"] = sum(Novelty) / len(Novelty)
    metrics_all["Internal_diversity1"] = sum(Internal_diversity1) / len(Internal_diversity1)
    metrics_all["Internal_diversity2"] = sum(Internal_diversity2) / len(Internal_diversity2)
    metrics_all["Sca_sucess"] = sum(Sca_sucess) / len(Sca_sucess)

    if args.print_metrics:
        for key, value in metrics_all.items():
            print('{}, {:.4}'.format(key, value))
        return metrics_all
    else:
        return metrics_all

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',default='D:\Python\ScaffoldVAE\data\chembl31.csv',type=str, required=False,help='Path to train molecules csv')
    parser.add_argument('--ref_smiles', default="D:\Python\ScaffoldVAE\data\CDK2\Ref_CDK2.smi", type=str,
                        required=False, help='Path to train molecules csv')
    parser.add_argument('--fine_tune_path', default="D:\Python\ScaffoldVAE\data\CDK2\Smiles_Sca_CDK2.smi", type=str, required=False,help='Path to fine molecules csv')
    parser.add_argument('--gen_path',default='D:\Python\ScaffoldVAE\data\CDK2\our\our_CDK2_sample.csv',type=str, required=False,help='Path to generated molecules csv')
    parser.add_argument('--output',default='D:\Python\ScaffoldVAE\data\CDK2\our\our_CDK2_metrics.csv',type=str, required=False,help='Path to save results csv')
    parser.add_argument('--print_metrics', action='store_true', default=True,help="Print results of metrics or not? [Default: False]")
    parser.add_argument('--n_jobs',type=int, default=1,help='Number of processes to run metrics')
    parser.add_argument('--device',type=str, default='cpu',help='GPU device id (`cpu` or `cuda:n`)')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    metrics = main(args)
    table = pd.DataFrame([metrics]).T
    table.to_csv(args.output, header=False)