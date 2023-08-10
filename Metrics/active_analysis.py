import pandas as pd
import argparse
from utils import side_no_sca_change,valid_smiles
import statistics

def get_metrics(sample_score, ref_score, mol_one, sca_one):
    metrics = {}
    active = 0
    sucess = 0
    valid = 0
    hop = 0

    #将sample_score中的GraphDTA列进行从高到低排序 并取前10%的的列1
    sample_score = sample_score.sort_values(by='score', ascending=True).head(int(len(sample_score) * 0.3))
    score_all = sample_score['score'].tolist()
    #将score_all 里面为0删除
    score_all = [x for x in score_all if x != 0]
    metrics["score_mean"] = sum(score_all) / len(score_all)
    for index, row in sample_score.iterrows():
        if valid_smiles(str(row['SMILES'])):
            valid += 1
            if row['score'] <= ref_score:
                active += 1
            if side_no_sca_change(row['SMILES'], mol_one, sca_one):
                hop +=1
            if side_no_sca_change(row['SMILES'], mol_one, sca_one) and row['score'] <= ref_score:
                sucess += 1
    metrics["active_rate"] = active / valid
    metrics["sucess_rate"] = sucess / valid
    metrics["hop_rate"] = hop / valid
    return metrics

def main(args):
    ref_score = pd.read_csv(args.ref_score)
    sample_score = pd.read_csv(args.sample_score)
    ref_score.rename(columns={'compound_iso_smiles': 'SMILES'}, inplace=True)
    sample_score.rename(columns={'compound_iso_smiles': 'SMILES'}, inplace=True)
    if args.ref_smiles is not None:
        with open(args.ref_smiles, 'r') as f:
            mol = []
            sca = []
            for i, line in enumerate(f):
                mol.append(line.split(" ")[0])
                sca.append(line.split(" ")[1].strip("\n"))
        ref_mol = mol
        ref_scas = sca
    mean_all = []
    active_all = []
    sucess_all = []
    hop_all = []
    for i in range(20):
        mol_one = ref_mol[i]
        sca_one = ref_scas[i]
        sample_one = sample_score[i * 5000:(i + 1) * 5000]
        ref_score_one = ref_score.iloc[i]['docking']
        metrics = get_metrics(sample_one, ref_score_one, mol_one, sca_one)
        for key, value in metrics.items():
            if key == "score_mean":
                mean_all.append(value)
            elif key == "active_rate":
                active_all.append(value)
            elif key == "sucess_rate":
                sucess_all.append(value)
            elif key == "hop_rate":
                hop_all.append(value)
        print("{} metrics is end".format(i))
        print(metrics)
    metrics_all = {}
    stdev_all = {}
    metrics_all["mean_all"] = sum(mean_all) / len(mean_all)
    stdev_all["mean_all"] = statistics.stdev(mean_all)
    metrics_all["active_all"] = sum(active_all) / len(active_all)
    stdev_all["active_all"] = statistics.stdev(active_all)
    metrics_all["sucess_all"] = sum(sucess_all) / len(sucess_all)
    stdev_all["sucess_all"] = statistics.stdev(sucess_all)
    metrics_all["hop_all"] = sum(hop_all) / len(hop_all)
    stdev_all["hop_all"] = statistics.stdev(hop_all)
    print("means_all")
    for key, value in metrics_all.items():
        print('{}, {:.4}'.format(key, value))
    print("stdev_all")
    for key, value in stdev_all.items():
        print('{}, {:.4}'.format(key, value))
    return metrics_all

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_score',default='D:\Python\ScaffoldVAE\data\CDK2\Ref_CDK2_docking.csv',type=str, required=False,help='Path to train molecules csv')
    parser.add_argument('--sample_score', default="D:\Python\ScaffoldVAE\data\CDK2\our\our_CDK2_Docking.csv", type=str,
                        required=False, help='Path to train molecules csv')
    parser.add_argument('--ref_smiles', default="D:\Python\ScaffoldVAE\data\CDK2\Ref_CDK2.smi", type=str,
                        required=False, help='Path to train molecules csv')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    metrics = main(args)
