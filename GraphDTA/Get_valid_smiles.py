import pandas as pd
from rdkit import Chem

def valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:          # check validity
        return False
    try:                     # check valence, aromaticity, conjugation and hybridization
        Chem.SanitizeMol(mol)
    except:
        return False
    return True

def main():
    smiles_data = pd.read_csv("D:\Python\ScaffoldVAE\data\CDK2\our\our_CDK2_sample.csv")
    smiles_data = smiles_data['SMILES'].tolist()
    smiles = []
    # 循环遍历smiles_data中的每一行
    for i in range(len(smiles_data)):
        if valid_smiles(str(smiles_data[i])):
            smiles.append(smiles_data[i])
    # 将smiles写入到csv文件中
    df_smiles = pd.DataFrame()
    df_smiles['SMILES'] = smiles
    df_smiles.to_csv("D:\Python\ScaffoldVAE\GraphDTA\data\CDK2\our_CDK2_valid.csv", index=False)

if __name__ == "__main__":
    main()