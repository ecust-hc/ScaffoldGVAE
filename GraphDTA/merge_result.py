import pandas as pd


def main():
    smiles_data = pd.read_csv("D:\Python\ScaffoldVAE\data\CDK2\our\our_CDK2_sample.csv")
    smiles_data_valid = pd.read_csv("D:\Python\ScaffoldVAE\GraphDTA\\result\our_CDK2_valid\our_CDK2_valid.csv")
    method = "GraphDTA"
    if method == "LeDock":
        for index, row in smiles_data.iterrows():
            if row['SMILES'] in smiles_data_valid['SMILES'].tolist():
                smiles_data.loc[index, 'score'] = \
                    smiles_data_valid.loc[smiles_data_valid['SMILES'] == row['SMILES'], 'score'].values[0]
            else:
                smiles_data.loc[index, 'score'] = 0
        smiles_data['score'] = smiles_data['score'].fillna(0)
    else:
        smiles_data_valid.rename(columns={'compound_iso_smiles': 'SMILES'}, inplace=True)
        for index, row in smiles_data.iterrows():
            if row['SMILES'] in smiles_data_valid['SMILES'].tolist():
                smiles_data.loc[index, 'GraphDTA'] = \
                    smiles_data_valid.loc[smiles_data_valid['SMILES'] == row['SMILES'], 'GraphDTA'].values[0]
            else:
                smiles_data.loc[index, 'GraphDTA'] = 0
        smiles_data['GraphDTA'] = smiles_data['GraphDTA'].fillna(0)
    smiles_data.to_csv("D:\Python\ScaffoldVAE\data\CDK2\our\our_CDK2_GraphDTA.csv", index=False, sep=',')


if __name__ == "__main__":
    main()
