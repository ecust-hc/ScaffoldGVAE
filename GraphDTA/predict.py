import argparse
import os
import sys
import time
import pandas as pd
import polars as pl
import polars
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import global_data

from create_data_test import TestbedDataset
from models.ginconv import GINConvNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
graph_model_path = "D:\Python\ScaffoldVAE\GraphDTA\model\model_GINConvNet_ChEMBL_total_lrrk.model"
RESULT_DIR = "D:\Python\ScaffoldVAE\GraphDTA\\result"
global_data._init()

def predicting(model, device, loader):
    """
    It takes a model, a device, and a dataloader, and returns the predictions of the model on the data in the dataloader

    :param model: the model to be used for prediction
    :param device: the device to run the model on (CPU or GPU)
    :param loader: the dataloader for the test set
    :return: The predictions of the model.
    """
    model.eval()
    total_preds = torch.Tensor()
    # total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            # total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_preds.numpy().flatten()

def load_model_dict(model, ckpt):
    """
    It loads a model's state dictionary from a checkpoint file

    :param model: the model to load the weights into
    :param ckpt: the path to the checkpoint file
    """
    model.load_state_dict(torch.load(ckpt))

# datasets = [['train401'][int(sys.argv[1])]]
# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
# model_st = modeling.__name__

cuda_name = "cuda:0"
# if len(sys.argv)>3:
#     cuda_name = ["cuda:0","cuda:1","cuda:2","cuda:3"][int(sys.argv[3])]
# print('cuda_name:', cuda_name)

# The hyperparameters of the model.
TEST_BATCH_SIZE = 512

def main(args):
    """
    It loads the model, loads the test data, and then runs the model on the test data
    :param args: the arguments passed in from the command line
    """
    time_start = time.time()
    smiles_data = pd.read_csv(args.smiles_file)
    #如果smiles_data不存在ID列，则添加ID列
    if 'ID' not in smiles_data.columns:
        smiles_data.insert(0, 'ID', range(len(smiles_data)), allow_duplicates=False)
    smiles_data.to_csv(args.smiles_file, index=False)

    data_df = pl.read_csv(args.smiles_file)
    file_name = os.path.splitext(os.path.basename(args.smiles_file))[0]
    print(file_name)
    if args.id_col_name is None:
        file_data = data_df.select([pl.col(args.smiles_col_name).alias('compound_iso_smiles'),])
        file_data = file_data.with_column(pl.Series(name="idnumber", values=[i for i in range(len(file_data))]))
    else:
        file_data = data_df.select([pl.col(args.smiles_col_name).alias('compound_iso_smiles'), pl.col(args.id_col_name).alias('idnumber'), ])

    os.makedirs(f'D:\Python\ScaffoldVAE\GraphDTA\\result\{file_name}\\raw', exist_ok=True)
    file_data.write_csv(f'D:\Python\ScaffoldVAE\GraphDTA\\result\{file_name}\\raw\\test_drop.csv')

    data_path = f"D:\Python\ScaffoldVAE\GraphDTA\\result\{file_name}"
    fpath = data_path
    test_data = TestbedDataset(root=fpath, phase='test')
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    print('dataloader generate success')
    # training the model
    device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")
    model = GINConvNet().to(device)
    load_model_dict(model, graph_model_path)
    pred = predicting(model, device, test_loader)
    origin_data = pd.read_csv(os.path.join(fpath, 'raw', 'test_drop.csv'),)
    assert len(pred) == len(origin_data)
    origin_data['GraphDTA'] = pred
    pic50_mean = sum(pred) / len(pred)
    result_file_name = f'{file_name}.csv'
    result_path = os.path.join(RESULT_DIR, f'{file_name}')
    os.makedirs(result_path, exist_ok=True)

    origin_data.to_csv(os.path.join(result_path, result_file_name), index=False)
    time_end = time.time()
    time_use = (time_end - time_start) / 60
    print('test all time use : ', time_use)
    print("pic50_mean {}".format(pic50_mean))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add argument
    parser.add_argument('--smiles_file', default='D:\Python\ScaffoldVAE\GraphDTA\data\CDK2\our_CDK2_valid.csv', help='dataset ready to test')
    parser.add_argument('--fasta_path',type=str,default= 'D:\Python\ScaffoldVAE\GraphDTA\data\CDK2\P24941.fasta',help='the path of protein fasta')
    parser.add_argument('--sep', type=str, default=',', help='the separator of CSV file')
    parser.add_argument('--smiles_col_name', type=str, default='SMILES', help='the column name of SMILES')
    parser.add_argument('--id_col_name', type=str, default='ID', help='the column name of id')
    args = parser.parse_args()
    global_data.set_fasta_path(args.fasta_path)
    main(args)
