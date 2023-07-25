from utils import construct_vocabulary,create_var,decrease_learning_rate,KLAnnealer,download_fasta_from_uniprot
from data_structs import Vocabulary,MolData
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DMPN
import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from time import strftime
from time import gmtime
import os.path
import math
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing and rnn')
parser.add_argument('--datasetPath', default='./data/PIM1/Smiles_Sca_PIM1.smi', help='dataset path')
parser.add_argument('--save_dir', default='./models/model_PIM1.ckpt', help='save model path')
parser.add_argument('--voc_dir', default='./data/Voc_chembl_all', help='voc path')
parser.add_argument('--loss_dir', default='./data/PIM1/batch_loss_PIM1.png', help='loss path')
parser.add_argument('--pretrain', default='./models/model_pretrain.ckpt', help='voc path')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',help='Input batch size for training ')
parser.add_argument('--hidden-size', type=int, default=200, metavar='N',help='NMPN , EMPN model hidden size')
parser.add_argument('--d_hid', type=int, default=256, metavar='N',help='DMPN model hidden size')
parser.add_argument('--d_z', type=int, default=128, metavar='N',help='z  size')
parser.add_argument('--depth', type=int, default=3, metavar='N',help='NMPN , EMPN model Hidden vector update times')
parser.add_argument('--out', type=int, default=100, metavar='N',help='EMPN model the size of output')
parser.add_argument('--atten_size', type=int, default=128, metavar='N',help='DMPN model the size of graph attention readout')
parser.add_argument('--r', type=int, default=3, metavar='N',help=' r different insights of node importance')
parser.add_argument('--epochs', type=int, default=50, metavar='N',help='Number of epochs to train (default: 50)')
parser.add_argument('--lr_decrease_rate', type=float, default=0.03, metavar='LR',help='Initial learning rate (default: 1e-4)')
parser.add_argument('--learning_rate', type=float, default=1e-5, metavar='LR',help='Initial learning rate (default: 1e-4)')
parser.add_argument('--kl_w_start', type=float, default=0, metavar='kl',help='kl weight')
parser.add_argument('--kl_w_end', type=float, default=0.1, metavar='kl',help='kl weight')
parser.add_argument('--beta', type=float, default=0.1,help='refers to a hyper-parameter of balancing two losses.')
parser.add_argument('--early_stop', type=int, default=10, metavar='N', help='early_stop')
args = parser.parse_args()
print(args)

def main(args):

    if os.path.isfile(args.voc_dir):
        voc = Vocabulary(init_from_file=args.voc_dir)
    else:
        print("Construct vocabulary")
        voc_chars = construct_vocabulary(args.datasetPath, args.voc_dir)
        voc = Vocabulary(init_from_file=args.voc_dir)

    #Create a Dataset from foles
    print("create dataset")
    moldata = MolData(args.datasetPath, voc)

    data = DataLoader(moldata, batch_size=args.batch_size, shuffle=False, drop_last=True,
                      collate_fn=MolData.collate_fn)

    #define model
    dmpn = DMPN(args.hidden_size, args.depth, args.out, args.atten_size, args.r, args.d_hid, args.d_z, voc)
    dmpn = dmpn.cuda()
    if torch.cuda.is_available():
        dmpn.load_state_dict(torch.load(args.pretrain))

    optimizer = torch.optim.Adam(dmpn.parameters(), lr=args.learning_rate)
    start = time.time()
    loss_plt = []
    step_plt = []
    x =0
    kl_annealer = KLAnnealer(args.epochs,args.kl_w_end,args.kl_w_start)
    best_loss, early_stop_count = math.inf, 0
    for epoch in range(0, args.epochs):
        dmpn.train()
        kl_weight = kl_annealer(epoch)
        loss_record = []
        for step, batch in tqdm(enumerate(data), total=len(data)):

            mol_batch, sca_batch, collated_arr= batch
            seq = collated_arr.long()


            kl_loss, recon_loss = dmpn.forward(mol_batch, sca_batch, seq)
            loss =kl_weight*kl_loss + recon_loss

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())
            if epoch % 2 == 0 and step == 1:
                decrease_learning_rate(optimizer, decrease_by= args.lr_decrease_rate)
            if step % 20 == 0 and step != 0:
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f} kl_loss: {:5.2f} recon_loss: {:5.2f} \n".format(epoch, step, loss.item(),kl_loss.item(), recon_loss.item()))
                loss_plt.append(loss.item())
                step_plt.append(x)
                x+=1
        # if (epoch+1) % 10 == 0:
        #     torch.save(dmpn.state_dict(), args.save_dir[:-4]+'_{0:03d}.pt'.format(epoch))
        mean_train_loss = sum(loss_record) / len(loss_record)
        tqdm.write(f'Epoch [{epoch}/{args.epochs}]: Train loss: {mean_train_loss:.4f}')
        if mean_train_loss < best_loss:
            best_loss = mean_train_loss
            torch.save(dmpn.state_dict(), args.save_dir)
            tqdm.write('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= args.early_stop:
            tqdm.write('\nModel is not improving, so we halt the training session.')

        # torch.save(dmpn.state_dict(), args.save_dir[:-4]+'_{0:03d}.pt'.format(epoch))
    end = time.time()
    time_spent = strftime("%H:%M:%S", gmtime(end - start))
    print("train time spent {time}".format(time=time_spent))
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(step_plt, loss_plt)
    plt.savefig(args.loss_dir)
    plt.show()

if __name__ == "__main__":
    main(args)