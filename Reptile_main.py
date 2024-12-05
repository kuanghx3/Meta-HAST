import numpy as np
import pandas as pd
import torch
import os
import myfunctions as fn
from torch.utils.data import DataLoader
import model
from torch import nn
import scipy.sparse as sp
from args import args, dev

adj = pd.read_csv(args.Atten_path+"gat_matrix.csv", header=0, index_col=0)
adj_dense = np.array(adj, dtype=float)
adj_dense = torch.Tensor(adj_dense)
adj = adj_dense.to_sparse_coo().to(dev)

fn.seed_torch(2023)
nr_meta, lr_meta, hr_meta, nr_finetuning, lr_finetuning, hr_finetuning, nr_test, lr_test, hr_test = fn.get_data(args)  # (nodes, days, 288)
nr_meta_t, lr_meta_t, hr_meta_t, nr_finetuning_t, lr_finetuning_t, hr_finetuning_t, nr_test_t, lr_test_t, hr_test_t = fn.get_temp_data(args)  # (days, 288)
nr_meta_dataset = fn.MyDataset(args, nr_meta, nr_meta_t, dev)  # occ:(batch, LOOK_BACK, node) temp:(batch, LOOK_BACK)
lr_meta_dataset = fn.MyDataset(args, lr_meta, lr_meta_t, dev)
hr_meta_dataset = fn.MyDataset(args, hr_meta, hr_meta_t, dev)
nr_finetuning_dataset = fn.MyDataset(args, nr_finetuning, nr_finetuning_t, dev)
lr_finetuning_dataset = fn.MyDataset(args, lr_finetuning, lr_finetuning_t, dev)
hr_finetuning_dataset = fn.MyDataset(args, hr_finetuning, hr_finetuning_t, dev)
nr_test_dataset = fn.MyDataset(args, nr_test, nr_test_t, dev)
lr_test_dataset = fn.MyDataset(args, lr_test, lr_test_t, dev)
hr_test_dataset = fn.MyDataset(args, hr_test, hr_test_t, dev)

nr_meta_loader = DataLoader(nr_meta_dataset, batch_size=len(nr_meta_dataset), shuffle=True, drop_last=True)
lr_meta_loader = DataLoader(lr_meta_dataset, batch_size=len(lr_meta_dataset), shuffle=True, drop_last=True)
hr_meta_loader = DataLoader(hr_meta_dataset, batch_size=len(hr_meta_dataset), shuffle=True, drop_last=True)
nr_finetuning_loader = DataLoader(nr_finetuning_dataset, batch_size=len(nr_finetuning_dataset), shuffle=True, drop_last=True)
lr_finetuning_loader = DataLoader(lr_finetuning_dataset, batch_size=len(lr_finetuning_dataset), shuffle=True, drop_last=True)
hr_finetuning_loader = DataLoader(hr_finetuning_dataset, batch_size=len(hr_finetuning_dataset), shuffle=True, drop_last=True)
nr_test_loader = DataLoader(nr_test_dataset, batch_size=len(nr_test_dataset), shuffle=False, drop_last=True)
lr_test_loader = DataLoader(lr_test_dataset, batch_size=len(lr_test_dataset), shuffle=False, drop_last=True)
hr_test_loader = DataLoader(hr_test_dataset, batch_size=len(hr_test_dataset), shuffle=False, drop_last=True)

model_proposed = model.proposed_Model(args, adj).to(dev)
model.reptile_training(model_proposed, nr_meta_loader, lr_meta_loader, hr_meta_loader, args)
model.fine_tuning(model_proposed, nr_finetuning_loader, lr_finetuning_loader, hr_finetuning_loader, args, "Reptile")
# torch.save(model_proposed, "./model_"+str(args.predict_time)+"/Reptile.pt")
nr_metrics, lr_metrics, hr_metrics, all_metrics = model.test(model_proposed, nr_test_loader, lr_test_loader, hr_test_loader, args, "Reptile")
print(all_metrics)