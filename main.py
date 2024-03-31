import argparse
import pandas as pd
import scanpy as sc
import torch
import numpy as np
import random
import dgl
from dgl.data.utils import load_graphs
from finetune import SNNet
from pretrain import pretrain
from utils import evaluate

def data_factory(data_path, data_name, is_training=True):
    if is_training:
        label = None
        data = load_graphs(data_path+"ref.bin")[0][0]
    else:
        with open(data_path+data_name+"_label.txt", "r") as tf:
            lines = tf.read().split(" ")
        label=[int(i) for i in lines]
        data = load_graphs(data_path+data_name+"_tgt.bin")[0][0]
    return data, label

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/data/', type=str)
    parser.add_argument('--data_name', default='A', type=str)
    parser.add_argument('--pre_seed', default=0, type=int)
    parser.add_argument('--fine_seed', default=2024, type=int)
    parser.add_argument('--unet_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--pre_lr', default=5e-4, type=float)
    parser.add_argument('--fine_epochs', default=[10,5], type=list)
    parser.add_argument('--fine_lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda:0', type=float)

    args = parser.parse_args()

    data, _ = data_factory(args.data_path, None, True)
    #pretrain
    pretrain(data, unet_epochs=args.unet_epochs,
             batch_size=args.batch_size,
             learning_rate=args.pre_lr, random_state=args.pre_seed, Mobile=True)
    #finetune
    model = SNNet(epochs=args.fine_epochs, batch_size=args.batch_size, learning_rate=args.fine_lr, 
                  GPU=args.device, Mobile=True, random_state=args.fine_seed)
    model.fit(data)

    return model, args

def test(model, args):
    data, label = data_factory(args.data_path, args.data_name, False)
    score = model.predict(data)
    auc, ap, f1 = evaluate(label, score)
    print(f'AUC: {auc:4.1f} AP: {ap:4.1f} F1:{f1:4.1f}')

if __name__ == '__main__':
    model, args = train()
    test(model, args)
