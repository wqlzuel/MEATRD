import dgl
import torch
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from math import e
from typing import Literal, Optional, List
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from PIL import ImageFile
from stand._utils import seed_everything
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from scipy.sparse import coo_matrix

class Build_multi_graph:
    def __init__(self, adata, image,
                 position, n_neighbors: int = [0,1,2,3,4,5][4],#4,
                 patch_size: int = 48, train_mode: bool = True):
        seed_everything(0)
        self.adata = adata
        self.adata_raw = adata
        self.image = image
        self.position = position
        self.n_dataset = len(adata)
        self.n_neighbors = n_neighbors
        self.patch_size = patch_size
        self.train_mode = train_mode

        self.batch = self.get_batch()
        u, v = self.get_edge()
        self.g = dgl.to_bidirected(dgl.graph((u, v)))
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['batch'] = self.batch
        self.g.ndata['gene'] = self.get_gene()
        self.g.ndata['patch'] = self.get_patch()
        #if self.image is not None:
            #self.g.ndata['patch'] = self.get_patch()

    def get_batch(self):
        adata = []
        for i in range(self.n_dataset):
            a = self.adata[i]
            a.obs['batch'] = i
            adata.append(a)
        self.adata = ad.concat(adata, merge='same')
        self.adata.obs_names_make_unique(join=',')
        batch = np.array(pd.get_dummies(self.adata.obs['batch']), dtype=np.float32)
        return torch.Tensor(batch)

    def get_edge(self):
        self.adata.obs['idx'] = range(self.adata.n_obs)
        u_list, v_list = [], []
        for i in range(self.n_dataset):
            adata = self.adata[self.adata.obs['batch'] == i]
            position = self.position[i]
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1)# 4
            nbrs = nbrs.fit(position)
            _, indices = nbrs.kneighbors(position)
            u = adata.obs['idx'][indices[:, 0].repeat(self.n_neighbors)]
            v = adata.obs['idx'][indices[:, 1:].flatten()]
            u_list = u_list + u.tolist()
            v_list = v_list + v.tolist()
        return u_list, v_list

    def get_patch(self):
        self.patch = self.image[0] 
        for i in range(self.n_dataset-1):
            self.patch = torch.concat([self.patch, self.image[i+1]], dim=0)
        return self.patch
    
    def get_gene(self):
        A = coo_matrix(self.adata.X).tocsr()
        Gene = A.todense()
        return torch.Tensor(Gene)#torch.Tensor(self.adata.X)
    
def preprocess_data(adata: ad.AnnData):
    seed_everything(0)
    adata = adata[:, adata.var_names.notnull()]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=e)
    return adata
def cut_patch(path, train_mode=False):
    patch_size = 32#128#32
    img = np.array(Image.open(path+'.png'))
    adata = sc.read(path+'.h5ad')
    position = adata.obsm['spatial']
    p_list = []
    #if not isinstance(img[0, 0, 0], np.uint8):
    img = np.uint8(img * 255)
    img = Image.fromarray(img)
    r = np.ceil(patch_size/2).astype(int)
    trans = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                #transforms.ToTensor()
            ])
    preprocess = transforms.Compose([
            transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    for i in range(len(position)):
        x, y = position[i, :]
        p = img.crop((x - r, y - r, x + r, y + r))
        #if train_mode:
            #p = trans(p)
        p = preprocess(p)
        p_list.append(p.reshape(3, 2*r, 2*r))
    return torch.stack(p_list)

def read(ref_dir, tgt_dir, ref_name, tgt_name, n_genes=3000, overlap=None, preprocess=False):
    seed_everything(0)
    ref, ref_img, ref_pos = [], [], []
    tgt, tgt_img, tgt_pos = [], [], []
    label = []
    ref_g_list = []
    tgt_g_list = []
    for r in ref_name:
        adata = sc.read(ref_dir + r + '.h5ad')
        position = adata.obsm['spatial']
        image = cut_patch(ref_dir + r, train_mode=False)
        ref.append(adata)
        ref_img.append(image)
        ref_pos.append(position)
    for t in tgt_name:
        adata = sc.read(tgt_dir + t + '.h5ad')
        position = adata.obsm['spatial']
        #image = np.array(Image.open(tgt_dir + r + '.png'))
        image = cut_patch(tgt_dir + t, train_mode=False)
        tgt.append(adata)
        tgt_img.append(image)
        tgt_pos.append(position)
        label.append(adata.obs['disease'].tolist())
    overlap_gene = list(set(ref[0].var_names))
    for i in range(len(ref)-1):    
        overlap_gene=list(set(overlap_gene) & set(ref[i+1].var_names))
    for i in range(len(tgt)):
        overlap_gene=list(set(overlap_gene) & set(tgt[i].var_names))
    ref = [i[:, overlap_gene] for i in ref]
    tgt = [i[:, overlap_gene] for i in tgt]
    ref = [preprocess_data(d) for d in ref]
    tgt = [preprocess_data(d) for d in tgt]
    sc.pp.highly_variable_genes(ref[0], n_top_genes=n_genes, subset=True)
    #ref[0] = ref[0][:, overlap]
    ref = [d[:, ref[0].var_names] for d in ref]
    tgt = [d[:, ref[0].var_names] for d in tgt]
    patch_size = 32#set_patch(ref[0])#!!!
    print('read over')
    ref_g = Build_multi_graph(ref, ref_img, ref_pos, patch_size=patch_size, train_mode=False).g
    for i in range(len(ref)):
        ref_g_list.append(Build_multi_graph([ref[i]], [ref_img[i]], [ref_pos[i]], patch_size=patch_size, train_mode=False).g)#True).g
    print('ref over')
    for i in range(len(tgt)):
        tgt_g_list.append(Build_multi_graph([tgt[i]], [tgt_img[i]], [tgt_pos[i]], patch_size=patch_size, train_mode=False).g)
    print('bulid graph over')
    return ref_g, ref_g_list, tgt_g_list, label, ref[0].var_names

import torch
import numpy as np
import random
import dgl
ref_dir = '/root/Human2/Normal/'#'/root/HumanBreast/Normal/'
tgt_dir = '/root/Human2/Anomaly/'#'/root/HumanBreast/Anomaly/'
ref_name = ['V03','V04','V05','V06','V07','V08','V09','V10']
tgt_name = ['A1', 'B1','C1','D1','E1','F1','G2','H1']
ref_g, ref_g_list, tgt_g_list, label, select_gene = read(ref_dir, tgt_dir, ref_name, tgt_name)#