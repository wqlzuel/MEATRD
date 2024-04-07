import os
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, List
from .model import Generator, SVDDEncoder
from .model import SSIMLoss, SVDDLoss, SCELoss
from .utils import seed_everything
from thop import profile

class SNNet:
    def __init__(self, epochs: List[int] = [10, 5], batch_size: int = 128,
                 learning_rate: float = 1e-4, GPU: Optional[str] = "cuda:0",
                 random_state: Optional[int] = None, Mobile: bool = False):
        if GPU is not None:
            if torch.cuda.is_available():
                self.device = torch.device(GPU)
            else:
                print("GPU isn't available, and use CPU to train STAND.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.Mobile = Mobile
        self.rate = 0.01

        if random_state is not None:
            seed_everything(random_state)

    def fit(self, ref_g: dgl.DGLGraph, weight_dir: Optional[str] = None, **kwargs):
        '''Training on reference graph'''
        tqdm.write('Begin to train the model on normal spots...')

        # dataset provides subgraph for training
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
        self.dataloader = dgl.dataloading.DataLoader(
            ref_g, ref_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=True,
            drop_last=False, num_workers=4, device=self.device)

        self.in_dim = ref_g.ndata['gene'].shape[1]
        self.patch_size = ref_g.ndata['patch'].shape[2]

        self.G = Generator(self.patch_size, self.in_dim, Mobile=self.Mobile, **kwargs).to(self.device)
        self.E = SVDDEncoder(self.patch_size, self.in_dim, **kwargs).to(self.device)

        # Load pretrained weights
        self.prepare(weight_dir)
        
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.sch_G = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_G, T_max=self.epochs[0])
        self.opt_E = optim.Adam(self.E.parameters(), lr=self.rate*self.lr, betas=(0.5, 0.999))
        self.sch_E = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_E, T_max=self.epochs[1])
    
        self.l1_loss = nn.L1Loss().to(self.device)
        self.ssim_loss = SSIMLoss().to(self.device)
        self.svdd_loss = SVDDLoss().to(self.device)
        self.sce_loss = SCELoss().to(self.device)

        self.G.train()
        self.E.train()
        for e in range(self.epochs[0]):
            loop = tqdm(self.dataloader, total=len(self.dataloader))

            for _, _, blocks in loop:
                # blocks = [b.to(self.device) for b in blocks]
                self.UpdateG(blocks)
                loop.set_description(f'Stage II Epochs [{e}/{self.epochs[0]}]')
                loop.set_postfix(Loss = self.G_loss.item())
            
            # Update learning rate for G
            self.sch_G.step()

        for e in range(self.epochs[1]):
            loop = tqdm(self.dataloader, total=len(self.dataloader))

            for _, _, blocks in loop:
                blocks = [b.to(self.device) for b in blocks]
                self.UpdateE(blocks)
                loop.set_description(f'Stage III Epochs [{e}/{self.epochs[1]}]')
                loop.set_postfix(Loss = self.E_loss.item())
        
            # Update learning rate for G and D
            self.sch_E.step()
    
        tqdm.write('Training has been finished.')

    @torch.no_grad()
    def predict(self, tgt_g: dgl.DGLGraph):
        '''Detect anomalous spots on target graph'''
        if (self.G is None):
            raise RuntimeError('Please fine-tune the model first.')

        dataloader = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=4, device=self.device)

        self.G.eval()
        self.E.eval()
        tqdm.write('Detect anomalous spots on test dataset...')
        
        # calucate anomaly score
        dis = []
        for _, _, blocks in dataloader:
            # blocks = [b.to(self.device) for b in blocks]
            input_g, input_p = self.read_input(blocks)
            real_g, real_p, fake_g, fake_p = self.G(blocks, input_g, input_p)
            real_fused = self.E(real_g.detach(), real_p.detach())
            fake_fused = self.E(fake_g.detach(), fake_p.detach())
            d = self.svdd_loss(real_fused, fake_fused)
            dis.append(d.cpu().detach())

        # Normalize anomaly scores
        dis = torch.mean(torch.cat(dis, dim=0), dim=1).numpy()
        score = (dis.max() - dis)/(dis.max() - dis.min())

        tqdm.write('Anomalous spots have been detected.\n')
        return list(score)

    @torch.no_grad()
    def prepare(self, weight_dir: Optional[str]):
        '''Prepare stage for pretrained weights'''
        weight_name = '/MobileUNet.pth' if self.Mobile else '/UNet.pth'
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + weight_name)

        # Load the pre-trained weights for Encoder and Decoder
        model_dict = self.G.UNet.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items()}
        model_dict.update(pretrained_dict)
        self.G.UNet.load_state_dict(model_dict)

        for param in self.G.UNet.encoder.parameters():
            param.requires_grad = False
    
    def read_input(self, blocks):
        input_g = blocks[0].srcdata['gene']
        input_p = blocks[0].srcdata['patch']
        return input_g, input_p

    def UpdateG(self, blocks):
        '''Updating generator'''
        input_g, input_p = self.read_input(blocks)
        real_g, real_p, fake_g, fake_p = self.G(blocks, input_g, input_p)
        # flops, params = profile(self.G, inputs=(blocks, input_g, input_p, ))
        # print(params)

        dg = self.in_dim
        dp = 3 * self.patch_size**2
        alpha = dp / (dg + dp)
        loss_g = self.sce_loss(real_g, fake_g)
        loss_p = self.l1_loss(real_p, fake_p) + self.ssim_loss(real_p, fake_p)

        self.G_loss = (1-alpha)*loss_g + alpha*loss_p
        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()
    
    def UpdateE(self, blocks):
        '''Updating SVDD Encoder'''
        input_g, input_p = self.read_input(blocks)
        real_g, real_p, fake_g, fake_p = self.G(blocks, input_g, input_p)
        real_fused = self.E(real_g.detach(), real_p.detach())
        fake_fused = self.E(fake_g.detach(), fake_p.detach())
    
        self.E_loss = self.svdd_loss(real_fused, fake_fused).mean()
        self.opt_E.zero_grad()
        self.E_loss.backward()
        self.opt_E.step()
