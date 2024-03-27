import os
import dgl
import torch
import torch.optim as optim

from tqdm import tqdm
from typing import Optional

from .model import Generator, SSIMLoss
from .utils import seed_everything


def pretrain(graph: dgl.DGLGraph,
             unet_epochs: int = 30,
             batch_size: int = 128,
             learning_rate: float = 1e-4,
             GPU: bool = True,
             random_state: int = None,
             weight_dir: Optional[str] = None,
             Mobile: bool = False
             ):
    if GPU:
        if torch.cuda.is_available():
            device = torch.device("cuda:1")
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if random_state is not None:
        seed_everything(random_state)
    
    # Initialize dataloader for train data
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.DataLoader(
        graph, graph.nodes(), sampler,
        batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=1, device=device)

    G = Generator(graph.ndata['patch'].shape[2], graph.ndata['gene'].shape[1], Mobile=Mobile).to(device)
    opt_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    sch_G = optim.lr_scheduler.CosineAnnealingLR(optimizer = opt_G, T_max = unet_epochs)
    ssim_loss = SSIMLoss().to(device) 
    l1_loss = torch.nn.L1Loss().to(device)

    for e in range(unet_epochs):
        loop = tqdm(dataloader, total=len(dataloader))

        for _, _, graph in loop:
            graph = graph[0]
            real_p = graph.dstdata['patch']
            input_p = graph.srcdata['patch']
            fake_p = G.pretrain_unet(graph, input_p)
            Loss = l1_loss(real_p, fake_p) + ssim_loss(real_p, fake_p)

            opt_G.zero_grad()
            Loss.backward()
            opt_G.step()

            loop.set_description(f'Pretrain UNet [{e}/{unet_epochs}]')
            loop.set_postfix(Loss = Loss.item())
        
        sch_G.step()

    weight_name = '/MobileUNet.pth' if Mobile else '/UNet.pth'
    if weight_dir is None:
        weight_dir = os.path.dirname(__file__) + weight_name
    torch.save(G.UNet.state_dict(), weight_dir)

    tqdm.write(f'UNet weights saved to: {weight_dir}')