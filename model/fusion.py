import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATv2Conv


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, nheads, single_layer=True):
        super().__init__()
        self.num_layers = 2
        self.gat_layers = nn.ModuleList()
        self.single_layer = single_layer

        if isinstance(nheads, int):
            nheads = [nheads]
 
        self.gat_layers.append(
            GATv2Conv(in_dim, out_dim, nheads[0], feat_drop=0.2, attn_drop=0.1)
        )

        if self.single_layer:
            self.gat_layers.append(
                nn.Linear(out_dim*nheads[0], out_dim)
            )
        else:
            self.gat_layers.append(
                GATv2Conv(out_dim*nheads[0], out_dim, nheads[1], feat_drop=0.2, attn_drop=0.1)
            )

    def forward(self, blocks, x):
        if self.single_layer:
            x = self.gat_layers[0](blocks, x).flatten(1)
            x = self.gat_layers[1](x)
        else:
            for i in range(self.num_layers):
                g = blocks[i]
                x = self.gat_layers[i](g, x).flatten(1)
        return x


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, attn):
        x = x + self.dropout(attn)
        return self.norm(x)


class FFNBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, d_model),
        )
        self.AddNorm = AddNorm(d_model, dropout)
    
    def forward(self, x):
        return self.AddNorm(x, self.fc(x))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.AddNorm = AddNorm(d_model, dropout)

        self.FFN = FFNBlock(d_model, d_model, dropout)

    def forward(self, x, mask=None):
        # Self attention
        attn = self.self_attn(x, x, x, attn_mask=mask)[0]
        x = self.AddNorm(x, attn)

        # Position-wise feed-forward network
        x = self.FFN(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, out_dim, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm
        self.fc = nn.Linear(encoder_layer.d_model, out_dim)

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)

        for mod in self.layers:
            x = mod(x, mask)
        if self.norm is not None:
            x = self.norm(x)
        
        x = x.squeeze(1)
        return self.fc(x)


class GraphMBT(nn.Module):
    def __init__(self, g_dim, emb_chan, patch_size, fused_dim=16,
                 TF_layers=6, TF_nheads=4, GAT_nheads=[2, 1]):#TF_layer=6
        super().__init__()

        # Patch Projection (emb_chan * patch_size**2 -->> g_dim)
        self.emb_chan = emb_chan
        self.patch_size = patch_size
        p_dim = emb_chan * patch_size**2
        self.p_down = nn.Linear(p_dim, g_dim)
        self.p_up = nn.Linear(g_dim, p_dim)
    
        # Transformer Fusion
        encoder_layer = TransformerLayer(g_dim*2, TF_nheads)
        self.bottleneck = TransformerEncoder(encoder_layer, TF_layers, fused_dim)

        # GAT Fusion
        self.layer = GAT(g_dim+fused_dim, g_dim, nheads=GAT_nheads, single_layers=False)

        # mask
        self.g_mask_token = nn.Parameter(torch.zeros(1, g_dim))
        self.p_mask_token = nn.Parameter(torch.zeros(1, g_dim))

    def fusion(self, blocks, gene_tokens, patch_tokens):
        '''Latent Fusion'''
        # replace input data with mask tokens
        mask_tokens = self.g_mask_token.repeat(blocks[-1].num_dst_nodes(), 1)
        gene_tokens = torch.cat((mask_tokens, gene_tokens[blocks[-1].num_dst_nodes():]), dim=0)
        mask_tokens = self.p_mask_token.repeat(blocks[-1].num_dst_nodes(), 1)
        patch_tokens = torch.cat((mask_tokens, patch_tokens[blocks[-1].num_dst_nodes():]), dim=0)
        concat = torch.cat((gene_tokens, patch_tokens), dim=1)

        # attn mask matrix
        mask = torch.zeros(blocks[0].num_src_nodes(), blocks[0].num_src_nodes()).to(mask_tokens)
        mask[:blocks[-1].num_dst_nodes(), :blocks[-1].num_dst_nodes()] = 1
    
        fused = self.bottleneck(concat)#, mask.bool())

        gene_tokens = self.layer(blocks, torch.cat((gene_tokens, fused), dim=-1))
        patch_tokens = self.layer(blocks, torch.cat((patch_tokens, fused), dim=-1))

        return gene_tokens, patch_tokens

    def forward(self, blocks, g, p):
        p = self.p_down(p.flatten(1))

        # Bottleneck Fusion
        g, p = self.fusion(blocks, g, p)

        p = self.p_up(p)
        return g, p.reshape(-1, self.emb_chan, self.patch_size, self.patch_size)


class ConcatFusion(nn.Module):
    def __init__(self, g_dim, p_dim, z_dim=None):
        super().__init__()
        if z_dim is None:
            z_dim = g_dim
        self.fc_out = nn.Linear(g_dim+p_dim, z_dim)

    def forward(self, g, p):
        output = torch.cat((g, p), dim=1)
        output = self.fc_out(output)
        return output


class MGDAT(nn.Module):
    def __init__(self, g_dim, emb_chan, patch_size, fused_dim=16, blocks=2, 
                 TF_layers=2, TF_nheads=4, GAT_nheads=2, mask=True):
        super().__init__()
        
        # Patch Projection (emb_chan * patch_size**2 -->> g_dim)
        self.emb_chan = emb_chan
        self.patch_size = patch_size
        p_dim = emb_chan * patch_size**2
        self.p_down = nn.Linear(p_dim, g_dim)
        self.p_up = nn.Linear(g_dim, p_dim)
    
        # Transformer Fusion
        encoder_layer = TransformerLayer(g_dim*2, TF_nheads)
        self.bottleneck = TransformerEncoder(encoder_layer, TF_layers, fused_dim)

        # GAT Fusion
        self.GAT = GAT(g_dim+fused_dim, g_dim, nheads=GAT_nheads)

        # mask
        self.mask = mask
        if self.mask:
            self.g_mask_token = nn.Parameter(torch.zeros(1, g_dim))
            self.p_mask_token = nn.Parameter(torch.zeros(1, g_dim))

        self.blocks = blocks

    def make_mask(self, blocks, gene_tokens, patch_tokens):
        # replace input data with mask tokens
        mask_tokens = self.g_mask_token.repeat(blocks[-1].num_dst_nodes(), 1)
        gene_tokens = torch.cat((mask_tokens, gene_tokens[blocks[-1].num_dst_nodes():]), dim=0)
        mask_tokens = self.p_mask_token.repeat(blocks[-1].num_dst_nodes(), 1)
        patch_tokens = torch.cat((mask_tokens, patch_tokens[blocks[-1].num_dst_nodes():]), dim=0)

        return gene_tokens, patch_tokens
    
    def attn_mask(self, blocks, idx):
        mask = torch.zeros(blocks[idx].num_src_nodes(), blocks[idx].num_src_nodes())
        mask[:blocks[-1].num_dst_nodes(), :blocks[-1].num_dst_nodes()] = 1
        return mask.to(self.g_mask_token)

    def fusion(self, blocks, gene_tokens, patch_tokens):
        for i in range(self.blocks):
            concat = torch.cat((gene_tokens, patch_tokens), dim=1)
            
            if self.mask:
                mask = self.attn_mask(blocks, i).bool()
            else:
                mask = None
            
            fused = self.bottleneck(concat, mask)
            gene_tokens = self.GAT(blocks[i], torch.cat((gene_tokens, fused), dim=-1))
            patch_tokens = self.GAT(blocks[i], torch.cat((patch_tokens, fused), dim=-1))
        return gene_tokens, patch_tokens
    
    def forward(self, blocks, g, p):
        p = self.p_down(p.flatten(1))

        if self.mask:
            g, p = self.make_mask(blocks, g, p)

        g, p = self.fusion(blocks, g, p)

        p = self.p_up(p)
        return g, p.reshape(-1, self.emb_chan, self.patch_size, self.patch_size)
    


