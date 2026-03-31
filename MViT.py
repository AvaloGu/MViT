import inspect
import torch
import torch.nn as nn
from dataclasses import dataclass
from einops import rearrange
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

@dataclass
class MViTConfig:
    pooling_func = 'max' # 'max' or 'conv'
    dropout = 0.0 # att dropout, not used currently
    dropout_final_layer = 0.5
    drop_path_rate = 0.2
    n_layer = [1, 2, 11, 2]
    channel_size = [96, 192, 384, 768]
    head_size = 96 # constant throughout the network

    # patch embedding
    patch_embd_ks = (3, 7, 7)
    patch_embd_stride = (2, 4, 4)
    patch_embd_pad = (1, 2, 2)

    # stage resolutions
    resolutions = [(8, 56, 56), (8, 28, 28), (8, 14, 14), (8, 7, 7)]

    # Max pool on K and V tensors
    maxpool_ks = [(2, 9, 9), (2, 5, 5), (2, 3, 3), (2, 2, 2)] # kernel size = stride + 1
    maxpool_stride = [(1, 8, 8), (1, 4, 4), (1, 2, 2), (1, 1, 1)] # adpative pooling, keeps K, V resolution fixed across all stage
    maxpool_pad = (0, 1, 0, 1, 0, 1)
    
    maxpool_downsample_ks = (2, 3, 3)
    maxpool_downsample_stride = (1, 2, 2)

    # Conv pooling
    conv_ks = (3, 3, 3)
    conv_stride = [(1, 8, 8), (1, 4, 4), (1, 2, 2), (1, 1, 1)]
    conv_padding = [(1, 0, 0), (1, 0, 0), (1, 1, 1), (1, 1, 1)]
    
    conv_downsample_stride = (1, 2, 2)
    conv_downsample_pad = (1, 1, 1)


class MHPA(nn.Module):
    def __init__(self, config: MViTConfig, stage, drop_path, downsample = False):
        # downample: space time down sampling given stride (1, 2, 2)
        super().__init__()

        self.resolution = config.resolutions[stage-1] if downsample else config.resolutions[stage]
        channel_size = config.channel_size[stage]
        self.downsample = downsample
        self.pre_att_ln = nn.LayerNorm(channel_size)
        self.attn_proj = nn.Linear(in_features=channel_size, out_features=3*channel_size)
        self.pool_Q = nn.Identity()
        self.pool_X = nn.Identity()

        if config.pooling_func == 'max':
            ks = config.maxpool_ks[stage]
            stride = config.maxpool_stride[stage]
            if stride == (1, 1, 1):
                # we'll skip pooling if there's no dimension to reduce, i.e. resolution is alrady 7x7
                 self.pool_K = nn.Identity()
                 self.pool_V = nn.Identity()
            else:
                self.pool_K = nn.Sequential(nn.ZeroPad3d(config.maxpool_pad), 
                                            nn.MaxPool3d(kernel_size=ks, stride = stride))
                self.pool_V = nn.Sequential(nn.ZeroPad3d(config.maxpool_pad), 
                                            nn.MaxPool3d(kernel_size=ks, stride = stride))
            if downsample:
                self.pool_Q = nn.Sequential(nn.ZeroPad3d(config.maxpool_pad), 
                                            nn.MaxPool3d(kernel_size=config.maxpool_downsample_ks, 
                                                         stride=config.maxpool_downsample_stride))
                self.pool_X = nn.Sequential(nn.ZeroPad3d(config.maxpool_pad), 
                                            nn.MaxPool3d(kernel_size=config.maxpool_downsample_ks, 
                                                         stride=config.maxpool_downsample_stride))
        else:
            assert config.pooling_func == 'conv'
            stride = config.conv_stride[stage]
            padding = config.conv_padding[stage]
            if stride == (1, 1, 1):
                # we'll skip conv pooling if there's no dimension to reduce, i.e. resolution is alrady 7x7
                 self.pool_K = nn.Identity()
                 self.pool_V = nn.Identity()
            else:
                self.pool_K = nn.Conv3d(in_channels=channel_size,
                                        out_channels=channel_size,
                                        kernel_size=config.conv_ks,
                                        stride=stride,
                                        padding=padding,
                                        groups=channel_size) # channel wise conv
                self.pool_V = nn.Conv3d(in_channels=channel_size,
                                        out_channels=channel_size,
                                        kernel_size=config.conv_ks,
                                        stride=stride,
                                        padding=padding,
                                        groups=channel_size) # channel wise conv
            if downsample:
                self.pool_Q = nn.Conv3d(in_channels=channel_size, 
                                        out_channels=channel_size,
                                        kernel_size=config.conv_ks,
                                        stride=config.conv_downsample_stride,
                                        padding=config.conv_downsample_pad,
                                        groups=channel_size) # channel wise conv
                self.pool_X = nn.Conv3d(in_channels=channel_size, 
                                        out_channels=channel_size,
                                        kernel_size=config.conv_ks,
                                        stride=config.conv_downsample_stride,
                                        padding=config.conv_downsample_pad,
                                        groups=channel_size) # channel wise conv
        
        self.hs = config.head_size
        self.ln_K = nn.LayerNorm(self.hs)
        self.ln_V = nn.LayerNorm(self.hs)
        self.ln_Q = nn.Identity()

        if downsample:
            self.ln_Q = nn.LayerNorm(self.hs)

        self.dropout = config.dropout

        self.c_proj = nn.Linear(channel_size, channel_size)
        # self.resid_dropout = nn.Dropout(config.dropout)

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.channel_size = channel_size
        

    def forward(self, x):
        # cls_token: (B, 1, D)
        t, h, w = self.resolution
        L = t * h * w

        x_normalized = self.pre_att_ln(x)
        qkv_cls, qkv = self.attn_proj(x_normalized).split([1, L], dim=1) # (B, 1, 3*D), (B, L, 3*D)

        # process the cls token
        q_cls, k_cls, v_cls = qkv_cls.split(self.channel_size, dim=2) # (B, 1, D)
        q_cls = rearrange(q_cls, 'b l (nh hs) -> b nh l hs', hs=self.hs) # (B, nh, 1, hs)
        k_cls = rearrange(k_cls, 'b l (nh hs) -> b nh l hs', hs=self.hs) # (B, nh, 1, hs)
        v_cls = rearrange(v_cls, 'b l (nh hs) -> b nh l hs', hs=self.hs) # (B, nh, 1, hs)

        # process the data tokens
        q, k, v = qkv.split(self.channel_size, dim=2) # (B, L, D)
        q = rearrange(q, 'b (t h w) d -> b d t h w', t=t, h=h, w=w) # (B, D, T, H, W)
        k = rearrange(k, 'b (t h w) d -> b d t h w', t=t, h=h, w=w) # (B, D, T, H, W)
        v = rearrange(v, 'b (t h w) d -> b d t h w', t=t, h=h, w=w) # (B, D, T, H, W)
        
        b, d, t, h, w = q.size()
        assert (t, h, w) == self.resolution, f"Expected resolution {(t, h, w)} does not match actual resolution {self.resolution}"

        # poolings, reshape, and concatenate
        q = self.pool_Q(q) # (B, D, T, H, W) or (B, D, T, H/2, W/2)
        q = rearrange(q, 'b (nh hs) t h w -> b nh (t h w) hs', hs=self.hs) # (B, nh, L, hs) or (B, nh, L/4, hs)
        q = self.ln_Q(q) # (B, nh, L, hs)
        q = torch.cat((q_cls, q), dim=2) # (B, nh, L+1, hs)

        k = self.pool_K(k) # (B, D, T, H/8, W/8)
        k = rearrange(k, 'b (nh hs) t h w -> b nh (t h w) hs', hs=self.hs) # (B, nh, L/64, hs)
        k = self.ln_K(k) # (B, nh, L/64, hs)
        k = torch.cat((k_cls, k), dim=2) # (B, nh, L/64+1, hs)

        v = self.pool_V(v) # (B, D, T, H/8, W/8)
        v = rearrange(v, 'b (nh hs) t h w -> b nh (t h w) hs', hs=self.hs) # (B, nh, L/64, hs)
        v = self.ln_V(v) # (B, nh, L/64, hs)
        v = torch.cat((v_cls, v), dim=2) # (B, nh, L/64+1, hs)

        # attention!
        y = F.scaled_dot_product_attention(q, k, v,  dropout_p=self.dropout if self.training else 0) # (B, nh, L+1, hs)
        y = rearrange(y, 'b nh l hs -> b l (nh hs)') # (B, L+1, D)
        y = self.drop_path(self.c_proj(y))  # (B, L+1, D) or  # (B, L/4+1, D)

        if self.downsample:
            cls_token, x = x.split([1, L], dim=1) # (B, 1, D), (B, L, D)
            x = rearrange(x, 'b (t h w) d -> b d t h w', t=t, h=h, w=w) # (B, D, T, H, W)
            x = self.pool_X(x) # (B, D, T, H/2, W/2)
            x = rearrange(x, 'b d t h w -> b (t h w) d') # (B, L/4, D)
            x = torch.cat((cls_token, x), dim=1) #(B, L/4+1, D)

        x = x + y

        return x # (B, L+1, D) or  # (B, L/4+1, D)
    

class MLP(nn.Module):
    def __init__(self, config: MViTConfig, stage, drop_path, upsample = False):
        # upsample channel dim by x2
        super().__init__()
        channel_size = config.channel_size[stage]
        self.upsample = upsample
        self.pre_mlp_ln = nn.LayerNorm(channel_size)
        self.c_fc = nn.Linear(channel_size, 4 * channel_size)
        self.gelu = nn.GELU()

        if upsample:
            self.c_proj = nn.Linear(4 * channel_size, 2 * channel_size)
            self.resid_proj = nn.Linear(channel_size, 2 * channel_size)
        else:
            self.c_proj = nn.Linear(4 * channel_size, channel_size)
            self.resid_proj = nn.Identity()

        # self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


    def forward(self, x):
        # x: (B, L+1, D)
        x_normalized = self.pre_mlp_ln(x)

        y = self.c_fc(x_normalized)
        y = self.gelu(y)
        y = self.c_proj(y)
        y = self.drop_path(y)

        res_input = x_normalized if self.upsample else x
        return self.resid_proj(res_input) + y # (B, L+1, D) or (B, L+1, 2D)
    

class Block(nn.Module):
    def __init__(self, config: MViTConfig, stage, dp_rate, downsample = False, upsample = False):
        super().__init__()
        self.mhpa = MHPA(config, stage, dp_rate, downsample)
        self.mlp = MLP(config, stage, dp_rate, upsample)

    def forward(self, x):
        x = self.mhpa(x) # (B, L+1, D) or  # (B, L/4+1, D)
        x = self.mlp(x) # (B, L+1, D) or (B, L+1, 2D)
        return x


class MViT(nn.Module):

    def __init__(self, config: MViTConfig):
        super().__init__()
        self.config = config
        initial_channel_size = config.channel_size[0]
        self.patch_embd = nn.Conv3d(in_channels=3, 
                                    out_channels=initial_channel_size,
                                    kernel_size=config.patch_embd_ks,
                                    stride=config.patch_embd_stride,
                                    padding=config.patch_embd_pad) 
        
        t, h, w = config.resolutions[0]
        self.space_seq_len = h * w + 1 # + 1 for cls token
        self.pos_embd = nn.Embedding(num_embeddings=self.space_seq_len, embedding_dim=initial_channel_size) 

        self.time_dim = t + 1 # + 1 for cls token
        self.time_embd = nn.Embedding(num_embeddings=self.time_dim, embedding_dim=initial_channel_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, initial_channel_size))
        trunc_normal_(self.cls_token, std=0.02)

        # drop path rates
        dp_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.n_layer))]

        self.blocks = nn.ModuleList()
        last_stage = len(config.n_layer) - 1
        dp_rate_idx = 0

        for stage, num_layers in enumerate(config.n_layer):
            for j in range(num_layers):
                # the first block of every stage (except stage 0) is a space-time downsample block
                downsample = (stage > 0 and j == 0)

                # the last block of everg stage (except stage 3, the last stage) is a channel upsample block
                upsample = (stage < last_stage and j == num_layers-1)

                dp_rate = dp_rates[dp_rate_idx]
                dp_rate_idx += 1
                
                self.blocks.append(
                    Block(config, stage, dp_rate, downsample, upsample)
                )

        self.final_dropout = nn.Dropout(config.dropout_final_layer)
        self.ln_final = nn.LayerNorm(config.channel_size[-1])
        self.head = nn.Linear(config.channel_size[-1], 400)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # nn.Embedding don't have bias
            trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm): # this is redundant but just to be explicit
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight) 


    def forward(self, x):
        # x: (B, C, T, H, W)
        device = x.device
        B, C, T, H, W = x.size()

        # patch extraction and embedding, using overlapping cubes
        # x: (B, 3, 16, 224, 224)
        x = self.patch_embd(x) # (B, 96, 8, 56, 56)
        x = rearrange(x, 'b d t h w -> b t (h w) d') # (B, T, S, D)

        # positional space embedding
        pos = torch.arange(1, self.space_seq_len, dtype=torch.long, device=device) 
        pos_embd = self.pos_embd(pos) # (S, D)
        x = x + pos_embd # (B, T, S, D)

        # time embedding
        time = torch.arange(1, self.time_dim, dtype=torch.long, device=device)
        time_embd = self.time_embd(time) # (T, D)
        time_embd = time_embd.unsqueeze(0).unsqueeze(2) # (1, T, 1, D)
        x = x + time_embd # (B, T, S, D)
        x = rearrange(x, 'b t s d -> b (t s) d') # (B, L, D)

        # cls token
        cls_token = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        cls_idx = torch.tensor(0, dtype=torch.long, device=device)
        cls_pos_embd = self.pos_embd(cls_idx) # (D,)
        cls_time_embd = self.time_embd(cls_idx) # (D,)
        cls_token = cls_token + cls_pos_embd + cls_time_embd # (B, 1, D)

        x = torch.cat((cls_token, x), dim=1) # (B, L+1, D)

        for block in self.blocks:
            x = block(x)

        x = self.final_dropout(x)
        # extract cls token embedding
        x = x[:, 0] # (B, D)

        # norm and return logits
        x = self.ln_final(x)
        return self.head(x) # (B, 400)
    
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    def configure_optimizers(self, weight_decay, learning_rate):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, fused=fused_available)
        print(f"using fused AdamW: {fused_available}")

        return optimizer
