import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

import typing as ty
import math
from tools.util import CONTINUOUS_TOKEN, CATEGORY_TOKEN, TOTAL_CATEGORY, BINARY_TOKEN, N_TOKENS, THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS, D_TOKEN

def create_causal_mask(T: int, device: torch.device):
    return torch.triu(torch.full([T, T], float("-inf"), device=device), diagonal=1)


class Tokenizer(nn.Module):
    def __init__(self, input_dim, d_hidden, d_depth):
        super(Tokenizer, self).__init__()
        self.num_linear = nn.Linear(input_dim, d_hidden)
        self.depth_embedding = nn.Embedding(N_TOKENS, d_depth)
        self.final_linear = nn.Linear(d_hidden + d_depth, d_hidden)

    def forward(self, x_num, x_depth):
        # x_num: [B, T, CONTINUOUS_TOKEN+TOTAL_CATEGORY+BINARY_TOKEN]
        x_num = self.num_linear(x_num)
        x_depth = self.depth_embedding(x_depth.long()).squeeze(-2)
        x = torch.cat((x_num, x_depth), dim=-1)
        x = self.final_linear(x)
        return x
    
    
class Reconstructor(nn.Module):
    def __init__(self, hid_dim):
        super(Reconstructor, self).__init__()
        
        self.continuous_recons = nn.Linear(hid_dim, CONTINUOUS_TOKEN)
        self.theta_recons = nn.Linear(hid_dim, THETA_CLASS)
        self.phi_recons = nn.Linear(hid_dim, PHI_CLASS)
        self.jointx_range_recons = nn.Linear(hid_dim, JOINTX_CLASS)
        self.jointy_range_recons = nn.Linear(hid_dim, JOINTY_CLASS)
        self.binary_recons = nn.Linear(hid_dim, BINARY_TOKEN)

        self.depth_recons = nn.Linear(hid_dim, N_TOKENS)
        self.relu = nn.ReLU()

    def forward(self, h):
        recon_x_continous = self.continuous_recons(h)
        recon_x_category = torch.cat((self.theta_recons(h), self.phi_recons(h), self.jointx_range_recons(h), self.jointy_range_recons(h)), dim=-1)
        recon_x_binary = self.binary_recons(h)
        recon_x_num = torch.sigmoid(recon_x_continous)
        recon_x_cat = torch.cat([recon_x_category, recon_x_binary], dim=-1)
        recon_x_depth = self.depth_recons(h)

        return recon_x_num, recon_x_cat, recon_x_depth



class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization = 'kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression = None, value_compression = None, mask=None):
  
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        
        # Apply the mask to the attention scores before softmax
        if mask is not None:
            a += mask
            # a = a.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(a/b , dim=-1)

        
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x
        
class Transformer(nn.Module):

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout = 0.0,
        ffn_dropout = 0.0,
        residual_dropout = 0.0,
        activation = 'relu',
        prenormalization = True,
        initialization = 'kaiming',
        pos_embedding = 'learnt', # or abs   
        causal_masking = False,   
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
   
            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        # self.activation = lib.get_activation_fn(activation)
        # self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)
        self.causal_masking = causal_masking
        


    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        # Create a causal mask of shape (seq_len, seq_len) with ones on and below the diagonal and negative infinity elsewhere
        if self.causal_masking:
            causal_mask = create_causal_mask(T=x.size(1), device=x.device)
        else:
            causal_mask = None
        
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
                mask=causal_mask,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x



class VAE(nn.Module):
    def __init__(self, num_layers, n_tokens, d_token, d_depth=32, hid_dim=128, n_head = 1, factor = 4, attention_dropout=0., ffn_dropout=0., pos_embedding = 'learnt'):
        super(VAE, self).__init__()
 
        self.n_tokens = n_tokens
        self.hid_dim = hid_dim
        self.n_head = n_head
 
        # self.Tokenizer = nn.Linear(d_token + d_depth - 1, hid_dim)
        self.Tokenizer = Tokenizer(input_dim=CONTINUOUS_TOKEN+TOTAL_CATEGORY+BINARY_TOKEN, d_hidden=hid_dim, d_depth=d_depth)
        

        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor, attention_dropout=attention_dropout, ffn_dropout=ffn_dropout)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor,  attention_dropout=attention_dropout, ffn_dropout=ffn_dropout)

        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor,  attention_dropout=attention_dropout, ffn_dropout=ffn_dropout, causal_masking=True)
        
        if pos_embedding == "learnt":
            self.pos_embedding = PositionalEncoding(hid_dim, n_tokens)
        elif pos_embedding == "abs":
            self.pos_embedding = PositionalEncoding1D(hid_dim, n_tokens)
            
        # self.depth_embedding = DepthEncoding(d_model=d_depth, max_depth=10)

    def get_embedding(self, x):
        return self.encoder_mu(x, x).detach() 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num, x_depth):

        # x: (batch size, seq len, hid_dim)
        x = self.Tokenizer(x_num, x_depth)
        
        # Positional Encoding
        # x: (seq len, batch size, d_token)
        x = x.permute(1, 0, 2)
        x = self.pos_embedding(x)
        x = x.permute(1, 0, 2)

        mu_z = self.encoder_mu(x)
        std_z = self.encoder_logvar(x)

        z = self.reparameterize(mu_z, std_z)

        
        h = self.decoder(z) #(z[:,1:])
        
        return h, mu_z, std_z




class Model_VAE(nn.Module):
    def __init__(self, num_layers, n_tokens, d_token, d_depth=32, hid_dim=128, n_head = 1, factor = 4, attention_dropout=0., ffn_dropout=0.):
        super(Model_VAE, self).__init__()

        self.VAE = VAE(num_layers, n_tokens, d_token, d_depth, hid_dim, n_head = n_head, factor = factor, attention_dropout = attention_dropout, ffn_dropout = ffn_dropout)
        # self.Reconstructor = nn.Linear(hid_dim, d_token + d_depth - 1)
        self.Reconstructor = Reconstructor(hid_dim=hid_dim)

    def get_embedding(self, x_num, x_depth):
        x = self.Tokenizer(x_num, x_depth)
        return self.VAE.get_embedding(x)

    def forward(self, x_num, x_depth):

        h, mu_z, std_z = self.VAE(x_num, x_depth)

        # subtract the positional embeddings
        # pos_embeds = self.VAE.pos_embedding.pe.detach().permute(1, 0, 2).expand(h.size(0), -1, -1)
        # h = h - pos_embeds  
        recon_x_num, recon_x_cat, recon_x_depth = self.Reconstructor(h)

        return recon_x_num, recon_x_cat, recon_x_depth, mu_z, std_z


class Encoder_model(nn.Module):
    def __init__(self, num_layers, d_depth, hid_dim, n_head, factor):
        super(Encoder_model, self).__init__()
        self.Tokenizer = Tokenizer(input_dim=CONTINUOUS_TOKEN+TOTAL_CATEGORY+BINARY_TOKEN, d_hidden=hid_dim, d_depth=d_depth)
        # self.Tokenizer = nn.Linear(d_token + d_depth - 1, hid_dim)
        self.VAE_Encoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def load_weights(self, Pretrained_VAE):
        self.Tokenizer.load_state_dict(Pretrained_VAE.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(Pretrained_VAE.VAE.encoder_mu.state_dict())

    def forward(self, x_num, x_depth):
        x = self.Tokenizer(x_num, x_depth)
        z = self.VAE_Encoder(x)

        return z

class Decoder_model(nn.Module):
    def __init__(self, num_layers, d_depth, hid_dim, n_head, factor):
        super(Decoder_model, self).__init__()
        self.VAE_Decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor, causal_masking=True)
        # self.Detokenizer = nn.Linear(hid_dim, d_token + d_depth - 1)
        self.Reconstructor = Reconstructor(hid_dim=hid_dim)
        
    def load_weights(self, Pretrained_VAE):
        self.VAE_Decoder.load_state_dict(Pretrained_VAE.VAE.decoder.state_dict())
        self.Reconstructor.load_state_dict(Pretrained_VAE.Reconstructor.state_dict())

    def forward(self, z):

        h = self.VAE_Decoder(z)
        recon_x_num, recon_x_cat, recon_x_depth = self.Reconstructor(h)

        return recon_x_num, recon_x_cat, recon_x_depth
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)
    
    
class DepthEncoding(nn.Module):

    def __init__(self, d_model, max_depth=10, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        depth = torch.arange(max_depth).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_depth, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(depth * div_term)
        pe[:, 0, 1::2] = torch.cos(depth * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # depth   = x[:,:,-1].flatten().long() # get the depth
        depth = torch.zeros(2020).long().to(x.device)
        de = F.embedding(depth, self.pe.squeeze(1)).reshape(x.size(0), x.size(1), -1)
        x = torch.cat((x[:,:,:-1], de), dim=-1)
        return self.dropout(x)