'''Diffusion decoder with chain of thought'''

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import flax
from flax import linen as nn
from typing import Optional

from qldpc import *
from utils import *

dtype = jnp.float32
  
class FMHA(nn.Module):
  d_model: int  # dimension of the embedding space
  n_heads: int  # number of heads
  d_input: int  # lenght of the input sequence/dimension of input/(logerr+syndro) error e.g.,=96 in N72K12D6
  param_dtype = dtype

  def setup(self):
    self.V = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype)
    self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype)
    self.alpha = self.param("alpha", nn.initializers.xavier_uniform(), (self.n_heads, self.d_input, self.d_input), self.param_dtype)
    self.d_eff = self.d_model//self.n_heads

  def __call__(self, x, mask):
    # apply the value matrix in paralell for each head
    x = self.V(x)

    # split the representations of the different heads
    x = jnp.reshape(x, (-1, self.d_input, self.n_heads, self.d_eff))
    # 'batch d_input (d_model) -> batch d_input n_heads d_eff'
  
    # factored attention mechanism
    x = jnp.transpose(x, (0,2,1,3))
    # 'batch d_input n_heads d_eff -> batch n_heads d_input d_eff'
    if mask==None:
      x = jnp.matmul(self.alpha, x)
    else:  
      x = jnp.matmul(self.alpha*mask, x)
      
    x = jnp.transpose(x, (0,2,1,3))
    # 'batch n_heads d_input d_eff -> batch d_input n_heads d_eff'
  
    # concatenate the different heads
    x = jnp.reshape(x, (-1, self.d_input, self.d_model))
    # 'batch d_input n_heads d_eff ->  batch d_input (d_model=n_heads d_eff)'
  
    # the representations of the different heads are combined together
    x = self.W(x)
    return x

class EncoderBlock(nn.Module):
  d_model: int   # dimensionality of the embedding space
  n_heads: int   # number of heads
  d_input: int   # length of the input sequence
  dropout_prob: Optional[float] = 0.1
  param_dtype = dtype

  def setup(self):
    self.attn = FMHA(d_model=self.d_model, n_heads=self.n_heads, d_input=self.d_input)
    self.dropout_a = nn.Dropout(self.dropout_prob)
    
    self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
    self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)
    
    self.ff_a = nn.Dense(2*self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype)
    self.ff_dropout = nn.Dropout(self.dropout_prob)
    self.ff_b = nn.gelu 
    self.ff_c = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=self.param_dtype)
    
    self.dropout_b = nn.Dropout(self.dropout_prob)

  def __call__(self, x, mask, train=True):
    x_tmp = self.attn(self.layer_norm_1(x), mask)
    x = x + self.dropout_a(x_tmp, deterministic=not train)
    x = self.layer_norm_2(x)
    x_tmp = self.ff_a(x)
    x_tmp = self.ff_dropout(x_tmp, deterministic=not train)
    x_tmp = self.ff_b(x_tmp)
    x_tmp = self.ff_c(x_tmp)  
    
    x = x + self.dropout_b(x_tmp, deterministic=not train)
    return x

class Encoder(nn.Module):
  num_layers: int    # number of layers
  d_model: int       # dimensionality of the embedding space
  n_heads: int       # number of heads
  d_input: int       # lenght of the input sequence
  d_output: int       # lenght of the input sequence 
  dropout_prob: Optional[float] = 0.1
  
  def setup(self):
    self.layers = [EncoderBlock(d_model=self.d_model, n_heads=self.n_heads, d_input=self.d_input, dropout_prob=self.dropout_prob) for _ in range(self.num_layers)]

  def __call__(self, x, mask, train=True):
    for l in self.layers:
      x = l(x, mask, train=train)
    return x[:,:self.d_output]

class OuputHead(nn.Module):
  #d_output: int # dimension of output/logic error e.g.,=24 in N72K12D6
  dropout_prob: Optional[float] = 0.1 
  param_dtype = dtype

  def setup(self):
    self.out_layer_norm = nn.LayerNorm(use_scale=True, use_bias=True, param_dtype=self.param_dtype)
    self.output_layer = nn.Dense(1, param_dtype=self.param_dtype, kernel_init=nn.initializers.xavier_uniform(), bias_init=jax.nn.initializers.zeros)
    #self.nonlinear = nn.sigmoid
    self.dropout = nn.Dropout(self.dropout_prob)
    
  def __call__(self, x, train=True):
    #x = x[:,:self.d_output]
    x = self.out_layer_norm(x)
    x = self.dropout(x, deterministic=not train)
    x = self.output_layer(x)
    #x = self.nonlinear(x)
    return jnp.squeeze(x, axis=-1)

class ViT(nn.Module):
  encode_layers: int  # number of encoder layers
  decode_layers: int  # number of decoder layers  
  d_model: int  # dimensionality of the embedding space
  n_heads: int  # number of heads
  encode_input: int  # lenght of the encoder input sequence
  encode_output: int # lenght of the encoder output sequence
  decode_input: int  # lenght of the decoder input sequence 
  decode_output: int # lenght of the decoder output sequence 
  nr: int # when to start predicting 
  nR: int # rounds of syndromes
  att_mask_init: list # attention mask
  dropout_prob: Optional[float] = 0.1
  
  def setup(self, num_class=2):#num_class=2 handles the binary 0/1 outcomes
    self.num_class = num_class
    self.label_embx = nn.Embed(self.num_class+1, self.d_model)
    self.label_emby = nn.Embed(self.num_class, self.d_model)#embeds logical tokens and reserves one extra slot (value 2) for the masked   
    self.encoder = Encoder(num_layers=self.encode_layers, d_model=self.d_model, 
                           n_heads=self.n_heads, d_input=self.encode_input, 
                           d_output=self.encode_output,
                           dropout_prob=self.dropout_prob)
    self.decoder = Encoder(num_layers=self.decode_layers, d_model=self.d_model, 
                           n_heads=self.n_heads, d_input=self.decode_input, 
                           d_output=self.decode_output,
                           dropout_prob=self.dropout_prob)
    self.output = OuputHead(dropout_prob=self.dropout_prob)
    self.att_mask = [self.param(f"attmask{k}", lambda key, shape: self.att_mask_init[k]+0.0, self.att_mask_init[k].shape) for k in range(len(self.att_mask_init))] 
  #forward pass  
  def __call__(self, x, y, train=True):
    y0 = self.label_emby(y[:,0])
    y0 = self.encoder(y0, self.att_mask[0], train=train)
    for nr_sub in range(1,self.nr):
      y0 = y0+self.label_emby(y[:,nr_sub])
      y0 = self.encoder(y0, self.att_mask[nr_sub], train=train)
    xy_rs = jnp.zeros_like(x, dtype=float)
    for nr_sub in range(self.nr, self.nR):    
      y0 = y0+self.label_emby(y[:,nr_sub])
      x_r = x[:,nr_sub-self.nr]
      y0 = self.encoder(y0, self.att_mask[nr_sub], train=train)
      xy_r = jnp.concatenate((self.label_embx(x_r), y0), axis=1)
      xy_r = self.decoder(xy_r, None, train=train)
      xy_r = self.output(xy_r, train=train)
      xy_rs = xy_rs.at[:,nr_sub-self.nr].set(xy_r)
    return xy_rs

  def Get_Syndrome_Message(self, y, train=False):
    y0 = self.label_emby(y[:,0])
    y0 = self.encoder(y0, self.att_mask[0], train=train)
    for nr_sub in range(1,self.nR):
      y0 = y0+self.label_emby(y[:,nr_sub])
      y0 = self.encoder(y0, self.att_mask[nr_sub], train=train)
    return y0
  
  def Get_Logerr_Message(self, x, y0, train=False):
    #x_r = x[:,0]
    xy_r = jnp.concatenate((self.label_embx(x[:,0]), y0), axis=1)
    xy_r = self.decoder(xy_r, None, train=train)
    xy_r = self.output(xy_r, train=train)
    return jnp.expand_dims(xy_r, 1)

def Attention_Mask(hxz, lx, nR, syndro_size, noise_size):
  att_mask = []
  for nr_sub in range(nR):
    hxz_sub = hxz[:(nr_sub+1)*syndro_size].reshape(nr_sub+1, syndro_size, noise_size)
    hxz_sub = jnp.sum(hxz_sub, axis=0)
    hxz_sub = jnp.where(hxz_sub>0, 1.0, 0)
    mask = jnp.pow(hxz_sub@hxz_sub.T, 1/8)
    mask = jnp.expand_dims(mask, axis=0)  
    att_mask.append(mask)
  return att_mask

