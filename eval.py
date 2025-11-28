'''
  Performance evaluation (LER and latency) of masked diffusion decoders
  NOTE: This script remains for backward compatibility. The Rectified Flow
  decoder lives in `RecDecoder/` and can be evaluated via `python -m RecDecoder.infer`.
  To run this eval.py: python eval.py arg1 arg2 arg3 arg4. Example: ` python eval.py 72 0 0.001 DF `
  arg1 = 72 for [72,12,6] or 144 for [144,12,12]
  arg2 = {0,1,2,3,4} => {1,2,4,6,12} determine the unmasking time steps or number of unmasking bits at each step
  arg3 = physical error rate
  arg4 = model type DF: train with T=n_l; LR: train with T=1 
  arg5 = number of shots (optional)
  ###seed=9122 for DF (train with T=n_l=k); 91222 for LR (train with T=1)
'''
import pickle
import numpy as np
import math
import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import jit, grad, value_and_grad, vmap
from jax.lax import stop_gradient
import flax
from flax import linen as nn
from typing import Optional
import time

from modules import *
from utils import *
from qldpc import *

from gym import QECGym


N = int(sys.argv[1])
if N != 72 and N != 144:
  print(f"Unsupported blocklength {N}, please choose between 72 and 144.")
  exit(-1)

''' Choose the number of bits to unmask at each time step '''
M_list = jnp.array([1,2,4,6,12]) #0-4
M_id = int(sys.argv[2]) 
M_n = M_list[M_id] # number of bits to unmask at each time step

physical_error_rate = float(sys.argv[3])
if physical_error_rate > 0.1 or physical_error_rate <= 0:
  print(f"Invalid physical error rate {physical_error_rate}")
  exit(-1)

model_type = str(sys.argv[4])

#how many test samples are used in the LER evaluation
N72_shot_dict = {5e-4: 2e7, 7e-4: 1e7, 1e-3: 1e7, 2e-3: 1e5}
N144_shot_dict = {1e-3: 1e7, 2e-3: 1e6, 3e-3: 1e5}
shot_dict = N72_shot_dict if N == 72 else N144_shot_dict
if len(sys.argv) < 6: #argv0 = python eval.py
  # shots not specified
  if physical_error_rate in shot_dict.keys():
    total_sample_test = int(shot_dict[physical_error_rate])
  else:
    total_sample_test = 100000
else:
  total_sample_test = (int(sys.argv[5])//1000) * 1000
  
if N == 72:
  code_name = "N72K12D6"
  batch_size_test = 1000
  gym = QECGym(
    "bbc-72-12-6",
    "X",       # which memory experiment, X or Z
    "circuit", # circuit-level noise
    physical_error_rate=physical_error_rate,
    num_rounds=6,
    measure_both=True,
    load_saved_logical_ops=True
  )
elif N == 144:
  code_name = "N144K12D12"
  batch_size_test = 500
  gym = QECGym(
    "bbc-144-12-12",
    "X",       # which memory experiment, X or Z
    "circuit", # circuit-level noise
    physical_error_rate=physical_error_rate,
    num_rounds=12,
    measure_both=True,
    load_saved_logical_ops=True
  )

hxz = gym.get_spacetime_parity_check_matrix()
prior = gym.get_channel_probabalities()
lx = gym.get_logical_decoding_matrix()
hxz, prior, lx = jnp.array(hxz), jnp.array(prior), jnp.array(lx)
num_logical_obs = lx.shape[0] #number of logical observables (rows in L), e.g., k=12
num_syndromes, num_faults = hxz.shape

'''
% --- Entry definitions ---
Entry \( H_{ij} = 1 \) means fault \( j \) flips syndrome \( i \).
Entry \( L_{ij} = 1 \) means fault \( j \) contributes to flipping logical observable \( i \).
\[
H_{XZ} \in \{0,1\}^{N_S \times N_F}
\]
N_S= total number of syndrome bits over all measurement rounds,
N_F= total number of circuit-level faults.

\[
L_X \in \{0,1\}^{N_L \times N_F}
\]

% --- Interpretations ---
\( H_{XZ} \): tells you how faults affect \textbf{syndromes}. \\
\( L_{X} \): tells you how faults affect \textbf{logical qubits}.

% --- Matrix equations ---
\[
\text{Syndrome:} \quad s = H_{XZ} e \pmod{2}
\]
\[
\text{Logical error:} \quad \ell = L_X e \pmod{2}
\]
'''

''' Model-related settings ''' 
if N == 72:
  d_model = 256; n_heads = 8; 
  encode_layers = 3; decode_layers = 3
  num_syndrome_per_round = 72  
  num_rounds = num_syndromes//num_syndrome_per_round
  assert num_rounds == 7
  projs = Hmat_Project(hxz, num_syndrome_per_round)

elif N == 144:
  d_model = 512; n_heads = 8; 
  encode_layers = 3; decode_layers = 3
  num_syndrome_per_round = 144  
  num_rounds = num_syndromes//num_syndrome_per_round
  assert num_rounds == 13
  projs = Hmat_Project(hxz, num_syndrome_per_round) #projs = [H^(1), H^(2), ..., H^(R)] where H_{XZ}^{(r)} \in \{0,1\}^{n_{\text{# of syn. per round}} \times N_F}.

T_step = num_logical_obs  #k=12=T_step

''' Model initialization '''
key = jrd.key(1234); keys = jrd.split(key, 10)
nR = num_rounds; nr = nR-1
att_mask = Attention_Mask(hxz, lx, num_rounds, num_syndrome_per_round, num_faults)
model = ViT(encode_layers=encode_layers, decode_layers=decode_layers, 
            d_model=d_model, n_heads=n_heads, 
            encode_input=num_syndrome_per_round, encode_output=num_syndrome_per_round,
            decode_input=num_syndrome_per_round+num_logical_obs, decode_output=num_logical_obs,
            nr = nr, nR = nR, att_mask_init=att_mask,
            )
x_example = jrd.randint(keys[1], (4,nR-nr,num_logical_obs), minval=0, maxval=2)#(4, 1, 12)
#So this mimics the model’s expected input:
#a sequence of masked logical bits per decoding step.
#It corresponds to x_tokens during training.
y_example = jrd.randint(keys[2], (4,nR,num_syndrome_per_round), minval=0, maxval=2)#(4, 7, 72)  
#Shape (4, nR, num_syndrome_per_round) → 4 batches × nR rounds × 72 (for 72-qubit code).
#This mimics the syndrome history input to the model.
#It corresponds to syndrome_tokens during training.
params = model.init({'params': keys[1], 'dropout': keys[2]}, x_example, y_example, train=True)['params']
#Model weight tree (initialized)
''' DF: train with T=n_l; LR: train with T=1 '''
params = LoadModel(code_name, model_type)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
print("parameter numbers =", param_count)

''' Functions for evaluation '''
''' Input sample is in batch'''
''' Standard version: unmasking one bit per step'''
def remask(x, indice, n_remask):#Re-mask the least-confident bits during sampling
  is_remask = jnp.zeros_like(indice, dtype=bool)
  is_remask = is_remask.at[0,indice[0,:n_remask]].set(True)
  x = jnp.where(is_remask, 2, x)
  return x, is_remask 
remask_vmap = vmap(remask, in_axes=(0,0,None))

@jit
def sample_prep(params, x, mask_ids, y, t, w=3):
  #It prepares the updated logical vector x_t and selects which bits are candidates for remasking at timestep t.
  x_pred = stop_gradient( model.apply({'params': params}, x, y, train=False, rngs={'dropout': key}) )
  x_t = jnp.where(mask_ids, jnp.where(x_pred>0, 1, 0),  x)
  remask_c = jnp.where(mask_ids, jnp.where(x_pred>0, jax.nn.sigmoid(x_pred), jax.nn.sigmoid(-x_pred)), 1)
  indices = jnp.argsort(remask_c, axis=2)
  n_remask = t-1
  return x_t, indices, n_remask#How many bits to re-mask at this step, e,g. n_remask =2
  
'''
sample_mini returns the remasked variant, along with the updated mask_ids, 
while the raw x_t from sample_prep is just an intermediate used to decide which positions
to keep masked.
'''  
def sample_mini(params, x, mask_ids, y, t):
  x_t, indices, n_remask = sample_prep(params, x, mask_ids, y, t)
  x_t, mask_ids = remask_vmap(x_t, indices, n_remask)
  return x_t, mask_ids

'''
• In the default sample loop we take T_step = num_logical_obs (e.g., 12 logical bits), iterate t downward from T_step to 1, and each iteration keeps n_remask = t - 1
  bits masked. That means:

  - At the very start, all bits are masked.
  - After the first iteration (t = T_step), only T_step - 1 bits remain masked → exactly one bit has been permanently revealed.
  - After the next, T_step - 2 bits remain masked, so another single bit gets fixed, and so on.

  So each pass permanently unmasks exactly one logical bit—the one the model was most confident about at that stage. The variant sample_m changes this by unmasking m
  bits per step (multiple bits), but the basic sample is “single-bit-per-step.”
'''
def sample(params, n, y):
  x = jnp.ones((n,1,num_logical_obs),dtype=int)*2
  mask_ids = jnp.ones((n,1,num_logical_obs),dtype=bool)
  for t in range(T_step,0,-1): 
    x, mask_ids = sample_mini(params, x, mask_ids, y, t)
  return x

''' Variant version: unmasking m bits per step'''
@jit
def sample_prep_m(params, x, mask_ids, y, t, m):
  x_pred = stop_gradient( model.apply({'params': params}, x, y, train=False, rngs={'dropout': key}) )
  x_t = jnp.where(mask_ids, jnp.where(x_pred>0, 1, 0),  x)
  remask_c = jnp.where(mask_ids, jnp.where(x_pred>0, jax.nn.sigmoid(x_pred), jax.nn.sigmoid(-x_pred)), 1)
  indices = jnp.argsort(remask_c, axis=2)
  n_remask = (t-1)*m # this (number of remasking bits) is the difference from the standard
  return x_t, indices, n_remask
  
def sample_mini_m(params, x, mask_ids, y, t, m):
  x_t, indices, n_remask = sample_prep_m(params, x, mask_ids, y, t, m)
  x_t, mask_ids = remask_vmap(x_t, indices, n_remask)
  return x_t, mask_ids

def sample_m(params, n, y, m=2):
  x = jnp.ones((n,1,num_logical_obs),dtype=int)*2
  mask_ids = jnp.ones((n,1,num_logical_obs),dtype=bool)
  for t in range(T_step//m,0,-1): # Different actual T steps for unmasking
    x, mask_ids = sample_mini_m(params, x, mask_ids, y, t, m)
  return x

''' Input sample is single (for evaluating latency decoding time)'''
''' Standard version: unmasking one bit per step'''

'''
sample_mini_single function is the single-sample (non-batched) version of the iterative sampling step (sample_mini).
It’s used in the latency-measurement mode, where you decode one logical vector at a time and time how long it takes.
'''
@jit
def sample_mini_single(params, x, mask_ids, y0, t): #,w=3
  x_pred = stop_gradient( model.apply({'params': params}, x, y0, rngs={'dropout': key}, method=ViT.Get_Logerr_Message) )
  remask_c = jnp.where(mask_ids, jnp.where(x_pred>0, jax.nn.sigmoid(x_pred), jax.nn.sigmoid(-x_pred)), 1)
  indices = jnp.argsort(remask_c, axis=2)
  indice_unmask = indices[0,0,t-1]
  mask_ids = mask_ids.at[0,0,indice_unmask].set(False)
  x_t = x.at[0,0,indice_unmask].set(jnp.where(x_pred[0,0,indice_unmask]>0, 1, 0))
  return x_t, mask_ids

@jit
def prepare_init_x_mask(params, y):
  x = jnp.ones((1,1,num_logical_obs),dtype=int)*2
  mask_ids = jnp.ones((1,1,num_logical_obs),dtype=bool)
  y0 = stop_gradient( model.apply({'params': params}, y, rngs={'dropout': key}, method=ViT.Get_Syndrome_Message) )
  return x, y0, mask_ids 

def sample_single(params, y): # decode only one sample
  x, y0, mask_ids = prepare_init_x_mask(params, y)
  for t in range(T_step,0,-1): 
    x, mask_ids = sample_mini_single(params, x, mask_ids, y0, t)#unmasks one logical bit based on confidence
  return x

''' Variant version: unmasking m={2,4,6,12} bits per step'''
@jit
def sample_mini_single_m2(params, x, mask_ids, y0, t):
  x_pred = stop_gradient( model.apply({'params': params}, x, y0, rngs={'dropout': key}, method=ViT.Get_Logerr_Message) )
  remask_c = jnp.where(mask_ids, jnp.where(x_pred>0, jax.nn.sigmoid(x_pred), jax.nn.sigmoid(-x_pred)), 1)
  indices = jnp.argsort(remask_c, axis=2)
  for m_j in range(1,3):
    indice_unmask = indices[0,0,2*t-m_j]
    mask_ids = mask_ids.at[0,0,indice_unmask].set(False)
    x = x.at[0,0,indice_unmask].set(jnp.where(x_pred[0,0,indice_unmask]>0, 1, 0))
  return x, mask_ids

def sample_single_m2(params, y): # decode only one sample
  x, y0, mask_ids = prepare_init_x_mask(params, y)
  for t in range(T_step//2,0,-1): 
    x, mask_ids = sample_mini_single_m2(params, x, mask_ids, y0, t)
  return x

@jit
def sample_mini_single_m4(params, x, mask_ids, y0, t):
  x_pred = stop_gradient( model.apply({'params': params}, x, y0, rngs={'dropout': key}, method=ViT.Get_Logerr_Message) )
  remask_c = jnp.where(mask_ids, jnp.where(x_pred>0, jax.nn.sigmoid(x_pred), jax.nn.sigmoid(-x_pred)), 1)
  indices = jnp.argsort(remask_c, axis=2)
  for m_j in range(1,5):
    indice_unmask = indices[0,0,4*t-m_j]
    mask_ids = mask_ids.at[0,0,indice_unmask].set(False)
    x = x.at[0,0,indice_unmask].set(jnp.where(x_pred[0,0,indice_unmask]>0, 1, 0))
  return x, mask_ids

def sample_single_m4(params, y): # decode only one sample
  x, y0, mask_ids = prepare_init_x_mask(params, y)
  for t in range(T_step//4,0,-1): 
    x, mask_ids = sample_mini_single_m4(params, x, mask_ids, y0, t)
  return x

@jit
def sample_mini_single_m6(params, x, mask_ids, y0, t):
  x_pred = stop_gradient( model.apply({'params': params}, x, y0, rngs={'dropout': key}, method=ViT.Get_Logerr_Message) )
  remask_c = jnp.where(mask_ids, jnp.where(x_pred>0, jax.nn.sigmoid(x_pred), jax.nn.sigmoid(-x_pred)), 1)
  indices = jnp.argsort(remask_c, axis=2)
  for m_j in range(1,7):
    indice_unmask = indices[0,0,6*t-m_j]
    mask_ids = mask_ids.at[0,0,indice_unmask].set(False)
    x = x.at[0,0,indice_unmask].set(jnp.where(x_pred[0,0,indice_unmask]>0, 1, 0))
  return x, mask_ids

def sample_single_m6(params, y): # decode only one sample
  x, y0, mask_ids = prepare_init_x_mask(params, y)
  for t in range(T_step//6,0,-1): 
    x, mask_ids = sample_mini_single_m6(params, x, mask_ids, y0, t)
  return x

@jit
def sample_mini_single_m12(params, x, y0):
  x_pred = stop_gradient( model.apply({'params': params}, x, y0, rngs={'dropout': key}, method=ViT.Get_Logerr_Message) )
  x = jnp.where(x_pred>0, 1, 0)
  return x

def sample_single_m12(params, y): # decode only one sample
  x, y0, _ = prepare_init_x_mask(params, y)
  x = sample_mini_single_m12(params, x, y0)
  return x

''' Calculate if the predicted logical is the same as the test sample '''
''' Return an array (batch_size,) with bool data: True if different; False if same'''
def IsLogicallyCorrect(logerr, logpred):
  log_diff = jnp.sum(jnp.abs(logerr-logpred), axis=-1)
  return log_diff > 0

''' LER evaluation '''
num_logical_errors = 0; temp_cnt = 0
for _ in range(total_sample_test//batch_size_test): # big batch for generating samples
  ''' generating testing samples '''
  syndro_test, logerr_test, _ = gym.get_decoding_instances(batch_size_test, return_errors=False)
  syndro_test, logerr_test = jnp.array(syndro_test.astype(np.uint8)), jnp.array(logerr_test.astype(np.uint8))
  ''' reshape to be fed to the DF model ''' 
  syndro_xz_in = syndro_test.reshape(batch_size_test, nR, num_syndrome_per_round)
  logerr_xz_in = logerr_test.reshape(batch_size_test, 1, num_logical_obs)     
  logerr_xz_pred = sample_m(params, batch_size_test, syndro_xz_in, M_n)
  wrong_jd = IsLogicallyCorrect(logerr_xz_in, logerr_xz_pred)
  num_logical_errors += jnp.sum( wrong_jd )
  temp_cnt += batch_size_test
ler = num_logical_errors*1.0/temp_cnt        
print(f"LER at p_error={physical_error_rate:.4f}: {num_logical_errors}/{temp_cnt}={ler:.5e}")  
    
#measures decoding latency per sample —
#i.e. how long it takes to decode one syndrome using the iterative unmasking process.    
''' latency decoding time evaluation '''
syndro_xz_bt = syndro_test.reshape(-1, 1, nR, num_syndrome_per_round)
logerr_xz_bt = logerr_test.reshape(-1, 1, 1, num_logical_obs) 

if M_n==1:
  sample_single_m = sample_single
elif M_n==2:
  sample_single_m = sample_single_m2
elif M_n==4:
  sample_single_m = sample_single_m4
elif M_n==6:
  sample_single_m = sample_single_m6
elif M_n==12:
  sample_single_m = sample_single_m12
else:
  print(f"Wrong M: M_n={M_n}"); sys.exit(1) 
  
for tr in range(len(syndro_xz_bt)):
  syndro_xz_in = syndro_xz_bt[tr]; logerr_xz_in = logerr_xz_bt[tr] 
  for _ in range(10):
    start_time = time.perf_counter()
    logerr_xz_pred = sample_single_m(params, syndro_xz_in).block_until_ready()
    end_time = time.perf_counter()
    latency = end_time-start_time
    print(f"Latency decoding time at p_error={physical_error_rate:.4f}: {latency:.5f} seconds")
    
    
    
