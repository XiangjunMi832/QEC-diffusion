import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap

from utils import *


@jit
def CodeSyndrome_jax(hmat, noise_mat):
  syndro_mat = noise_mat@hmat.T%2
  return syndro_mat

def DepolarNoise(num_sample, p_error_dist, key):
  return jrd.choice(key, 2, (num_sample,), p=p_error_dist)
DepolarNoise_vmap = vmap(DepolarNoise, in_axes=(None,0,0))

def CodeSyndromeProj_jax(hmat, noise_mat, proj): 
  return CodeSyndrome_jax(hmat, noise_mat*proj)
CodeSyndromeProj_vmap = vmap(CodeSyndromeProj_jax, in_axes=(None,None,0), out_axes=(1))

def Hmat_Project(hxz, syndro_size):
  syndro_size_total, noise_size = hxz.shape
  syndro_rounds = syndro_size_total//syndro_size
  projs = []; proj = jnp.zeros((1,noise_size),dtype=int)
  for syn_r in range(syndro_rounds):
    hmat_group = hxz[syn_r*syndro_size:syn_r*syndro_size+syndro_size]
    for hmat_row in hmat_group:
      proj = proj.at[0, jnp.argwhere(hmat_row)].set(1)
    projs.append(proj)
  projs = jnp.array(projs)  
  return projs

def SyndroSamples_jax(num_sample, p_errors, hxz, lx, key, projs_nr, nR, syndro_size, num_sp=1000):
  noise_size = len(p_errors)
  keys = jrd.split(key, noise_size)
  p_errors_dist = jnp.stack((1-p_errors, p_errors), axis=1)
  if num_sample>num_sp:
    noise_xz = DepolarNoise_vmap(num_sp, p_errors_dist, keys)
    noise_xz = noise_xz.T 
    syndro_xz = CodeSyndrome_jax(hxz, noise_xz)
    logerr_x = CodeSyndromeProj_vmap(lx, noise_xz, projs_nr)
  else:
    noise_xz = DepolarNoise_vmap(num_sample, p_errors_dist, keys)
    noise_xz = noise_xz.T 
    syndro_xz = CodeSyndrome_jax(hxz, noise_xz)
    logerr_x = CodeSyndromeProj_vmap(lx, noise_xz, projs_nr)
  for  _ in range(num_sample//num_sp-1):
    keys = jrd.split(keys[1], noise_size)
    noise_xz = DepolarNoise_vmap(num_sp, p_errors_dist, keys)
    noise_xz = noise_xz.T 
    syndro_xz_tmp = CodeSyndrome_jax(hxz, noise_xz)
    logerr_x_tmp = CodeSyndromeProj_vmap(lx, noise_xz, projs_nr)
    syndro_xz = jnp.vstack((syndro_xz, syndro_xz_tmp))    
    logerr_x = jnp.vstack((logerr_x, logerr_x_tmp))    
  num_sp_remain = num_sample%num_sp
  if num_sp_remain>0 and num_sample>num_sp:
    keys = jrd.split(keys[1], noise_size); 
    noise_xz = DepolarNoise_vmap(num_sp_remain, p_errors_dist, keys)
    noise_xz = noise_xz.T 
    syndro_xz_tmp = CodeSyndrome_jax(hxz, noise_xz)
    logerr_x_tmp = CodeSyndromeProj_vmap(lx, noise_xz, projs_nr)
    syndro_xz = jnp.vstack((syndro_xz, syndro_xz_tmp))    
    logerr_x = jnp.vstack((logerr_x, logerr_x_tmp))        
  return syndro_xz.reshape(num_sample,nR,syndro_size), logerr_x
