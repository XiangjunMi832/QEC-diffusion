import numpy as np
import pickle

def LoadModel(code_name, model_type):
  path = "model/"+code_name+"masked"+model_type+"model.pickle"
  with open(path, 'rb') as file:
    params = pickle.load(file)
  return params



