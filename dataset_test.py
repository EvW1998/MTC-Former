import pickle
import numpy as np
import torch

dataset = pickle.load(open('data/mosi_data.pkl', 'rb'))

print(dataset['valid'].keys())
print(len(dataset['valid']['vision']))

