import numpy as np
import pandas as pd 
import collections
from collections import defaultdict
import copy
import joblib
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sklearn
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.dummy import *
from sklearn.neighbors import *
from sklearn.linear_model import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor

import hyperopt

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper', font_scale=1.4)
plt.style.use('seaborn-whitegrid')

import warnings
warnings.filterwarnings("ignore")

aminoacids = tuple('ACDEFGHIKLMNPQRSTVWY')