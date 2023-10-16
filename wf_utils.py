#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from os import listdir
import torch
import random
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.optim import Adam
import os
from sklearn.preprocessing import StandardScaler
import math
import json
from matplotlib import pyplot as plt
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import jaccard_score
import cv2
from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
import warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore')
from PIL import Image
# Where to save the figures
PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "Figure_PDFs"
import datetime
today = datetime.date.today()
import time
import torchvision.models.segmentation
time = datetime.datetime.now()
from sklearn.model_selection import KFold
from scipy.special import softmax
from torch.utils.data import Dataset, ConcatDataset
import pickle as pkl
seed= np.random.randint(0,10000)
torch.manual_seed(seed)
from sklearn.model_selection import train_test_split
import torchmetrics
import pickle
import glob
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn
import torchvision
resnet = torchvision.models.resnet.resnet50(pretrained=True)
#U-Net from scratch
import torchvision.models as models
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train
from scipy import stats
from sklearn import metrics
import torchvision.models as models
import torch.nn.functional as F
from scipy import stats
from sklearn import metrics
import torchvision.models as models
import torch.nn.functional as F

if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory didn''t exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.') 

model_root_dir = '.'
model_save_dir = 'saved_models'

if not (os.path.isdir(model_root_dir + '/' + model_save_dir)):
    print('Model saving directory didn''t exist, creating now.')
    os.mkdir(model_root_dir+'/'+model_save_dir)
    
else:
    print('Model saving directory exists.')
    

def train_test_loss(loss_train, loss_test, epochs, save = False, fig_name=''):
    fig    = plt.plot(figsize=(16, 16))
    ax     = plt.gca()
    epoch = range(epochs)
    ax.plot(epoch, loss_train, color='b', linewidth=0.5, label='Train loss')
    ax.plot(epoch, loss_test, color='r', linewidth=0.5, label='Test loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss of predicting ground-level $PM_{2.5}$')
    ax.legend()
    plt.show()
    
    if save:
        savepdf(fig, fig_name)       

def plot_(outputs):
    predictions = []
    ground_truth = []
    for i in range(len(outputs)):
        g = outputs[i][1].detach().cpu().numpy().flatten().tolist()
        p = outputs[i][0].detach().cpu().numpy().flatten().tolist()
        predictions.extend(p)
        ground_truth.extend(g)
        
    return predictions, ground_truth

def eval_stat(predictions, ground_truth):
    Rsquared, pvalue = stats.spearmanr(predictions, ground_truth)
    Rsquared_pearson, pvalue_pearson = stats.pearsonr(predictions, ground_truth)
    return Rsquared, pvalue, Rsquared_pearson, pvalue_pearson

def plot_result(predictions, ground_truth, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, lower_bound=0, upper_bound=175,
                save = True, plot_title = '', fig_name = ''):
    plt.rcParams.update({'mathtext.default':  'regular' })
    
    fig, ax = plt.subplots(figsize = (7,7))
    ax.scatter(ground_truth, predictions,color = 'green', alpha=0.5, edgecolors=(0, 0, 0),  s = 100)
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], 'k--', lw=1)
    ax.set_xlabel('True $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 17)
    ax.set_ylabel('Predicted $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 17)
    ax.set_title(str(plot_title))
    
    ax.tick_params(labelsize = 10)
    fig.text(0.15, 0.85, 'Spearman r = '+ str(np.round(Rsquared,2))) 
    plt.axis('tight')
    fig.text(0.15, 0.81, 'Pearson r = '+ str(np.round(Rsquared_pearson,2)))
    fig.text(0.15, 0.77, 'RMSE = '+ str(np.round(np.sqrt(metrics.mean_squared_error(ground_truth, predictions)),2)))#
 
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    if save:
        fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'_'+'.png', transparent=False, facecolor='white', bbox_inches='tight')
    pass
    return