import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
RS = 20150101
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import scipy.io as sio
import h5py

pthfile = r'./snapshot/log/clTpr/known_best_model.pth'
net = torch.load(pthfile).cpu()
pthfile1 = r'./snapshot/log/clTpr/unknown_onemore_best_model.pth'
net1 = torch.load(pthfile1).cpu()
net3 = net1[0]
net2 = net1[1]
#for name in net2.state_dict():
#    print(name)
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#net = nn.DataParallel(net, device_ids=[0])

config = {}
config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
prep_dict = {}
prep_config = config["prep"]
prep_dict["source"] = prep.image_train(**config["prep"]['params'])
prep_dict["target"] = prep.image_train(**config["prep"]['params'])

dsets = {}
dset_loaders = {}
dsets["source"] = ImageList(open('../data/office-home/clipart_tsn_8.txt').readlines(), \
                     transform=prep_dict["source"])
dset_loaders["source"] = DataLoader(dsets["source"], batch_size=36, \
       shuffle=True, num_workers=4, drop_last=True)
dsets["target"] = ImageList(open('../data/office-home/product_tsn_9.txt').readlines(), \
                     transform=prep_dict["target"])
dset_loaders["target"] = DataLoader(dsets["target"], batch_size=36, \
       shuffle=True, num_workers=4, drop_last=True)
len_train_source = len(dset_loaders["source"])
len_train_target = len(dset_loaders["target"])
source_data = iter(dset_loaders["source"])
target_data = iter(dset_loaders["target"])
features_source_total = []
features_target_total = []
label_source_total = []
label_target_total = []
for i in range(len_train_source):
    inputs_source, labels_source, path_source = source_data.next()
    #features_source, outputs_source = net(inputs_source)
    features_source, _ = net3(inputs_source)
    _, outputs_source = net2(features_source)
    source_softmax_out = nn.Softmax(dim=1)(outputs_source)
    predict_source_prob, predict_source_labels = torch.max(source_softmax_out, 1)
    features_source_total = torch.cat((torch.tensor(features_source_total).float(),torch.tensor(features_source).float()),dim=0)
    #label_source_total = torch.cat((torch.tensor(label_source_total).float(),torch.tensor(labels_source).float()),dim=0)
    label_source_total = torch.cat((torch.tensor(label_source_total).float(),torch.tensor(predict_source_labels).float()),dim=0)
sio.savemat('cl_clTpr_8.mat', {'fea':features_source_total.data.numpy()})
sio.savemat('cl_clTpr_label_8.mat', {'lab':label_source_total.data.numpy()})

for i in range(len_train_target):
    inputs_target, labels_target, path_target = target_data.next()
    features_target, _ = net3(inputs_target)
    _, outputs_target = net2(features_target)
    target_softmax_out = nn.Softmax(dim=1)(outputs_target)
    predict_target_prob, predict_target_labels = torch.max(target_softmax_out, 1)
    features_target_total = torch.cat((torch.tensor(features_target_total).float(),torch.tensor(features_target).float()),dim=0)
    label_target_total = torch.cat((torch.tensor(label_target_total).float(),torch.tensor(predict_target_labels).float()),dim=0)
sio.savemat('pr_clTpr_9.mat', {'fea':features_target_total.data.numpy()})
sio.savemat('pr_clTpr_label_9.mat', {'lab':label_target_total.data.numpy()})
#target_label = sio.loadmat('cl_label_8.mat')
#print(target_label)
