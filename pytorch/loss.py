import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import network

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, aux_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

def OSDA(input_list, ad_net, onemore_net, iter_num, onemore_labels, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    feature_onemore = input_list[0]
    source_feature, target_feature = torch.chunk(feature, 2, dim=0)
    source_softmax, target_softmax = torch.chunk(softmax_output, 2, dim=0)

    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1))) # output of discriminator D
        onemore_op = feature_onemore.unsqueeze(1)
        _, onemore_out = onemore_net(onemore_op.view(-1, feature_onemore.size(1))) # output of classifier F
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda() #labels for D
    
    onemore_prob = nn.Softmax(dim=1)(onemore_out)
    _, unknown_prob = torch.split(onemore_prob, onemore_prob.size()[1]-1, dim=1) #probability of belonging to the unknown class
    _, target_onemore_out = torch.chunk(unknown_prob, 2, dim=0) #target probability of belonging to the unknown class
    _, onemore_predict = torch.max(onemore_prob, 1) # predict result of classifier F
    onemore_weight = torch.from_numpy(np.array([[1.0]]*2*batch_size)).float().cuda().detach()

    dc = torch.from_numpy(np.array([[1.0]] * batch_size)).float().cuda()-target_onemore_out #target prob of belonging to known classes
    dc_sum = torch.sum(dc)
    if dc_sum == 0:
        dc_target_weight = torch.from_numpy(np.array([[1.0 / (1.0 * batch_size)]] * batch_size)).float().cuda()
    else:
        dc_target_weight = dc.detach() / dc_sum.detach()
    dc_weight = torch.cat((torch.from_numpy(np.array([[1]] * batch_size)).float().cuda() / batch_size, dc_target_weight.view(-1,1)), dim=0).detach() #weight for discriminator
    for i in range(int(onemore_prob.size()[0]/2)):
        if onemore_predict[i+int(onemore_prob.size()[0]/2)] < (softmax_output.size()[1]) and iter_num > 50:
            onemore_weight[i+int(onemore_prob.size()[0]/2)] = 0
    for i in range(int(onemore_prob.size()[0]/2)):
        onemore_weight[i] = 1.0
    onemore_weight = onemore_weight / torch.sum(onemore_weight).detach().float() #weight for classifier F

    return torch.sum(dc_weight.view(-1,1)*nn.BCEWithLogitsLoss(reduction='none')(ad_out, dc_target)), torch.sum(onemore_weight.view(-1, 1)*nn.CrossEntropyLoss(reduction='none')(onemore_out, onemore_labels).view(-1,1))

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCEWithLogitsLoss()(ad_out, dc_target)
