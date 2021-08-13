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


def image_classification_test(loader, base_model, onemore_model, test_10crop=True):
    class_num = config["network"]["params"]["class_num"]
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                onemore_outputs = []
                for j in range(10):
                    predict_features, predict_out = base_model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                    _, onemore_out = onemore_model(predict_features)
                    onemore_outputs.append(nn.Softmax(dim=1)(onemore_out))
                outputs = sum(outputs)
                onemore_outputs = sum(onemore_outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    onemore_all_output = onemore_outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                    onemore_all_output = torch.cat((onemore_all_output, onemore_outputs.float().cpu()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    _, onemore_predict = torch.max(onemore_all_output, 1)
    total_sum = torch.from_numpy(np.array([[0]] * (class_num+1))).float().cuda()
    correct_sum = torch.from_numpy(np.array([[0]] * (class_num+1))).float().cuda()
    known_total_sum = torch.from_numpy(np.array([[0]] * class_num)).float().cuda()
    known_correct_sum = torch.from_numpy(np.array([[0]] * class_num)).float().cuda()
    for i in range(onemore_all_output.size()[1]):
    #record how many target are predicted to each category and how many of them are correctly calssified including the unknown class
        total_sum[i] = torch.sum(torch.squeeze(onemore_predict).float() == i).item()
        correct_sum[i] = torch.sum(all_label[np.where(torch.squeeze(onemore_predict).float() == i)] == i).item()
    for i in range(all_output.size()[1]):
    #record how many target are predicted to each known category and how many of them are correctly calssified
        known_total_sum[i] = torch.sum(torch.squeeze(onemore_predict).float() == i).item()
        known_correct_sum[i] = torch.sum(all_label[np.where(torch.squeeze(onemore_predict).float() == i)] == i).item()
    accuracy_known = torch.mean(known_correct_sum[np.where(torch.squeeze(known_total_sum).float().cpu() != 0)] \
        / known_total_sum[np.where(torch.squeeze(known_total_sum).float().cpu() != 0)]) # OS* accuracy
    accuracy_unknown = torch.mean(correct_sum[np.where(torch.squeeze(total_sum).float().cpu() != 0)] / total_sum[np.where(torch.squeeze(total_sum).float().cpu() != 0)]) # OS accuracy
    return accuracy_unknown, accuracy_known


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                            transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                            transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    
    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
        onemore_net = network.AuxNetwork(config["loss"]["random_dim"], 256, class_num+1)
        onemore_random_layer = network.RandomLayer([base_network.output_num(), 1], config["loss"]["random_dim"])

    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)# discriminator D
        onemore_net = network.AuxNetwork(base_network.output_num(), 256, class_num+1) # classifier F
   
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    onemore_net = onemore_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    onemore_parameter_list = onemore_net.get_parameters()
    
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    optimizer_config1 = config["optimizer1"]
    optimizer1 = optimizer_config1["type"](onemore_parameter_list, \
                    **(optimizer_config1["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    param_lr1 = []
    for param_group in optimizer1.param_groups:
        param_lr1.append(param_group["lr"])
    schedule_param1 = optimizer_config1["lr_param"]
    lr_scheduler1 = lr_schedule.schedule_dict[optimizer_config1["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        onemore_net = nn.DataParallel(onemore_net, device_ids=[int(i) for i in gpus])

    ## train   
    onemore_loss_value = transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    unknown_onemore_best_acc = known_onemore_best_acc = unknown_best_acc = known_best_acc = 0.0
    iter_num = 0
    alpha = 10
    low = 0.0
    high = 1.0
    max_iter = 10000.0
    def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
        return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

    for i in range(config["num_iterations"]):
        iter_num += 1
        coeff = calc_coeff(iter_num, high, low, alpha, max_iter)
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            onemore_net.train(False)
            unknown_temp_acc, known_temp_acc = image_classification_test(dset_loaders, \
                base_network, onemore_net, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            total_onemore_net = nn.Sequential(base_network, onemore_net)
            onemore_temp_model = nn.Sequential(total_onemore_net)
            if unknown_temp_acc > unknown_best_acc and i > 15000: # record accuracy after 15000 iterations to promise that it is stable
                unknown_best_acc = unknown_temp_acc # OS accuracy
                known_acc = known_temp_acc
                unknown_best_model = total_onemore_net
            if known_temp_acc > known_best_acc and i > 15000:
                known_best_acc = known_temp_acc #OS* accuracy
                unknown_acc = unknown_temp_acc
                known_best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, unknown_temp_acc)
            config["onemore_out_file"].write(log_str+"\n")
            config["onemore_out_file"].flush()
            print("unknown acc: "+log_str)
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, known_temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print("known acc: "+log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))
            torch.save(nn.Sequential(onemore_net), osp.join(config["output_path"], \
                "onemore_iter_{:05d}_model.pth.tar".format(i)))

        loss_params = config["loss"]                  
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        onemore_net.train(True)

        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        optimizer1 = lr_scheduler1(optimizer1, i, **schedule_param1)
        optimizer1.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source, path_source = iter_source.next()
        inputs_target, labels_target, path_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        source_softmax_out = nn.Softmax(dim=1)(outputs_source)
        target_softmax_out = nn.Softmax(dim=1)(outputs_target)
        onemore_labels = torch.cat((labels_source.cuda(), torch.from_numpy(np.array([class_num] * train_bs)).cuda()), dim=0)#labels used by classifier F
        predict_target_prob, predict_target_labels = torch.max(target_softmax_out, 1)
        target_onemore_labels = predict_target_labels.detach() + torch.from_numpy(np.array([class_num]*train_bs)).cuda()
        if config['method'] == 'CDAN+E':           
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method']  == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method']  == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        elif config['method'] == 'OSDA':
            transfer_loss, onemore_loss = loss.OSDA([features, softmax_out], ad_net, onemore_net, iter_num, onemore_labels, None, coeff, random_layer)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = classifier_loss + onemore_loss + loss_params["trade_off"]*transfer_loss
        total_loss.backward()
        optimizer.step()
        optimizer1.step()

    torch.save(known_best_model, osp.join(config["output_path"], "known_best_model.pth.tar"))
    torch.save(unknown_best_model, osp.join(config["output_path"], "unknown_onemore_best_model.pth.tar"))
    log_str = "unknown_best_acc: {:.5f}".format(unknown_best_acc)
    config["out_file"].write(log_str+"\n")
    config["out_file"].flush()
    print(log_str)
    log_str = "known_acc: {:.5f}".format(known_acc)
    config["out_file"].write(log_str+"\n")
    config["out_file"].flush()
    print(log_str)
    log_str = "known_best_acc: {:.5f}".format(known_best_acc)
    config["out_file"].write(log_str+"\n")
    config["out_file"].flush()
    print(log_str)
    log_str = "unknown_acc: {:.5f}".format(unknown_acc)
    config["out_file"].write(log_str+"\n")
    config["out_file"].flush()
    print(log_str)

    return unknown_best_acc, known_best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Against Adversarial Learning')
    parser.add_argument('method', type=str, default='SEDA', choices=['CDAN', 'CDAN+E', 'DANN', 'OSDA'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../../data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../../data/office/webcam_10_list.txt', help="The target dataset path list")
    #parser.add_argument('--t_dset_path_1', type=str, default='../../data/office/webcam_10_list.txt', help="The aux target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 30004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    config["onemore_out_file"] = open(osp.join(config["output_path"], "onemore_log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }
    config["optimizer1"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }
    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":36, "list_path_1": args.t_dset_path}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":10, "list_path_1": args.t_dset_path}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0001 # optimal parameters
            config["optimizer1"]["lr_param"]["lr"] = 0.0001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
             ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0001 # optimal parameters     
            config["optimizer1"]["lr_param"]["lr"] = 0.0001# optimal parameters     
        config["network"]["params"]["class_num"] = 10 
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config["aux_network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config["aux_network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.0001 # optimal parameters
        config["optimizer1"]["lr_param"]["lr"] = 0.001
        config["network"]["params"]["class_num"] = 25
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
