#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

import scipy as sp
from scipy.linalg import svdvals

import numpy as np
import powerlaw

import sklearn
from sklearn.decomposition import TruncatedSVD

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, Tensor

from torch.nn.utils import prune as prune

from torch.utils.data.sampler import SubsetRandomSampler

import kneed as kneed


# In[2]:


device = torch.device('cuda')

#model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True)

#model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True)


model = model.to(device)


# In[3]:


print(model.features)


# In[ ]:





# In[4]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


#######################



train_set = torchvision.datasets.CIFAR100('./datasets', train=True, 
                                         download=True, transform=transform)
test_set = torchvision.datasets.CIFAR100('./datasets', train=False, 
                                        download=True, transform=transform)
# Number of subprocesses to use for data loading
num_workers = 4
# How many samples per batch to load
batch_size = 32
# Percentage of training set to use as validation
valid_size = 0.5

num_test = len(test_set)
indices = list(range(num_test))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_test))
test_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches
test_sampler = SubsetRandomSampler(test_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


# Prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                           num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(test_set, batch_size= 1, sampler=valid_sampler, 
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2048, sampler=test_sampler, num_workers=num_workers)

# In[5]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(valid_loader)


# In[6]:


def ten2mat(tensor):
    r,c,ch,f = tensor.shape
    new_dim = [c,r*ch*f]
    return np.reshape(tensor, new_dim)

def eff_rank(matrix):
    frob = np.linalg.norm(matrix, 'fro')
    svals = svdvals(matrix)
    S = max(svals)
    r = frob/S
    return (r)


# In[7]:


#utility functions


#reshape weight/feature tensor into a matrix
def ten2mat(tensor):
    r,c,ch,f = tensor.shape
    new_dim = [c,r*ch*f]
    return np.reshape(tensor, new_dim)
################################################################################################################
#Compute stable rank
def eff_rank(matrix):
    frob = np.linalg.norm(matrix, 'fro')
    svals = svdvals(matrix)
    S = max(svals)
    r = frob/S
    return (r)

################################################################################################################

# a dict to store the activations
activation = {}
def getActivation(name):
  # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

################################################################################################################

#Function to get conv+bn feature means
def get_mean_features(nc, num_classes, num_samples, conv_dict, model, dataiter):
    
    activation = {}
    def getActivation(name):
      # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    mat_features_list_conv = [ [] ]*(nc*num_classes)
    mat_features_list_bn = [ [] ]*(nc*num_classes)

    count = np.zeros((nc*num_classes))

    #with torch.no_grad():
    for i in range(num_samples):
        
        try:
            image,label = dataiter.next()
        except StopIteration:
            dataiter = iter(train_loader)
            image,label = dataiter.next()
        
        
        l2 =label.detach().numpy()
        l2 = l2[0]
        #print(l2)
        for j in range(nc):#range(len(conv_dict)):
            activation = {}
            z1 = conv_dict[j]
            z2 = z1 + 1

            #print(z1,z2)
            h1 = model.features[z1].register_forward_hook(getActivation('conv2d'))
            h2 = model.features[z2].register_forward_hook(getActivation('bn'))

            output = model(image.to(device))

            Xx1 = activation['conv2d']
            Xx2 = activation['bn']

            Xx1 = Xx1.to(device = 'cpu')
            Xx2 = Xx2.to(device = 'cpu')

            Xx01 = Xx1.detach().numpy()
            Xx02 = Xx2.detach().numpy()

            temp_mat1 = ten2mat(Xx01)
            temp_mat2 = ten2mat(Xx02)

            temp_idx = int(num_classes*j + l2)

            count[temp_idx]+=1
            #print(i, np.linalg.norm(temp_mat1),np.linalg.norm(temp_mat2), np.linalg.norm(temp_mat3), l2, j, z1, temp_idx)

            l31 = len(mat_features_list_conv[temp_idx])
            l32 = len(mat_features_list_bn[temp_idx])

            if l31 == 0:
                mat_features_list_conv[temp_idx] = temp_mat1

            else:
                mat_features_list_conv[temp_idx] = mat_features_list_conv[temp_idx] + temp_mat1

            if l32 == 0:
                mat_features_list_bn[temp_idx] = temp_mat2

            else:
                mat_features_list_bn[temp_idx] = mat_features_list_bn[temp_idx] + temp_mat2


            h1.remove()
            h2.remove()
       
    for i in range(nc*num_classes):
        mat_features_list_conv[i] = mat_features_list_conv[i] / count[i]
        mat_features_list_bn[i] = mat_features_list_bn[i] / count[i]
    
    return mat_features_list_conv, mat_features_list_bn, count


#################################################################################################################

#function to get differences between features

def get_features_diff(nc, num_classes, mat_features_list, conv_dict):
    
    mat_diff_list = []
    
    ll = int(0.5 * num_classes * (num_classes - 1))


    count_list = [ [[]]*ll ]*nc

    for k in range(nc):
        idx1 = 0
        idx2 = num_classes*k
        temp_list = [[]]*ll

        for i in range(num_classes):
            for j in range(i+1,num_classes):
                #print(idx1)
                ti = i+idx2
                tj = j+idx2
                temp1 = mat_features_list[ti] - mat_features_list[tj]
                
                temp_list[idx1] = temp1

                count_list[k][idx1]= (i,j)
                idx1+=1
        mat_diff_list.append([temp_list])

        
    return mat_diff_list

#################################################################################################################
        
def get_min_list(nc, num_classes, mat_diff_list):
    min_list = []
    ll = int(0.5 * num_classes * (num_classes - 1))

    for k in range(nc):
        layer_list = mat_diff_list[k][0]


        tempmat = layer_list[0]
        
        Y_conv = np.shape(tempmat)
        ll = Y_conv[0]
        #print(ll)
        temp_min_list= [[]]*ll


        for i in range(ll):
            temp1  = layer_list[i]



            temp1n = np.linalg.norm(temp1, axis=1)


            for j in range(ll):           
                if bool(temp_min_list[j]):
                    if temp_min_list[j] > temp1n[j]:
                        temp_min_list[j] = temp1n[j]                    
                else:
                    temp_min_list[j] = temp1n[j]




        min_list.append(temp_min_list)
    return min_list


################################################################################################################

def test(model, data_loader, device):
    acc = 0  # TODO compute top1
    correct_samples = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for (idx, (x, t)) in enumerate(data_loader):
            x = model.forward(x.to(device))
            t = t.to(device)
            _, indices = torch.max(x, 1)
            correct_samples += torch.sum(indices == t)
            total_samples += t.shape[0]

    acc = float(correct_samples) / total_samples
    return acc

################################################################################################################

def get_crit_vals(nc, min_list, crit_frac):
    minlist = []
    min_means = []
    min_crit = []
    for i in range(nc):
        temp_list = min_list[i] / np.max(min_list[i])
        mean = np.mean(temp_list)


        #print(i,mean_conv, mean_bn)

        crit = crit_frac* mean
        minlist.append(np.min(temp_list))
        min_means.append(mean)
        min_crit.append(crit)
    
    return min_list, min_means, min_crit

################################################################################################################

def get_pruning_mask2(nc, min_crit, min_list,model, conv_dict):
    
    Mask = [[]]*nc
    filters_pruned = [[]]*nc
    filters_remain = [[]]*nc
    
    for k in range(nc):


                #Conv output
        Scores= min_list[k] 
        Scores = Scores / np.max(Scores)
        len_sc = int(len(Scores))


        Criterion = min_crit[k]
        mask = np.ones(len_sc)
        for i in range(len(Scores)):
            temp = Scores[i]-Criterion
            if temp <= 0:
                mask[i]= 0

                
        num_rem = int(np.sum(mask))
        num_prn = int(len(mask) - num_rem)  
        
        filters_pruned[k] = num_prn
        filters_remain[k] = num_rem

        
        Test_ten = model.features[conv_dict[k]].weight
        
        Test_ten = Test_ten.to(device = 'cpu')

        Test_ten = Test_ten.detach().numpy()

        r,c,ch,f = np.shape(Test_ten)
        Test_Mask = [[]]*r
        One_ten = np.ones((c,ch,f))
        for i in range(r):
            Test_Mask[i] = One_ten*mask[i]
        X = torch.tensor(Test_Mask).to(device)
       
        
       
        
        Mask[k] = X
    return Mask, filters_pruned, filters_remain


#################################################################################################################

def prune_with_mask(nc, model, X, conv_dict):
    
    for k in range(nc):
        mm = prune.custom_from_mask(model.features[conv_dict[k]], name='weight', mask=X)
    
    return model

#################################################################################################################
def get_pruned_acc(nc, min_crit, min_list,model, conv_dict,test_loader,device):
    
    filters_pruned = [[]]*nc
    filters_remain = [[]]*nc
           
    for k in range(nc):
        #print(idx)
        
        
        #Conv output
        Scores = min_list[k] 
        Scores = Scores / np.max(Scores)
        len_sc = int(len(Scores))


        Criterion = min_crit[k]
        mask = np.ones(len_sc)
        for i in range(len(Scores)):
            temp = Scores[i]-Criterion
            if temp <= 0:
                mask[i]= 0

                
        num_rem = int(np.sum(mask))
        num_prn = int(len(mask) - num_rem)  
        
        filters_pruned[k] = num_prn
        filters_remain[k] = num_rem

      
        
        
        
        Test_ten = model.features[conv_dict[k]].weight
        
        Test_ten = Test_ten.to(device = 'cpu')

        Test_ten = Test_ten.detach().numpy()

        r,c,ch,f = np.shape(Test_ten)
        Test_Mask = [[]]*r
        One_ten = np.ones((c,ch,f))
        for i in range(r):
            Test_Mask[i] = One_ten*mask[i]
        X = torch.tensor(Test_Mask).to(device)
       
        m1 = prune.custom_from_mask( model.features[conv_dict[k]], name='weight', mask=X)
        
        
        #print(model_accs1[k, idx])
        #print(k, crit_frac, num_pruned, pf, T1)
    
    #device = torch.device('cuda')
    
    T1 = test(model, test_loader,device)
    T1 = 100*T1
    
    return model, T1


# In[8]:


U = 60

# accs = np.zeros(U)

accs = []

# num_pruned = np.zeros(U)
# num_remain = np.zeros(U)

num_pruned = []
num_remain = []


# conv_dict = [0,3,7,10,14,17,20,24,27,30, 34, 37, 40]
conv_dict = [0,4,8,11,15,18,22,25]
# conv_dict = [0,3,7,10,14,17,20,23,27,30,33,36,40,43,46,49]

nc = int(len(conv_dict))
num_classes = 100
num_samples = 4096

fracs = [.99,.9,.8, .7, .65,.6,.5,.4,.3,.2,.1,0]
Crit_frac = .5

threshold = 56.00
frac_count = 0
accuracy = 100000
uu=0

while Crit_frac > 0:
    #print(uu)
    mat_features_list_conv, mat_features_list_bn, count = get_mean_features(nc, num_classes, num_samples, conv_dict, model, dataiter) 
    diff_list_conv = get_features_diff(nc, num_classes, mat_features_list_conv, conv_dict)
    diff_list_bn = get_features_diff(nc, num_classes, mat_features_list_bn, conv_dict)
    min_list_conv = get_min_list(nc, num_classes, diff_list_conv)
    min_list_bn = get_min_list(nc, num_classes, diff_list_bn)
   
    minlist_bn, minmeans_bn, min_crit_bn = get_crit_vals(nc, min_list_bn, Crit_frac)
    
#     test_model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True)
    #test_model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    test_model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True)


    test_model1 = test_model1.to(device)
    
    Mask_bn, filters_pruned_bn, filters_remain_bn = get_pruning_mask2(nc, min_crit_bn, min_list_bn,test_model1, conv_dict)
    
#     num_pruned[uu] = np.sum(filters_pruned_bn)
#     num_remain[uu] = np.sum(filters_remain_bn)
    fpbn = np.sum(filters_pruned_bn)
    frbn = np.sum(filters_remain_bn)
    np.append(num_pruned, fpbn)
    np.append(num_remain, frbn)
    
    pf = 100 * (fpbn / (fpbn + frbn))
    
    for k in range(nc):
        mm = prune.custom_from_mask( test_model1.features[conv_dict[k]], name='weight', mask=Mask_bn[k])
        mm = prune.remove(mm, name='weight')
    
    T2 = test(test_model1, test_loader, device)
    accuracy = T2 * 100
    #accs[uu] = T2 *100
    np.append(accs,accuracy)
    
    if accuracy >= threshold:
        for k in range(nc):
            mmm = prune.custom_from_mask( model.features[conv_dict[k]], name='weight', mask=Mask_bn[k])
            mmm = prune.remove(mmm, name='weight')
            
        
        print(Crit_frac, uu, T2*100, pf, 100-pf)
    else:
        print(Crit_frac, uu, T2*100, pf, 100-pf)
        frac_count +=1
        Crit_frac = np.maximum(0, Crit_frac - 0.025)
        
    uu +=1
        
    
    
    
    


# In[9]:


a_file = open("test_vgg11_cifar100.txt", "a")


np.savetxt(a_file, [filters_pruned_bn], fmt = '%1.1i')

np.savetxt(a_file, [filters_remain_bn], fmt = '%1.1i')

np.savetxt(a_file, [accuracy], fmt = '%1.4f')


a_file.close()

