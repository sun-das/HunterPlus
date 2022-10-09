import random 
import torch
import numpy as np
from copy import deepcopy
from src.constants import *
from src.adahessian import Adahessian
import matplotlib.pyplot as plt

def convertToOneHot(dat, cpu_old, HOSTS):
    alloc = []
    for i in dat:
        oneHot = [0] * HOSTS; alist = i.tolist()[-HOSTS:]
        oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
    new_dat_oneHot = torch.cat((cpu_old, torch.FloatTensor(alloc)), dim=1)
    return new_dat_oneHot

def scaleToHosts(containers):
    max_value = max(containers)
    min_value = min(containers)
    #print(containers)
    if (max_value-min_value) != 0:
        #print("Containers:")
        #print(containers)
        normalised_values = [int ((x-min_value)*3.99/(max_value-min_value))  for x in containers]
        #print("Normalised_values:")
        #print(normalised_values)
    else:
        normalised_values = [np.random.randint(0,4) for x in containers] 
    assigned_hosts = normalised_values
    return assigned_hosts

def convertToCorrectImage_2(old_data):
    containers_alloc = []
    for i in range(6,8):
        for j in range(3,13):
            containers_alloc.append(old_data[i,j])

    containers_alloc = scaleToHosts(containers_alloc)
    containers_1 = containers_alloc[0:10]
    containers_2 = containers_alloc[10:20]
    for i in range(10):
        old_data[6, i+3] = containers_1[i]
    for i in range(10):
        old_data[7, i+3] = containers_2[i] 

    new_image = deepcopy(old_data)
    return new_image

def convertToCorrectImage(old_data):
    for i in range(6,8):
        for j in range(3,13):
            max_value = max(255, old_data[i,j])
            actual_value = int((max_value+1)/64)*64 - 1
            old_data[i,j] = actual_value
    for i in range(1,3):
        max_value = max(255, old_data[0,i])
        actual_value = int((max_value+1)/25)*25 - 1
        old_data[0, i] = actual_value
    for i in range(0,2):
        max_value = max(255, old_data[1,i])
        actual_value = int((max_value+1)/25)*25 - 1
        old_data[1, i] = actual_value
    new_image = deepcopy(old_data)
    return new_image

def opt(init, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = torch.optim.AdamW([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:,0:-HOSTS]); alloc_old = deepcopy(init.data[:,-HOSTS:])
        z = model(init)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = convertToOneHot(init.data, cpu_old, HOSTS)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-HOSTS:])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    #     zs.append(z.item())
    # plt.plot(zs); plt.show(); plt.clf()
    init.requires_grad = False 
    return init.data, iteration, model(init)

def optCNN(init, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = torch.optim.AdamW([init], lr = 0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        old_data = deepcopy(init.data)
        alloc_old = deepcopy(init.data[6:8, 3:13])
        image_1 = torch.unsqueeze(init, 0)
        image_1 = torch.unsqueeze(image_1, 0)
        #print("Iteration:")
        #print(alloc_old)
        #print("Init (Before optimization):")
        #print(init)
        z = model(image_1)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = convertToCorrectImage_2(old_data)
        #print("Init (After optimization):")
        #print(init)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[6:8, 3:13])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    init.requires_grad = False
    image_2 = torch.unsqueeze(init, 0)
    image_2 = torch.unsqueeze(image_2, 0)
    return init.data, iteration, model(image_2)


def optCNN_2(init, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = torch.optim.AdamW([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:,0:-HOSTS]); alloc_old = deepcopy(init.data[:,-HOSTS:])
        #print("I am hereeeee!!!")
        image_1 = torch.unsqueeze(init, 0)
        image_1 = torch.unsqueeze(image_1, 0)
        z = model(image_1)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = convertToOneHot(init.data, cpu_old, HOSTS)
        #print(init.data[:,-HOSTS:])
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-HOSTS:])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    init.requires_grad = False 
    image_2 = torch.unsqueeze(init, 0)
    image_2 = torch.unsqueeze(image_2, 0)
    return init.data, iteration, model(image_2)

def optGNN(init, graph, data, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = torch.optim.AdamW([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:,0:-HOSTS]); alloc_old = deepcopy(init.data[:,-HOSTS:])
        #print("I am hereeeee!!!")
        z = model(graph, data, init)
        #print("fffffff")
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = convertToOneHot(init.data, cpu_old, HOSTS)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-HOSTS:])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    init.requires_grad = False 
    return init.data, iteration, model(graph, data, init)

def so_opt(init, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = Adahessian([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:,0:-HOSTS]); alloc_old = deepcopy(init.data[:,-HOSTS:])
        z = model(init)
        optimizer.zero_grad(); z.backward(create_graph=True); optimizer.step(); scheduler.step()
        init.data = convertToOneHot(init.data, cpu_old, HOSTS)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-HOSTS:])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    #     zs.append(z.item())
    # plt.plot(zs); plt.show(); plt.clf()
    init.requires_grad = False 
    return init.data, iteration, model(init)
