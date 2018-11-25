import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import MyDataset 
import torch.nn.functional as F
import numpy as np
import glob
import random
import time
from CNN import *
import scipy.io as sio
import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
dataset = sys.argv[1]
if dataset == 'redwood':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 500
batch_size = 100
learning_rate = 0.001

if dataset == 'redwood':
    training_folder = '/home/xrhuang/Projects/zhenpei/Learn2Sync/training/' 
else:
    training_folder = '/media/xrhuang/DATA1/'

#if os.path.exists('%s.train_list' % dataset):
assert os.path.exists('%s.test_list' % dataset)
with open('%s.train_list' % dataset, 'r') as fin:
    train_models = ['%s/%s' % (training_folder, line.strip()) for line in fin.readlines()]
with open('%s.test_list' % dataset, 'r') as fin:
    test_models = ['%s/%s' % (training_folder, line.strip()) for line in fin.readlines()]
#else:
#    training_folder = os.path.normpath(training_folder)
#    models = glob.glob('{}/{}'.format(training_folder, '*'))
#    random.shuffle(models)
#    n = len(models)
#    split_point = n * 2 // 3
#    train_models = models[:split_point]
#    test_models = models[split_point:]
#
#    with open('%s.train_list' % dataset, 'w') as fout:
#        for model in train_models:
#            fout.write('%s\n' % model)
#    with open('%s.test_list' % dataset, 'w') as fout:
#        for model in test_models:
#            fout.write('%s\n' % model)

train_dataset = MyDataset(train_models, dataset, 'train')

test_dataset = MyDataset(test_models, dataset, 'test')

print('%d training samples, %d test samples' % (len(train_dataset), len(test_dataset)))
print('train models=%s, test models=%s' % (train_models, test_models))

#test_dataset = torchvision.datasets.MNIST(root='../../data', 
#                                          train=False, 
#                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           num_workers=4,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          num_workers=4,
                                          shuffle=False)


model = CNN().to(device)

#model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

prefix = '%s.models/model.ckpt' % dataset
# Train the model
total_step = len(train_loader)
save_counter = 0
checkpoints = glob.glob('%s-*' % prefix)
if len(checkpoints) != 0:
    largest = -1
    load_path = ''
    for checkpoint in checkpoints:
        save_counter = int(checkpoint.split('/')[-1].split('-')[-1])
        if save_counter > largest:
            largest = save_counter
            load_path = checkpoint
    model, optimizer, save_counter = load_checkpoint(model, optimizer, load_path)
    dump_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            if ('fc2' in name):
                print(name)
                dump_dict[name] = np.array(param.data)
                #sio.savemat('%s_25_%s.mat' % (dataset, name), mdict={name: np.array(param.data)})
filelist = train_dataset.files + test_dataset.files

start_epoch = 0
while os.path.exists('%s.predicts/%d.mat' % (dataset, start_epoch)):
    start_epoch += 1

for epoch in range(start_epoch, num_epochs):
    start_time = time.time()
    preds = []
    ground_truths = []
    feats = []
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        loading_time = time.time() - start_time
        # Forward pass
        outputs, last_layer = model(images)
        preds.append(outputs.data)
        predicted = torch.round(outputs).data
        ground_truths.append(labels)
        feats.append(last_layer.data)
        #with open('scannet.log', 'a', buffering=1) as fout:
        #    for c in range(min(100, len(np.array(outputs.data)))):
        #        fout.write(filelist[i*100+c])
        #        fout.write(' ')
        #        fout.write('%f' % np.array(outputs.data)[c])
        #        fout.write(' ')
        #        fout.write('%f' % np.array(labels.data)[c])
        #        for j in range(128):
        #            fout.write(' ')
        #            fout.write('%f' % np.array(last_layer.data)[c, j])
        #        fout.write('\n')
        
        #output0 = model(images[:, 0:1, :, :])
        #output1 = model(images[:, 1:2, :, :])
        #score = torch.sigmoid(output0 + output1)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loading Time: {:.4f}, Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loading_time, loss.item()))
            start_time = time.time()
        #print('train', i)
        if (i+1) % 800 == 0:
            save_checkpoint(save_counter, model, optimizer, prefix)
            save_counter += 1
    #with open('%s.predicts/train%s' % (dataset, epoch), 'w') as fout:
    #    for pred, gt in zip(preds, ground_truths):
    #        fout.write('%f %d\n' % (pred, gt))
    #break
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct1 = 0
        correct0 = 0
        correct = 0
        total1 = 0
        total0 = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs, last_layer = model(images)
            feats.append(last_layer.data)
            preds.append(outputs.data)
            predicted = torch.round(outputs).data
            ground_truths.append(labels)
            total0 += labels.data[labels == 0].size()[0]
            total1 += labels[labels == 1].size()[0]
            total += labels.size()[0]
            correct1 += (predicted[labels == 1] == 1).sum().item()
            correct0 += (predicted[labels == 0] == 0).sum().item()
            correct += (predicted == labels).sum().item()
            #print('test', i)
            #with open('scannet.test_log', 'a', buffering=1) as fout:
            #    for c in range(min(100, len(np.array(outputs.data)))):
            #        fout.write(filelist[i*100+c])
            #        fout.write(' ')
            #        fout.write('%f' % np.array(outputs.data)[c])
            #        fout.write(' ')
            #        fout.write('%f' % np.array(labels.data)[c])
            #        for j in range(128):
            #            fout.write(' ')
            #            fout.write('%f' % np.array(last_layer.data)[c, j])
            #        fout.write('\n')
        #preds = np.concatenate(preds, axis=0)
        #ground_truths = np.concatenate(ground_truths, axis=0)
        #with open('%s.predicts/%s' % (dataset, epoch), 'w') as fout:
        #    for pred, gt in zip(preds, ground_truths):
        #        fout.write('%f %d\n' % (pred, gt))
        print('total0=%d, total1=%d, total=%d' % (total0, total1, total))
        print('Accuracy for 1: {}, for 0: {}, for all: {} %'.format(100.0 * correct1 / total1, 100.0 * correct0 / total0, 100.0 * correct / total))
    dump_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ('fc2' in name):
                dump_dict[name] = np.array(param.data)
    try:
        preds = np.concatenate(preds, axis=0)
        ground_truths = np.concatenate(ground_truths, axis=0)
        feats = np.concatenate(feats, axis=0)
        dump_dict['predict'] = preds
        dump_dict['gt'] = ground_truths
        dump_dict['feat'] = feats 
        assert not os.path.exists('%s.predicts/%d.mat' % (dataset, epoch))
        sio.savemat('%s.predicts/%d.mat' % (dataset, epoch), mdict=dump_dict)
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()
        print(e)
    
#torch.save(model.state_dict(), '%s.models/model.ckpt' % dataset)

# Save the model checkpoint
