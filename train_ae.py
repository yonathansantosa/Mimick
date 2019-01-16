import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, gradcheck
from torch.utils.data import SubsetRandomSampler, DataLoader

import numpy as np
import math

from autoencoder import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse
from tqdm import trange, tqdm
import os
from logger import Logger
import shutil
from distutils.dir_util import copy_tree
import pickle

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=30, help='maximum iteration (default=1000)')
parser.add_argument('--run', default=0, help='starting epoch (default=1000)')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
parser.add_argument('--load', default=False, action='store_true', help='whether to load model or not')
parser.add_argument('--lang', default='en', help='choose which language for word embedding')
parser.add_argument('--model', default='autoencoder', help='choose which mimick model')
parser.add_argument('--lr', default=0.1, help='learning rate')
parser.add_argument('--charlen', default=20, help='maximum length')
parser.add_argument('--charembdim', default=300)
parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--loss_fn', default='mse')
parser.add_argument('--dropout', default=0)
parser.add_argument('--bsize', default=64)
parser.add_argument('--epoch', default=0)
parser.add_argument('--asc', default=False, action='store_true')
parser.add_argument('--init_weight', default=False, action='store_true')
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--nesterov', default=False, action='store_true')
parser.add_argument('--num_feature', default=100)
parser.add_argument('--weight_decay', default=0)
parser.add_argument('--momentum', default=0)
parser.add_argument('--encoder', nargs='+', type=int, default='200 100')
parser.add_argument('--decoder', nargs='+', type=int, default='100 200')
parser.add_argument('--latent_dim', default=50)



args = parser.parse_args()

cloud_dir = '/content/gdrive/My Drive/train_dropout/'
saved_model_path = 'autoencoder'
logger_dir = '%s/logs/run%s/' % (saved_model_path, args.run)
logger_val_dir = '%s/logs/val-run%s/' % (saved_model_path, args.run)


if not args.local:
    # logger_dir = cloud_dir + logger_dir
    saved_model_path = cloud_dir + saved_model_path

print(saved_model_path)
logger = Logger(logger_dir)

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = int(args.charembdim)
char_max_len = int(args.charlen)
random_seed = 64
shuffle_dataset = args.shuffle
validation_split = .8
latent_dim = int(args.latent_dim)

# *Hyperparameter/
batch_size = int(args.bsize)
val_batch_size = 64
max_epoch = int(args.maxepoch)
learning_rate = float(args.lr)
weight_decay = float(args.weight_decay)
momentum = float(args.momentum)

dataset = Word_embedding(lang=args.lang, embedding=args.embedding)
emb_dim = dataset.emb_dim

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

#* Creating PT data samplers and loaders:
train_indices, val_indices = indices[:split], indices[split:]

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=valid_sampler)

enc = encoder(emb_dim=emb_dim, layers=args.encoder, latent_dim=latent_dim)
dec = encoder(emb_dim=emb_dim, layers=args.decoder, latent_dim=latent_dim)

enc.to(device)
dec.to(device)

criterion = nn.MSELoss()

if args.load:
    enc.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, 'encoder')))
    dec.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, 'decoder')))
    
elif not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
        
word_embedding = dataset.embedding_vectors.to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=args.nesterov)

if args.init_weight: model.apply(init_weights)

step = 0

# *Training
for epoch in trange(int(args.epoch), max_epoch, total=max_epoch, initial=int(args.epoch)):
    for it, (X, y) in enumerate(train_loader):

        inputs = Variable(dataset.embedding_vectors(X)).to(device)
        target = Variable(y).squeeze().to(device) # (batch x word_emb_dim)

        output = model.forward(inputs1) # (batch x word_emb_dim)
        loss = criterion(output, target)

        # ##################
        # Tensorboard
        # ################## 
        info = {
            'loss-Train-%s-run%s' % (args.model, args.run) : loss.item(),
        }
        # save_iteration(step, args.local)

        step += 1
        if args.run != 0:
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if it % int(dataset_size/(batch_size*5)) == 0:
            tqdm.write('loss = %.4f' % loss)
    
    torch.cuda.empty_cache()
    model.eval()
    # conv2weight -= model.conv2.weight.data
    # mlpweight -= model.mlp[2].weight.data

    # print('mlp =', mlpweight.mean().data)
    # print('conv2 =', conv2weight.mean().data)

    print()
    ############################
    # SAVING TRAINED MODEL
    ############################

    if not args.local:
        copy_tree(logger_dir, cloud_dir+logger_dir)
        
    torch.save(enc.state_dict(), '%s/%s.pth' % (saved_model_path, 'encoder'))
    torch.save(dec.state_dict(), '%s/%s.pth' % (saved_model_path, 'decoder'))