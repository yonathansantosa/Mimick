import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler, DataLoader

import numpy as np

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse
from tqdm import tqdm
import os
from logger import Logger
import shutil
from distutils.dir_util import copy_tree
import pickle

def cosine_similarity(tensor1, tensor2):
    # tensor2 += 1.e-15
    tensor1_norm = torch.norm(tensor1, 2, 1)
    tensor2_norm = torch.norm(tensor2, 2, 1)
    tensor1_dot_tensor2 = torch.mm(tensor2, torch.t(tensor1)).t()

    divisor = [t * tensor2_norm for t in tensor1_norm]

    divisor = torch.stack(divisor)

    # result = (tensor1_dot_tensor2/divisor).data.cpu().numpy()
    result = (tensor1_dot_tensor2/divisor.clamp(min=1.e-09)).data.cpu()

    return result

def l2_dist(tensor1, tensor2):
    all_dist = []
    for t1 in tensor1:
        dist = []
        for t2 in tensor2:
            dist += [torch.dist(t1, t2, 2)]
        dist = torch.stack(dist)
        all_dist += [dist]
    all_dist = torch.stack(all_dist)
    return all_dist

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=30,
                    help='maximum iteration (default=1000)')
parser.add_argument('--run', default=0,
                    help='starting epoch (default=1000)')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')
parser.add_argument('--load', default=False, action='store_true',
                    help='whether to load model or not')
parser.add_argument('--lang', default='en',
                    help='choose which language for word embedding')
parser.add_argument('--model', default='lstm',
                    help='choose which mimick model')
parser.add_argument('--lr', default=0.1,
                    help='learning rate')
parser.add_argument('--charlen', default=20,
                    help='maximum length')
parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--loss_fn', default='mse')
parser.add_argument('--dropout', default=0)
parser.add_argument('--bsize', default=64)

args = parser.parse_args()

# if os.path.exists('logs/%s' % args.model): shutil.rmtree('./logs/%s/' % args.model)

cloud_dir = '/content/gdrive/My Drive/train_dropout/'
saved_model_path = 'trained_model_%s_%s_%s' % (args.lang, args.model, args.loss_fn)
logger_dir = '%s/logs/run%s/' % (saved_model_path, args.run)
logger_val_dir = '%s/logs/val-run%s/' % (saved_model_path, args.run)

if not args.local:
    # logger_dir = cloud_dir + logger_dir
    saved_model_path = cloud_dir + saved_model_path

print(saved_model_path)
logger = Logger(logger_dir)
logger_val = Logger(logger_val_dir)

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = 300
char_max_len = int(args.charlen)
random_seed = 64
shuffle_dataset = False
validation_split = .8

# *Hyperparameter/
batch_size = int(args.bsize)
val_batch_size = 64
max_epoch = int(args.maxepoch)
learning_rate = float(args.lr)
momentum = 0.2

char_embed = Char_embedding(char_emb_dim, max_len=char_max_len, random=True)
if args.load or int(args.run) > 1 and os.path.exists('%s/charembed.pth' % saved_model_path):
    char_embed.char_embedding.load_state_dict(torch.load('%s/charembed.pth' % saved_model_path))

dataset = Word_embedding(lang=args.lang, embedding=args.embedding)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

#* Creating PT data samplers and loaders:
train_indices, val_indices = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=val_batch_size,
                                sampler=valid_sampler)

if args.model == 'lstm':
    model = mimick(char_emb_dim, char_embed.char_embedding, dataset.emb_dim, 128, 2)
else:
    model = mimick_cnn(char_emb_dim, char_embed.char_embedding, dataset.emb_dim, 10000)

model.to(device)

criterion = nn.MSELoss() if args.loss_fn == 'mse' else nn.CosineSimilarity()
# criterion = nn.CrossEntropyLoss()

if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
else:
    if args.load or int(args.run) > 1 and os.path.exists('%s/%s.pth' % (saved_model_path, args.model)):
        model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args.model)))
        
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
step = 0

# *Training
word_embedding = dataset.embedding_vectors.to(device)
for epoch in tqdm(range(max_epoch)):
    for it, (X, y) in enumerate(train_loader):
        words = dataset.idxs2words(X)
        inputs = char_embed.char_split(words)
        inputs = Variable(inputs).to(device) # (length x batch x char_emb_dim)
        target = Variable(y).squeeze().to(device) # (batch x word_emb_dim)

        model.zero_grad()

        output = model.forward(inputs) # (batch x word_emb_dim)
    
        loss = criterion(output, target)
        if args.loss_fn == 'cosine':
            loss = 1 - loss
            loss = torch.mean(loss)
        # print(loss)

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
            model.eval()
            random_input = np.random.randint(len(X))
            
            words = dataset.idx2word(X[random_input]) # list of words  

            # inputs_test = char_embed.char_split(words)

            # inputs_test = inputs_test.to(device) # (length x batch x char_emb_dim)
            # target_test = y.to(device) # (batch x word_emb_dim)

            # output_test = model.forward(inputs_test) # (batch x word_emb_dim)
            cos_dist = cosine_similarity(output[random_input].unsqueeze(0), word_embedding)
            loss_dist = cos_dist[0, random_input].unsqueeze(0)
            
            dist, nearest_neighbor = torch.sort(cos_dist, descending=True)
            nearest_neighbor = nearest_neighbor[:, :5]
            dist = dist[:, :5].data.cpu().numpy()
            
            tqdm.write('%d %.4f | ' % (step, loss_dist[0]) + words + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[0]))
            model.train()
            tqdm.write('')
    model.eval()
    print()
    
    ############################
    # SAVING TRAINED MODEL
    ############################

    if not args.local:
        copy_tree(logger_dir, cloud_dir+logger_dir)
        
    torch.save(model.state_dict(), '%s/%s.pth' % (saved_model_path, args.model))
    torch.save(char_embed.char_embedding.state_dict(), '%s/charembed.pth' % saved_model_path)
    
    total_val_loss = 0.
    for it, (X, target) in enumerate(validation_loader):
        words = dataset.idxs2words(X)
        inputs = char_embed.char_split(words, dropout=float(args.dropout))
       
        inputs = inputs.to(device) # (length x batch x char_emb_dim)
        target = target.to(device) # (batch x word_emb_dim)

        model.zero_grad()

        output = model.forward(inputs) # (batch x word_emb_dim)
        # loss = criterion(output, target)

        loss_val = F.cosine_similarity(output, target)
        loss_val = 1 - loss_val
        loss_val = torch.sum(loss_val/(dataset_size-split))
        total_val_loss += loss_val.item()
        if it < 1:
            cos_dist = cosine_similarity(output, word_embedding)
    
            # cos_dist = l2_dist(output, word_embedding)

            dist, nearest_neighbor = torch.sort(cos_dist, descending=True)

            # nearest_neighbor = np.argsort(cos_dist, 1)

            nearest_neighbor = nearest_neighbor[:, :5]
            dist = dist[:, :5].data.cpu().numpy()
            for i, word in enumerate(X):
                if i >= 3: break
                # print(len(X))
                loss_dist = cosine_similarity(output[i].unsqueeze(0), target[i].unsqueeze(0))
                tqdm.write('%.4f | ' % loss_dist[0, -1] + dataset.idx2word(word) + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[i]))
                # total_val_loss += loss_dist[0, -1]
                # *SANITY CHECK
                # dist_str = 'dist: '
                # for j in dist[i]:
                #     dist_str += '%.4f ' % j
                # tqdm.write(dist_str)
    info = {
        'loss-val-%s-run%s' % (args.model, args.run) : total_val_loss,
    }

    if args.run != 0:
        for tag, value in info.items():
            logger_val.scalar_summary(tag, value, epoch)
    model.train()