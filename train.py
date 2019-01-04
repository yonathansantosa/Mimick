import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, gradcheck
from torch.utils.data import SubsetRandomSampler, DataLoader

import numpy as np
import math

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse
from tqdm import trange, tqdm
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
    d, n = torch.sort(result, descending=False)
    neighbor = neighbor[:, :5]
    dist = dist[:, :5]
    return dist, neighbor

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)


# def l2_dist(tensor1, tensor2):
#     dist = torch.FloatTensor(0)
#     neighbor = torch.LongTensor(0)
#     for i, t1 in enumerate(tensor1):
#         # subtract = torch.abs(torch.add(tensor2, -1, t1))
#         # squared = torch.pow(subtract, 2)
#         # result = torch.norm(torch.pow(torch.add(tensor2, -1, t1), 2), 2, 1).unsqueeze(0)
#         d, n = torch.sort(torch.sum(torch.abs(torch.add(tensor2, -1, t1)).unsqueeze(0), 1), descending=False)
#         n = n[:, :5]
#         d = d[:, :5]
#         dist = torch.cat((dist, d))
#         neighbor = torch.cat((neighbor, n))

#     return dist, neighbor
def pairwise_distances(x, y=None, loss=False):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    if loss:
        result = F.pairwise_distance(x, y)
        return result
    else:
        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1)) 
        d, n = torch.sort(dist, descending=False)
        n = n[:, :5]
        d = d[:, :5]
        return d, n

def decaying_alpha_beta(epoch=0, loss_fn='cosine'):
    # decay = math.exp(-float(epoch)/200)
    if loss_fn == 'cosine':
        alpha = 1
        beta = 0.5
    else:
        alpha = 0.5
        beta = 1
    return alpha, beta

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
parser.add_argument('--epoch', default=0)
parser.add_argument('--asc', default=False, action='store_true')
parser.add_argument('--init_weight', default=False, action='store_true')

args = parser.parse_args()

# if os.path.exists('logs/%s' % args.model): shutil.rmtree('./logs/%s/' % args.model)

cloud_dir = '/content/gdrive/My Drive/train_dropout/'
saved_model_path = 'trained_model_%s_%s_%s' % (args.lang, args.model, args.loss_fn)
logger_dir = '%s/logs/run%s/' % (saved_model_path, args.run)
logger_val_dir = '%s/logs/val-run%s/' % (saved_model_path, args.run)
logger_val_cosine_dir = '%s/logs/val-cosine-run%s/' % (saved_model_path, args.run)


if not args.local:
    # logger_dir = cloud_dir + logger_dir
    saved_model_path = cloud_dir + saved_model_path

print(saved_model_path)
logger = Logger(logger_dir)
logger_val = Logger(logger_val_dir)
logger_val_cosine = Logger(logger_val_cosine_dir)


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

char_embed = Char_embedding(char_emb_dim, char_max_len, random=True)
if args.load or int(args.run) > 1:
    char_embed.char_embedding.load_state_dict(torch.load('%s/charembed.pth' % saved_model_path))

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

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=val_batch_size,
                                sampler=valid_sampler)

if args.model == 'lstm':
    model = mimick(char_emb_dim, char_embed.char_embedding, dataset.emb_dim, 128, 2)
else:
    model = mimick_cnn(char_max_len=char_embed.char_max_len, char_emb_dim=char_embed.char_emb_dim, emb_dim=emb_dim, num_feature=100, random=False, asc=args.asc)

model.to(device)
criterion2 = nn.MSELoss() if args.loss_fn == 'mse' else nn.CosineSimilarity()

criterion1 = nn.CosineSimilarity()
# criterion2 = nn.L1Loss()

if args.load:
    model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args.model)))
elif not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
        
word_embedding = dataset.embedding_vectors.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
# optimizer1 = optim.Adam(
#     [
#         {"params": model.conv2.parameters(), "lr": learning_rate},
#         {"params": model.conv3.parameters(), "lr": learning_rate},
#         {"params": model.conv4.parameters(), "lr": learning_rate},
#         {"params": model.conv5.parameters(), "lr": learning_rate},
#         {"params": model.conv6.parameters(), "lr": learning_rate},
#         {"params": model.mlp.parameters(), "lr": learning_rate},
#     ],
# )
# optimizer2 = optim.SparseAdam(
#     [
#         {"params": model.embed.parameters(), "lr": learning_rate},
#     ],
# )

if args.init_weight: model.apply(init_weights)

step = 0
print(model.modules())
# *Training
for epoch in trange(int(args.epoch), max_epoch, total=max_epoch, initial=int(args.epoch)):
    for it, (X, y) in enumerate(train_loader):
        alpha, beta = decaying_alpha_beta(epoch, args.loss_fn)
        words = dataset.idxs2words(X)
        inputs = char_embed.char_split(words)
        inputs = inputs.unsqueeze(1)
        inputs = char_embed.embed(inputs).float()
        inputs = Variable(inputs, requires_grad=True).to(device) # (length x batch x char_emb_dim)
        target = Variable(y).squeeze().to(device) # (batch x word_emb_dim)
        # print(target.size())
        model.zero_grad()

        output = model.forward(inputs) # (batch x word_emb_dim)
        # loss1 = torch.mean(1 - criterion1(output, target))
        loss = criterion2(output, target)
        # loss = alpha*loss1 + beta*loss2
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
        # gradcheck(model.forward, inputs[0].unsqueeze(0).unsqueeze(0), eps=1e-4)
        loss.backward()
        # optimizer1.step()
        # optimizer1.zero_grad()
        # optimizer2.step()
        # optimizer2.zero_grad()
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
            # distance, nearest_neighbor = l2_dist(output[random_input].unsqueeze(0).cpu(), word_embedding.cpu())
            distance, nearest_neighbor = pairwise_distances(output[random_input].unsqueeze(0), word_embedding)
            loss_dist = torch.dist(output[random_input], target[random_input])
            tqdm.write('%d %.4f | ' % (step, loss_dist.item()) + words + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[0]))
            model.train()
            tqdm.write('')
    
    torch.cuda.empty_cache()
    model.eval()
    print()
    ############################
    # SAVING TRAINED MODEL
    ############################

    if not args.local:
        copy_tree(logger_dir, cloud_dir+logger_dir)
        
    torch.save(model.state_dict(), '%s/%s.pth' % (saved_model_path, args.model))
    torch.save(char_embed.embed.state_dict(), '%s/charembed.pth' % saved_model_path)

    mse_loss = 0.
    cosine_dist = 0.
    for it, (X, target) in enumerate(validation_loader):
        words = dataset.idxs2words(X)
        inputs = char_embed.char_split(words, dropout=float(args.dropout))
        inputs = inputs.unsqueeze(1)
        inputs = char_embed.embed(inputs).float()
        inputs = inputs.to(device) # (length x batch x char_emb_dim)
        target = target.to(device) # (batch x word_emb_dim)

        model.zero_grad()

        output = model.forward(inputs) # (batch x word_emb_dim)
        
        # cosine_dist += ((1 - F.cosine_similarity(output, target)) / ((dataset_size-split))).sum().item()
        # mse_loss += (F.mse_loss(output, target, reduction='sum') / ((dataset_size-split)*emb_dim)).item()
        mse_loss += ((output-target)**2 / ((dataset_size-split)*emb_dim)).sum().item()
        # mse_loss += (torch.abs(output-target).sum() / ((dataset_size-split)*emb_dim)).item()
        
        if it < 1:
            # distance, nearest_neighbor = mse_loss(output.cpu(), word_embedding.cpu())
            distance, nearest_neighbor = pairwise_distances(output, word_embedding)

            # dist, nearest_neighbor = torch.sort(distance, descending=False)
            for i, word in enumerate(X):
                if i >= 3: break
                loss_dist = torch.dist(output[i], target[i])
                
                tqdm.write('%.4f | ' % loss_dist.item() + dataset.idx2word(word) + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[i]))
                # *SANITY CHECK
                # dist_str = 'dist: '
                # for j in dist[i]:
                #     dist_str += '%.4f ' % j
                # tqdm.write(dist_str)
    # total_val_loss = alpha*cosine_dist + beta*mse_loss
    total_val_loss = mse_loss
    print()
    # print('l2 validation loss =', mse_loss)
    # print('cosine validation loss =', cosine_dist)
    print('total loss =', total_val_loss)
    print()
    info_val = {
        'loss-Train-%s-run%s' % (args.model, args.run) : total_val_loss
    }
    # info_cosine_val = {
    #     'loss-Train-%s-run%s' % (args.model, args.run) : cosine_dist
    # }

    if args.run != 0:
        for tag, value in info_val.items():
            logger_val.scalar_summary(tag, value, step)
        # for tag, value in info_cosine_val.items():
        #     logger_val_cosine.scalar_summary(tag, value, step)    
    model.train()

    if not args.local:
        copy_tree(logger_val_dir, cloud_dir+logger_val_dir)
        # copy_tree(logger_val_cosine_dir, cloud_dir+logger_val_cosine_dir)
