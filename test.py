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

# def save_iteration(iteration, local):
#     iteration_file = 'iteration.pkl'
#     with open(iteration_file, 'wb') as f:
#         pickle.dump(iteration, f)
#     if not local:
#         from google.colab import files
#         files.download(iteration_file)

# def load_iteration(local):
#     iteration_file = 'iteration.pkl'
#     with open(iteration_file, 'rb') as f:
#         itx = pickle.load(f)
#     return itx

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

args = parser.parse_args()

# if os.path.exists('logs/%s' % args.model): shutil.rmtree('./logs/%s/' % args.model)

cloud_dir = '/content/gdrive/My Drive/'
saved_model_path = 'trained_model_%s_%s_%s' % (args.lang, args.model, args.loss_fn)
logger_dir = '%s/logs/run%s/' % (saved_model_path, args.run)

if not args.local:
    logger_dir = cloud_dir + logger_dir
    saved_model_path = cloud_dir + saved_model_path

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = 300
char_max_len = int(args.charlen)
word_emb_dim = 64
random_seed = 64
shuffle_dataset = False
validation_split = .8

val_batch_size = 64

char_embed = Char_embedding(char_emb_dim, max_len=char_max_len, random=True)
char_embed.char_embedding.load_state_dict(torch.load('%s/charembed.pth' % saved_model_path))

dataset = Word_embedding(lang=args.lang, embedding=args.embedding)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
total_val_size = dataset_size - split

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

#* Creating PT data samplers and loaders:
train_indices, val_indices = indices[:split], indices[split:]
valid_sampler = SubsetRandomSampler(val_indices)
validation_loader = DataLoader(dataset, batch_size=val_batch_size,
                                sampler=valid_sampler)

if args.model == 'lstm':
    model = mimick(char_emb_dim, char_embed.char_embedding, dataset.emb_dim, 128, 2)
else:
    model = mimick_cnn(char_emb_dim, char_embed.char_embedding, dataset.emb_dim, 10000)

model.to(device)

criterion = nn.MSELoss() if args.loss_fn == 'mse' else nn.CosineSimilarity()

model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args.model)))



# *Training
word_embedding = dataset.embedding_vectors.to(device)
model.eval()
total_loss = 0.0
for it, (X, y) in enumerate(validation_loader):
    words = dataset.idxs2words(X)
    inputs = char_embed.char_split(words)
    # # word_embedding = dataset.embedding_vectors.to(device)
    # # target = torch.stack([dataset.embedding_vectors[idx] for idx in X]).squeeze()
    # target = y

    # inputs = inputs.to(device) # (length x batch x char_emb_dim)
    # target = target.to(device) # (batch x word_emb_dim)

    # model.zero_grad()

    # output = model.forward(inputs) # (batch x word_emb_dim)

    # cos_dist = cosine_similarity(output, word_embedding)

    # dist, nearest_neighbor = torch.sort(cos_dist, descending=True)

    # nearest_neighbor = nearest_neighbor[:, :5]
    # dist = dist[:, :5].data.cpu().numpy()
    
    print(X.size())
    print(inputs.size())
    # print(output.size())
    print(y.size())

    # for i, word in enumerate(X):
    #     loss_dist = cosine_similarity(output[i].unsqueeze(0), target[i].unsqueeze(0))
    #     # print(loss_dist)
    #     total_loss += float(loss_dist[0, -1])/total_val_size
    #     tqdm.write('%.4f | ' % loss_dist[0, -1] + dataset.idx2word(word) + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[i]))
    #     # *SANITY CHECK
    #     # dist_str = 'dist: '
    #     # for j in dist[i]:
    #     #     dist_str += '%.4f ' % j
    #     # tqdm.write(dist_str)

print(total_loss)
# print('total loss = ', np.mean(total_loss))