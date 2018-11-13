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
parser.add_argument('--epoch', default=0,
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
parser.add_argument('--embedding', default='polyglot')

args = parser.parse_args()

if os.path.exists('logs/%s' % args.model): shutil.rmtree('./logs/%s/' % args.model)

logger = Logger('./logs/%s/' % args.model)
saved_model_path = 'trained_model_%s_%s/' % (args.lang, args.model)

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = 300
char_max_len = 20
word_emb_dim = 64
random_seed = 64
shuffle_dataset = False
validation_split = .8
start = int(args.epoch)

# *Hyperparameter/
batch_size = 128
val_batch_size = 3
max_epoch = int(args.maxepoch)
learning_rate = float(args.lr)
momentum = 0.2

char_embed = Char_embedding(char_emb_dim, max_len=char_max_len, random=True)
if args.load:
    char_embed.char_embedding.load_state_dict(torch.load('%scharembed.pth' % saved_model_path))

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

criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

if not os.path.exists(saved_model_path): os.makedirs(saved_model_path)


if args.load:
    model.load_state_dict(torch.load('%slstm.pth' % saved_model_path))

step = start

# *Training
word_embedding = dataset.embedding_vectors.to(device)
for epoch in tqdm(range(start, max_epoch)):
    for it, (X, y) in enumerate(train_loader):
        words = dataset.idxs2words(X)
        inputs = char_embed.char_split(words)

        # embedding = torch.stack([dataset.embedding_vectors[idx] for idx in X]).squeeze()
        # cos_dist = cosine_similarity(embedding.to(device), word_embedding)
        # _, target = torch.sort(cos_dist, descending=True)
        # target = target[:, 0].squeeze()

        inputs = Variable(inputs).to(device) # (length x batch x char_emb_dim)
        target = Variable(y).squeeze().to(device) # (batch x word_emb_dim)

        model.zero_grad()

        output = model.forward(inputs) # (batch x word_emb_dim)
        loss = criterion(output, target)

        # ##################
        # Tensorboard
        # ################## 
        info = {
            'loss-Train-%s' % args.model : loss.item(),
        }

        step += 1
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if it % int(dataset_size/(batch_size*5)) == 0:
        #     for rep in range(5):
        #         random_input = np.random.randint(len(X))
        #         # inputs = dataset.embedding_vectors[X[random_input]].unsqueeze(0).to(device)
        #         y = dataset.embedding_vectors[target[random_input]].unsqueeze(0).to(device)
        #         # print(output.size())
        #         model_output = torch.argmax(F.softmax(output[random_input], 0))
        #         out = dataset.embedding_vectors[model_output].unsqueeze(0).to(device)

        #         loss_dist = cosine_similarity(out, y)
        #         dist, nearest_neighbor = torch.sort(loss_dist, descending=True)

        #         nearest_neighbor = nearest_neighbor[:, 0]
        #         dist = dist[:, 0].data.cpu().numpy()
        #         tqdm.write('%.4f | ' % loss_dist[0] + dataset.idx2word(X[random_input]) + 
        #             '\t=> ' + 
        #             dataset.idx2word(model_output))
        #     tqdm.write('')
        if it % int(dataset_size/(batch_size*5)) == 0:
            tqdm.write('loss = %.4f' % loss)
            model.eval()
            random_input = np.random.randint(len(X))
            
            words = dataset.idx2word(X[random_input])

            inputs = char_embed.char_split(words)

            inputs = inputs.to(device) # (length x batch x char_emb_dim)
            target = target.to(device) # (batch x word_emb_dim)

            output = model.forward(inputs) # (batch x word_emb_dim)
            cos_dist = cosine_similarity(output, word_embedding)
            loss_dist = cosine_similarity(output, target[random_input].unsqueeze(0))

            dist, nearest_neighbor = torch.sort(cos_dist, descending=True)

            nearest_neighbor = nearest_neighbor[:, :5]
            dist = dist[:, :5].data.cpu().numpy()

            tqdm.write('%d %.4f | ' % (it, loss_dist[0]) + words + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[0]))
            model.train()
            tqdm.write('')
        # if it == 1: break

    # *TESTING
    # model.eval()
    # for it, (X) in enumerate(validation_loader):
    #     if it >= 1: break
        
    #     words = dataset.idxs2words(X)
    #     inputs = char_embed.char_split(words)

    #     inputs = inputs.to(device) # (length x batch x char_emb_dim)
    #     target = X.squeeze().to(device) # (batch x word_emb_dim)

    #     model.zero_grad()

    #     output = model.forward(inputs) # (batch x word_emb_dim)
    #     model_output = torch.argmax(F.softmax(output, 0), dim=1)
    #     out = dataset.word_embedding(model_output.cpu()).to(device)


    #     cos_dist = cosine_similarity(out, word_embedding)
    #     dist, nearest_neighbor = torch.sort(cos_dist, descending=True)

    #     nearest_neighbor = nearest_neighbor[:, :5]
    #     dist = dist[:, :5].data.cpu().numpy()
    #     for i, word in enumerate(X):
    #         y = dataset.embedding_vectors[target[i]].unsqueeze(0).to(device)
    #         loss_dist = cosine_similarity(out[i].unsqueeze(0), y)
    #         tqdm.write('%.4f | ' % loss_dist[0, -1] + 
    #             dataset.idx2word(X[i, -1]) + 
    #             '\t=> ' + 
    #             dataset.idxs2sentence(nearest_neighbor[i]))
    #         # *SANITY CHECK
    #         # dist_str = 'dist: '
    #         # for j in dist[i]:
    #         #     dist_str += '%.4f ' % j
    #         # tqdm.write(dist_str)
    #     tqdm.write('==================')
    # model.train()
    model.eval()
    print()
    for it, (X, y) in enumerate(validation_loader):
        if it >= 1: break
        
        words = dataset.idxs2words(X)
        inputs = char_embed.char_split(words)
        # word_embedding = dataset.embedding_vectors.to(device)
        # target = torch.stack([dataset.embedding_vectors[idx] for idx in X]).squeeze()
        target = y

        inputs = inputs.to(device) # (length x batch x char_emb_dim)
        target = target.to(device) # (batch x word_emb_dim)

        model.zero_grad()

        output = model.forward(inputs) # (batch x word_emb_dim)
        loss = criterion(output, target)

        cos_dist = cosine_similarity(output, word_embedding)
        
        # cos_dist = l2_dist(output, word_embedding)

        dist, nearest_neighbor = torch.sort(cos_dist, descending=True)

        # nearest_neighbor = np.argsort(cos_dist, 1)

        nearest_neighbor = nearest_neighbor[:, :5]
        dist = dist[:, :5].data.cpu().numpy()

        for i, word in enumerate(X):
            loss_dist = cosine_similarity(output[i].unsqueeze(0), target[i].unsqueeze(0))
            tqdm.write('%.4f | ' % loss_dist[0, -1] + dataset.idx2word(word) + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[i]))
            # *SANITY CHECK
            # dist_str = 'dist: '
            # for j in dist[i]:
            #     dist_str += '%.4f ' % j
            # tqdm.write(dist_str)

    model.train()


    torch.save(model.state_dict(), '%slstm.pth' % saved_model_path)
    torch.save(char_embed.char_embedding.state_dict(), '%scharembed.pth' % saved_model_path)