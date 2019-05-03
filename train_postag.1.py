import nltk
import numpy as np
from nltk.corpus import brown
from tagset.tagset import Postag, Postagger, Postagger_adaptive

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, gradcheck
from torch.utils.data import SubsetRandomSampler, DataLoader

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse
from tqdm import trange, tqdm
import os
from logger import Logger
import shutil
from distutils.dir_util import copy_tree

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(0.01)
        m.bias.data.fill_(0.01)

# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--maxepoch', default=30, help='maximum iteration (default=1000)')
parser.add_argument('--run', default=0, help='starting epoch (default=1000)')
parser.add_argument('--save', default=False, action='store_true', help='whether to save model or not')
parser.add_argument('--load', default=False, action='store_true', help='whether to load model or not')
parser.add_argument('--lang', default='en', help='choose which language for word embedding')
parser.add_argument('--model', default='lstm', help='choose which mimick model')
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
parser.add_argument('--quiet', default=False, action='store_true')
parser.add_argument('--init_weight', default=False, action='store_true')
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--nesterov', default=False, action='store_true')
parser.add_argument('--loss_reduction', default=False, action='store_true')
parser.add_argument('--oov_random', default=False, action='store_true')
parser.add_argument('--num_feature', default=100)
parser.add_argument('--weight_decay', default=0)
parser.add_argument('--momentum', default=0)
parser.add_argument('--multiplier', default=1)
parser.add_argument('--classif', default=200)
parser.add_argument('--neighbor', default=5)
parser.add_argument('--seq_len', default=5)
parser.add_argument('--seed', default=64)

args = parser.parse_args()

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cloud_dir = '/content/gdrive/My Drive/train_dropout/'
saved_model_path = 'trained_model_%s_%s_%s' % (args.lang, args.model, args.loss_fn)
saved_postag_path = 'trained_model_%s_%s_%s_postag' % (args.lang, args.model, args.loss_fn)
logger_dir = '%s/logs/run%s/' % (saved_postag_path, args.run)
logger_val_dir = '%s/logs/val-run%s/' % (saved_postag_path, args.run)
logger_val_cosine_dir = '%s/logs/val-cosine-run%s/' % (saved_postag_path, args.run)

if not args.local:
    # logger_dir = cloud_dir + logger_dir
    saved_model_path = cloud_dir + saved_model_path
    saved_postag_path = cloud_dir + saved_postag_path
    
logger = Logger(logger_dir)
logger_val = Logger(logger_val_dir)

# *Parameters
char_emb_dim = int(args.charembdim)
char_max_len = int(args.charlen)
random_seed = int(args.seed)
shuffle_dataset = args.shuffle
validation_split = .8
neighbor = int(args.neighbor)
seq_len = int(args.seq_len)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# *Hyperparameter
batch_size = int(args.bsize)
val_batch_size = 64
max_epoch = int(args.maxepoch)
learning_rate = float(args.lr)
weight_decay = float(args.weight_decay)
momentum = float(args.momentum)
multiplier = float(args.multiplier)
classif = int(args.classif)

char_embed = Char_embedding(char_emb_dim, char_max_len, asc=args.asc, random=True, device=device)
# char_embed.embed.load_state_dict(torch.load('%s/charembed.pth' % saved_model_path))
# char_embed.embed.eval()

#* Initializing model
word_embedding = Word_embedding(lang=args.lang, embedding=args.embedding)
if not args.oov_random: word_embedding.update_weight('%s/trained_embedding_%s.txt' % (saved_model_path, args.model))
emb_dim = word_embedding.emb_dim

if args.model == 'lstm':
    model = mimick(char_embed.embed, char_embed.char_emb_dim, char_embed.embed, emb_dim, int(args.num_feature))
elif args.model == 'cnn2':
    model = mimick_cnn2(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature), 
        random=False, asc=args.asc)
elif args.model == 'cnn':
    model = mimick_cnn(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature), 
        random=False, asc=args.asc)
elif args.model == 'cnn3':
    model = mimick_cnn3(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature),
        mtp=multiplier, 
        random=False, asc=args.asc)
elif args.model == 'cnn4':
    model = mimick_cnn4(
        embedding=char_embed.embed,
        char_max_len=char_embed.char_max_len, 
        char_emb_dim=char_embed.char_emb_dim, 
        emb_dim=emb_dim,
        num_feature=int(args.num_feature),
        classif=classif,
        random=False, asc=args.asc)
else:
    model = None

model.to(device)
model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args.model)))
model.eval()

#* Creating PT data samplers and loaders:
dataset = Postag(word_embedding, model, char_embed, args.model)
new_word = []

for (word, _) in dataset.tagged_words:
    if args.oov_random:
        if word not in word_embedding.stoi:
            new_word += [torch.normal(torch.zeros(emb_dim, dtype=torch.float), std=3.0, out=None)]
            
            word_embedding.stoi[word] = len(word_embedding.stoi)
            word_embedding.itos += word
    else:
        if word not in word_embedding.stoi:
            inputs = char_embed.word2idxs(word).unsqueeze(0).to(device).detach()
            if args.model != 'lstm': inputs = inputs.unsqueeze(1)
            output = model.forward(inputs).detach()
            word_embedding.stoi[word] = len(word_embedding.stoi)
            word_embedding.itos += word
            new_word += [output.cpu()]


if args.oov_random:
    new_word += [torch.normal(torch.zeros(emb_dim, dtype=torch.float), std=3.0, out=None)]
    new_word = torch.stack(new_word).squeeze()
    word_embedding.stoi['<pad>'] = len(word_embedding.stoi)
    word_embedding.itos += '<pad>'
else:
    inputs = char_embed.word2idxs('<pad>').unsqueeze(0).to(device).detach()
    if args.model != 'lstm': inputs = inputs.unsqueeze(1)
    output = model.forward(inputs).detach()                    
    new_word += [output.cpu()]
    new_word = torch.stack(new_word).squeeze()
    word_embedding.stoi['<pad>'] = len(word_embedding.stoi)
    word_embedding.itos += '<pad>'
    
word_embedding.word_embedding.weight.data = torch.cat((word_embedding.word_embedding.weight.data, new_word)).to(device)

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

train_indices, val_indices = indices[:split], indices[split:]

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, 
                                sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=val_batch_size,
                                sampler=valid_sampler)


postagger = Postagger_adaptive(seq_len, emb_dim, 20, len(dataset.tagset)).to(device)

# postagger = Postagger(seq_len, emb_dim, 20, len(dataset.tagset)).to(device)

if args.load:
    postagger.load_state_dict(torch.load('%s/postag.pth' % (saved_postag_path)))
    
optimizer = optim.SGD(postagger.parameters(), lr=learning_rate, momentum=momentum, nesterov=args.nesterov)
criterion = nn.NLLLoss()

if args.init_weight: postagger.apply(init_weights)
step = 0

print('before training')

#* Training
for epoch in trange(int(args.epoch), max_epoch, total=max_epoch, initial=int(args.epoch)):
# for epoch in range(int(args.epoch), max_epoch):
    loss_item = 0.
    postagger.train()
    for it, (X, y) in enumerate(train_loader):
        postagger.zero_grad()
        # print('here')
        inputs = Variable(word_embedding.word_embedding(X.to(device)), requires_grad=True)
        target = Variable(y).to(device)
        # output = postagger.forward(w_embedding, target).permute(0, 2, 1)
        output, loss = postagger.forward(inputs, target)

        # loss = criterion(output, target)
        
        # ##################
        # Tensorboard
        # ################## 
        loss_item = loss.item()
        info = {
            'loss-Train-%s-postag-run%s' % (args.model, args.run) : loss_item,
        }

        step += 1
        if args.run != 0:
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    if not args.local:
        copy_tree(logger_dir, cloud_dir+logger_dir)
        
    torch.save(postagger.state_dict(), '%s/postag.pth' % (saved_postag_path))
    if not args.quiet: tqdm.write('%d | %.4f ' % (epoch, loss_item))

    #* Validation
    postagger.eval()
    validation_loss = 0.
    accuracy = 0.
    for it, (X, y) in enumerate(validation_loader):
        inputs = Variable(word_embedding.word_embedding(X.to(device)))
        target = Variable(y).to(device)
        # output = postagger.forward(w_embedding).permute(0, 2, 1)
        output, validation_loss = postagger.validation(inputs, target)
        # output_tag = postagger.predict(output.view(X.shape[0], seq_len))
        output_tag = output.view(X.shape[0], seq_len)
        accuracy += float((output_tag == target).sum())

        # validation_loss += criterion(output, target)*X.shape[0]/(len(val_indices))
        if not args.quiet:
            if it == 0:
                tag = output_tag[0]
                # output_tag = postagger.predict(output.view(X.shape[0], seq_len)[0])
                for i in range(len(X[0])):
                    word_idx = X[0][i].cpu().numpy()
                    word = word_embedding.idx2word(word_idx)
                    tg = dataset.tagset.idx2tag(int(tag[i].cpu()))
                    tgt = dataset.tagset.idx2tag(int(y[0][i]))
                    tqdm.write('(%s, %s) => %s' % (word, tgt, tg))
    accuracy /= (seq_len*len(val_indices))
    if not args.quiet: tqdm.write('accuracy = %.4f' % accuracy)

    info_val = {
        'loss-Train-%s-postag-run%s' % (args.model, args.run) : validation_loss
    }

    if args.run != 0:
        for tag, value in info_val.items():
            logger_val.scalar_summary(tag, validation_loss, step)
   
    if not args.quiet: tqdm.write('val_loss %.4f ' % validation_loss)
    
    postagger.train()

postagger.eval()

accuracy = 0.
for it, (X, y) in enumerate(validation_loader):
    inputs = Variable(word_embedding.word_embedding(X.to(device)))
    target = Variable(y).to(device)
    # output = postagger.forward(w_embedding).permute(0, 2, 1)
    output, _ = postagger.validation(inputs, target)
    # output_tag = postagger.predict(output.view(X.shape[0], seq_len))
    output_tag = output.view(X.shape[0], seq_len)
    accuracy += float((output_tag == target).sum())
    if it <= 3:
        tag = output_tag[0]
        for i in range(len(X[0])):
            word_idx = X[0][i].cpu().numpy()
            word = word_embedding.idx2word(word_idx)
            tg = dataset.tagset.idx2tag(int(tag[i].cpu()))
            tgt = dataset.tagset.idx2tag(int(y[0][i]))
            tqdm.write('(%s, %s) => %s' % (word, tgt, tg))
        tqdm.write('\n')
accuracy /= (seq_len*len(val_indices))
print('accuracy = %.4f' % accuracy)
    