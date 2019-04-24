import torch
import torch.nn as nn

import numpy as np

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse


def cosine_similarity(tensor1, tensor2, neighbor=5):
    '''
    Calculating cosine similarity for each vector elements of
    tensor 1 with each vector elements of tensor 2

    Input:

    tensor1 = (torch.FloatTensor) with size N x D
    tensor2 = (torch.FloatTensor) with size M x D
    neighbor = (int) number of closest vector to be returned

    Output:

    (distance, neighbor)
    '''
    tensor1_norm = torch.norm(tensor1, 2, 1)
    tensor2_norm = torch.norm(tensor2, 2, 1)
    tensor1_dot_tensor2 = torch.mm(tensor2, torch.t(tensor1)).t()

    divisor = [t * tensor2_norm for t in tensor1_norm]

    divisor = torch.stack(divisor)

    # result = (tensor1_dot_tensor2/divisor).data.cpu().numpy()
    result = (tensor1_dot_tensor2/divisor.clamp(min=1.e-09)).data.cpu()
    d, n = torch.sort(result, descending=True)
    n = n[:, :neighbor]
    d = d[:, :neighbor]
    return d, n


# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--lang', default='en',
                    help='choose which language for word embedding')
parser.add_argument('--model', default='lstm',
                    help='choose which mimick model')
parser.add_argument('--multiplier', default=1)
parser.add_argument('--embedding', default='polyglot')
parser.add_argument('--loss_fn', default='mse')
parser.add_argument('--classif', default=200)
parser.add_argument('--local', default=False, action='store_true',)
parser.add_argument('--asc', default=False, action='store_true')
parser.add_argument('--charlen', default=20, help='maximum length')
parser.add_argument('--charembdim', default=300)
parser.add_argument('--neighbor', default=5)


args = parser.parse_args()
cloud_dir = '/content/gdrive/My Drive/train_dropout/'
saved_model_path = 'trained_model_%s_%s_%s' % (args.lang, args.model, args.loss_fn)

classif = int(args.classif)
multiplier = int(args.multiplier)

if not args.local:
    # logger_dir = cloud_dir + logger_dir
    saved_model_path = cloud_dir + saved_model_path

if args.loss_fn == 'cosine':
    print('true')
# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = int(args.charembdim)
char_max_len = int(args.charlen)
neighbor = int(args.neighbor)

char_embed = Char_embedding(char_emb_dim, char_max_len, asc=args.asc, random=True, device=device)
# char_embed.embed.load_state_dict(torch.load('%s/charembed.pth' % saved_model_path))

dataset = Word_embedding(lang=args.lang, embedding=args.embedding)
emb_dim = dataset.emb_dim

#* Initializing model
if args.model == 'lstm':
    model = mimick(char_embed.embed, char_embed.char_emb_dim, char_embed.embed, dataset.emb_dim, int(args.num_feature))
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

# *Evaluating
words = 'MCT McNeally Vercellotti Secretive corssing flatfish compartmentalize pesky lawnmower developiong hurtling expectedly'.split()
inputs = char_embed.char_split(words)

embedding = dataset.embedding_vectors.to(device)
inputs = inputs.to(device) # (length x batch x char_emb_dim)
output = model.forward(inputs) # (batch x word_emb_dim)

cos_dist, nearest_neighbor = cosine_similarity(output, embedding, neighbor)

for i, word in enumerate(words):
    print('%.4f | ' % torch.mean(cos_dist[i]) + word + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[i]))