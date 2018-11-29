import torch
import torch.nn as nn

import numpy as np

from model import *
from charembedding import Char_embedding
from wordembedding import Word_embedding

import argparse


def cosine_similarity(tensor1, tensor2):
    # tensor2 += 1.e-15
    tensor1_norm = torch.norm(tensor1, 2, 1)
    tensor2_norm = torch.norm(tensor2, 2, 1)
    tensor1_dot_tensor2 = torch.mm(tensor2, torch.t(tensor1)).t()

    divisor = [t * tensor2_norm for t in tensor1_norm]
    divisor = torch.stack(divisor) + 1.e-15
    # result = (tensor1_dot_tensor2/divisor).data.cpu().numpy()
    result = (tensor1_dot_tensor2/divisor).data.cpu()

    return result


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
parser.add_argument('--loss_fn', default='mse')
parser.add_argument('--local', default=False, action='store_true',)

args = parser.parse_args()
saved_model_path = 'trained_model_%s_%s_%s' % (args.lang, args.model, args.loss_fn) if args.local else '/content/gdrive/My Drive/trained_model_%s_%s_%s' % (args.lang, args.model, args.loss_fn)

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# *Parameters
char_emb_dim = 300
char_max_len = 50
word_emb_dim = 64
random_seed = 64
shuffle_dataset = False
validation_split = .8
start = int(args.epoch)

char_embed = Char_embedding(char_emb_dim, max_len=char_max_len, random=True)
char_embed.char_embedding.load_state_dict(torch.load('%s/charembed.pth' % saved_model_path))

dataset = Word_embedding(lang=args.lang, embedding=args.embedding)
if args.model == 'lstm':
    model = mimick(char_emb_dim, char_embed.char_embedding, dataset.emb_dim, 128, 2)
else:
    model = mimick_cnn(char_emb_dim, char_embed.char_embedding, dataset.emb_dim, 10000)
model.to(device)

model.load_state_dict(torch.load('%s/%s.pth' % (saved_model_path, args.model)))
model.eval()

# *Evaluating
words = 'MCT McNeally Vercellotti Secretive corssing flatfish compartmentalize pesky lawnmower developiong hurtling expectedly'
inputs = char_embed.char_split(words)
inputs = torch.tensor(inputs).unsqueeze(1)
embedding = dataset.embedding_vectors.to(device)
inputs = inputs.to(device) # (length x batch x char_emb_dim)
output = model.forward(inputs) # (batch x word_emb_dim)

cos_dist = cosine_similarity(output, embedding)
dist, nearest_neighbor = torch.sort(cos_dist, descending=True)
nearest_neighbor = nearest_neighbor[:, :5]
dist = dist[:, :5].data.cpu().numpy()
for i, word in enumerate(words.split()):
    print('%.4f | ' % np.mean(dist[i]) + word + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[i]))