from model import *
from wordembedding import Word_embedding

import numpy as np

import argparse
# *Argument parser
parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--word', default='I')
parser.add_argument('--embedding', default='polyglot')

args = parser.parse_args()

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

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Word_embedding(embedding=args.embedding)
idx = dataset.word2idx(args.word)

inputs = dataset.embedding_vectors[idx].unsqueeze(0).to(device)
targets = dataset.embedding_vectors.to(device)
cos_dist = cosine_similarity(inputs, targets)

dist, nearest_neighbor = torch.sort(cos_dist, descending=True)
nearest_neighbor = nearest_neighbor[:, 1:6]
dist = dist[:, 1:6]
print('%.4f | ' % torch.mean(dist[0]) + args.word + '\t=> ' + dataset.idxs2sentence(nearest_neighbor[0]))
