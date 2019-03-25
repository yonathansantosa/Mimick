import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Char_embedding:
    def __init__(self, char_emb_dim=300, char_max_len=15, random=False, asc=False, device='cuda'):
        '''
        Initializing character embedding
        Parameter:
        emb_dim = (int) embedding dimension for character embedding
        ascii = mutually exclusive with random
        '''
        self.char_max_len = char_max_len
        self.asc = asc
        if random and not self.asc:
            table = np.transpose(np.loadtxt('glove.840B.300d-char.txt', dtype=str, delimiter=' ', comments='##'))
            self.weight_char = np.transpose(table[1:].astype(np.float))
            self.char = np.transpose(table[0])
            self.embed = nn.Embedding(len(self.char), char_emb_dim).to(device)
        elif self.asc:
            table = np.transpose(np.loadtxt('ascii.embedding.txt', dtype=str, delimiter=' ', comments='##'))
            self.char = np.transpose(table[0])
            self.weight_char = np.transpose(table[1:].astype(np.float))

            self.weight_char = torch.from_numpy(self.weight_char).to(device)
            
            self.embed = nn.Embedding.from_pretrained(self.weight_char, freeze=False)
        else:
            table = np.transpose(np.loadtxt('glove.840B.300d-char.txt', dtype=str, delimiter=' ', comments='##'))
            self.char = np.transpose(table[0])
            self.weight_char = np.transpose(table[1:].astype(np.float))
            self.weight_char = self.weight_char[:,:char_emb_dim]

            self.weight_char = torch.from_numpy(self.weight_char).to(device)
            
            self.embed = nn.Embedding.from_pretrained(self.weight_char, freeze=False)

        self.embed.padding_idx = 1
        self.char2idx = {}
        self.idx2char = {}
        self.char_emb_dim = self.weight_char.shape[1]
        for i, c in enumerate(self.char):
            self.char2idx[c] = int(i)
            self.idx2char[i] = c

    def char_split(self, sentence, dropout=0.):
        '''
        Splitting character of a sentences then converting it
        into list of index

        Parameter:

        sentence = list of words
        '''
        char_data = []
        numbers = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
        # split_sentence = sentence.split()
        # split_sentence = sentence.split()

        for word in sentence:
            c = list(word)
            c = ['<sow>'] + c
            if len(c) > self.char_max_len:
                # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.char_max_len]]
                c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.char_max_len]]
            elif len(c) <= self.char_max_len:
                # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]                
                if len(c_idx) < self.char_max_len: c_idx.append(self.char2idx['<eow>'])
                for i in range(self.char_max_len-len(c)-1):
                    c_idx.append(self.char2idx['<pad>'])
            char_data += [c_idx]

        char_data = torch.Tensor(char_data).long()
        char_data = F.dropout(char_data, dropout)
        return char_data

    def char2ix(self, c):
        return self.char2idx[c]

    def ix2char(self, idx):
        return self.idx2char[idx]

    def idxs2word(self, idxs):
        return "".join([self.idx2char[idx] for idx in idxs])

    def get_char_vectors(self, words):
        sentence = []
        for idxs in words:
            sentence += [self.char_embedding(idxs)]

        # return torch.unsqueeze(torch.stack(sentence), 1).permute(1, 0, 2)
        return torch.stack(sentence).permute(1, 0, 2)