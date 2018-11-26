import numpy as np
import torch
import torch.nn as nn


class Char_embedding:
    def __init__(self, emb_dim=300, max_len=15, random=False):
        '''
        Initializing character embedding
        Parameter:
        emb_dim = (int) embedding dimension for character embedding
        '''
        table = np.transpose(np.loadtxt('glove.840B.300d-char.txt', dtype=str, delimiter=' ', comments='##'))
        self.char = np.transpose(table[0])
        self.max_len = max_len

        if random:
            self.char_embedding = nn.Embedding(len(self.char), emb_dim)
        else:
            self.weight_char = np.transpose(table[1:].astype(np.float))
            self.weight_char = self.weight_char[:,:emb_dim]

            self.weight_char = torch.from_numpy(self.weight_char)
            
            self.char_embedding = nn.Embedding.from_pretrained(self.weight_char, freeze=False)

        self.char2idx = {}
        self.idx2char = {}

        for i, c in enumerate(self.char):
            self.char2idx[c] = int(i)
            self.idx2char[i] = c

    def char_split(self, sentence):
        '''
        Splitting character of a sentences then converting it
        into list of index

        Parameter:


        sentence = (str) input sentence
        '''
        char_data = []
        numbers = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
        # split_sentence = sentence.split()
        split_sentence = sentence.split()

        for word in split_sentence:
            c = list(word)
            dropout_prob = 0.2
            dropout_rand = np.random.uniform()
            if len(c) > self.max_len:
                # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.max_len]]
                c_idx = [self.char2idx[x] if (x in self.char2idx and dropout_rand > dropout_prob) else self.char2idx['<unk>'] for x in c[:self.max_len]]                
            elif len(c) <= self.max_len:
                # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                c_idx = [self.char2idx[x] if (x in self.char2idx and dropout_rand > dropout_prob) else self.char2idx['<unk>'] for x in c]                
                if len(c_idx) < self.max_len: c_idx.append(self.char2idx['<eow>'])
                for i in range(self.max_len-len(c)-1):
                    c_idx.append(self.char2idx['<pad>'])
            char_data += [c_idx]
        return torch.Tensor(char_data).long()

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