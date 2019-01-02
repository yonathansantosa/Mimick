import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class mimick(nn.Module):
    def __init__(self, char_emb_dim, char_emb, emb_dim, n_h, n_hl):
        super(mimick, self).__init__()
        self.embed = char_emb
        self.lstm = nn.LSTM(char_emb_dim, n_h, n_hl, bidirectional=True, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_h*n_hl*2, 250),
            nn.Tanh(),
            nn.Linear(250, emb_dim),
            nn.Tanh(),
        )

    def forward(self, inputs):
        embedding = self.embed(inputs)
        out_forw, (forw_h, c) = self.lstm(embedding)
        out_cat = torch.cat([hidden for hidden in forw_h], 1)
        out = self.mlp(out_cat)

        return out

class mimick_cnn(nn.Module):
    def __init__(self, char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, random=False):
        super(mimick_cnn, self).__init__()
        table = np.transpose(np.loadtxt('glove.840B.300d-char.txt', dtype=str, delimiter=' ', comments='##'))
        self.char = np.transpose(table[0])
        self.max_len = char_max_len

        if random:
            self.embed = nn.Embedding(len(self.char), char_emb_dim, sparse=True)
        else:
            self.weight_char = np.transpose(table[1:].astype(np.float))
            self.weight_char = self.weight_char[:,:char_emb_dim]

            self.weight_char = torch.from_numpy(self.weight_char)
            
            self.embed = nn.Embedding.from_pretrained(self.weight_char, freeze=False, sparse=True)

        self.char2idx = {}
        self.idx2char = {}

        for i, c in enumerate(self.char):
            self.char2idx[c] = int(i)
            self.idx2char[i] = c
        
        # self.embed = char_embedding
        self.conv2 = nn.Conv2d(1, num_feature, (2, char_emb_dim))
        self.conv3 = nn.Conv2d(1, num_feature, (3, char_emb_dim))
        self.conv4 = nn.Conv2d(1, num_feature, (4, char_emb_dim))
        self.conv5 = nn.Conv2d(1, num_feature, (5, char_emb_dim))
        self.conv6 = nn.Conv2d(1, num_feature, (6, char_emb_dim))

        self.mlp = nn.Sequential(
            nn.Linear(5*num_feature, 450),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            nn.Linear(450, 400),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            nn.Linear(400, 350),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            nn.Linear(350, emb_dim),
            # nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = self.embed(inputs).float()
        x2 = F.relu(self.conv2(inputs)).squeeze(-1)
        x3 = F.relu(self.conv3(inputs)).squeeze(-1)
        x4 = F.relu(self.conv4(inputs)).squeeze(-1)
        x5 = F.relu(self.conv5(inputs)).squeeze(-1)
        x6 = F.relu(self.conv6(inputs)).squeeze(-1)

        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(-1)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(-1)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(-1)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(-1)
        x6 = F.max_pool1d(x6, x6.size(2)).squeeze(-1)
        
        out_cat = torch.cat([x2, x3, x4, x5, x6], dim=1)

        out = self.mlp(out_cat)

        return out

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
            if len(c) > self.max_len:
                # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.max_len]]
                c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c[:self.max_len]]
            elif len(c) <= self.max_len:
                # c_idx = [self.char2idx['#'] if x in numbers else self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]
                c_idx = [self.char2idx[x] if x in self.char2idx else self.char2idx['<unk>'] for x in c]                
                if len(c_idx) < self.max_len: c_idx.append(self.char2idx['<eow>'])
                for i in range(self.max_len-len(c)-1):
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