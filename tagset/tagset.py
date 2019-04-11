import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


np.random.seed(0)

class Tagset:
    def __init__(self, tagset='brown'):
        self.idx2tags = {}
        self.tags2idx = {}
        with open ('tagset/%s.txt' % tagset, "r") as myfile:
            data=myfile.readlines()
            sent = "".join([d for d in data])
            processed = re.findall(r"(.*):", sent)
            for i, tag in enumerate(processed):
                self.tags2idx[tag] = i

            for i, tag in enumerate(processed):
                self.idx2tags[i] = tag

    def __len__(self):
        return len(self.idx2tags)
        
    def idx2tag(self, id):
        return self.idx2tags[id]

    def tag2idx(self, tag):
        return self.tags2idx[tag]


class Postag:
    def __init__(self, char_embed, corpus='brown', tagset='brown'):
        if corpus == 'brown':
            from nltk.corpus import brown as corpus
        self.char_embed = char_embed
        self.tagged_words = corpus.tagged_words(tagset='brown')
        self.tagged_sents = corpus.tagged_sents(tagset='brown')
        self.tagset = Tagset(tagset=tagset)

    def __len__(self):
        return len(self.tagged_sents)

    def __getitem__(self, index):
        length = len(self.tagged_sents[index])
        word = []
        tag = []
        
        if length-5 <= 0:
            for i in range(length):
                w, t = self.tagged_sents[index][i]
                if t in self.tagset.tags2idx:
                    tag_id = self.tagset.tag2idx(t)
                else:
                    tag_id = self.tagset.tag2idx('UNK')
                word += [w]
                tag += [tag_id]
            for i in range(length, 5):
                w = '<pad>'
                tag_id = self.tagset.tag2idx('UNK')
                word += [w]
                tag += [tag_id]
        else:
            start_index = np.random.randint(0, length-5)
            for i in range(start_index, start_index+5):
                w, t = self.tagged_sents[index][i]
                if t in self.tagset.tags2idx:
                    tag_id = self.tagset.tag2idx(t)
                else:
                    tag_id = self.tagset.tag2idx('UNK')
                word += [w]
                tag += [tag_id]

        word = self.char_embed.char_split(word)
        return (torch.LongTensor(word), torch.LongTensor(tag))


class Postagger(nn.Module):
    def __init__(self, seq_length, emb_dim, hidden_size, output_size):
        super(Postagger, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
        self.lstm.flatten_parameters()
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, output_size),
            nn.LogSoftmax(dim=2),
        )

        
    def forward(self, inputs):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(inputs)

        output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        
        out = self.mlp(output)

        return out