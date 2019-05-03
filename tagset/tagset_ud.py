from xml.dom import minidom
import pyconll
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





# tag = train[10][0].upos
# idx = ttoi[tag]
# print(itot[idx])
# print(ttoi[tag])
# print(tag)

# print(train[10].text)
# print(len(train[10].text.split()))

class Tagset:
    def __init__(self):
        mydoc = minidom.parse('tagset/UD_English-GUM/stats.xml')

        self.items = mydoc.getElementsByTagName('tag')

        self.itot = {}
        self.toti = {}
        for it, elem in enumerate(self.items):
            self.itot[it] = elem.attributes['name'].value
            self.toti[elem.attributes['name'].value] = it
        self.itot[len(self.itot)] = 'UNK'
        self.toti['UNK'] = len(self.itot)-1

    def __len__(self):
        return len(self.itot)
        
    def idx2tag(self, idx):
        return self.itot[idx]

    def tag2idx(self, tag):
        return self.toti[tag]

class Postag:
    def __init__(self, word_embed, device='cuda'):
        my_conll_file_location = 'tagset/UD_English-GUM/en_gum-ud-dev.conllu'
        self.word_embed = word_embed
        self.train = pyconll.load_from_file(my_conll_file_location)
        self.tagged_words = list(zip(
            [token.form for sentence in self.train._sentences for token in sentence], 
            [token.upos for sentence in self.train._sentences for token in sentence]))
        self.tagset = Tagset()
        self.count_bin = torch.zeros(len(self.tagset))
        
        new_itot = {}
        new_toti = {}

        for sentence in self.train:
            for token in sentence:
                self.count_bin[self.tagset.tag2idx(token.upos)] += 1
            
        _, self.idxs = torch.sort(self.count_bin, descending=True)

        for it, i in enumerate(self.idxs):
            new_itot[it] = self.tagset.itot[int(i)]
            new_toti[new_itot[it]] = it

        self.tagset.toti = new_toti
        self.tagset.itot = new_itot


    def __len__(self):
        return len(self.train)


    def __getitem__(self, index):
        length = len(self.train[index])
        word = []
        tag = []
        
        if length-5 <= 0:
            for i in range(length):
                w = self.train[index][i].form
                t = self.train[index][i].upos
                word += [self.word_embed.word2idx(w)]
                tag_id = self.tagset.tag2idx(t)
                tag += [tag_id]

            for i in range(length, 5):
                word += [self.word_embed.word2idx('<pad>')]
                tag_id = self.tagset.tag2idx('UNK')

                tag += [tag_id]

        else:
            start_index = np.random.randint(0, length-5)
            for i in range(start_index, start_index+5):
                w = self.train[index][i].form
                t = self.train[index][i].upos
                word += [self.word_embed.word2idx(w)]
                tag_id = self.tagset.tag2idx(t)
                tag += [tag_id]
        
        # word_emb = self.word_embed.word_embedding(torch.tensor(word).to(self.device))
        # return (word_emb, torch.LongTensor(tag).view(len(tag), 1), torch.LongTensor(word).view(len(word), 1))
        return (torch.LongTensor(word), torch.LongTensor(tag))

class Postagger_adaptive(nn.Module):
    def __init__(self, seq_length, emb_dim, hidden_size, output_size):
        super(Postagger_adaptive, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
        self.lstm.flatten_parameters()
        
        self.out = nn.AdaptiveLogSoftmaxWithLoss(hidden_size, output_size, cutoffs=[round(output_size/5),2*round(output_size/5)], div_value=4)
        
    def forward(self, inputs, targets):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(inputs)

        output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]

        output = output.view(output.shape[0]*output.shape[1], -1)
        targets = targets.view(targets.shape[0]*targets.shape[1])

        return self.out(output, targets)

    def validation(self, inputs, targets):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(inputs)

        output = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]

        output = output.view(output.shape[0]*output.shape[1], -1)
        targets = targets.view(targets.shape[0]*targets.shape[1])

        prediction = self.out.predict(output)
        _, loss = self.out(output, targets)

        return prediction, float(loss.cpu())

# a = Postag('a')
# print(a.train)
# print([token.form for sentence in a.train._sentences for token in sentence])
