import numpy as np
import torch
import torch.nn as nn

from polyglot.mapping import Embedding

from torchtext.vocab import Vectors, GloVe

class Word_embedding:
    def __init__(self, emb_dim=300, lang='en', embedding='polyglot'):
        '''
        Initializing word embedding
        Parameter:
        emb_dim = (int) embedding dimension for word embedding
        '''
        if embedding == 'glove':
            # *GloVE
            glove = GloVe('6B', dim=emb_dim)
            self.embedding_vectors = glove.vectors
            self.stoi = glove.stoi
            self.itos = glove.itos
        elif embedding == 'word2vec':
            # *word2vec
            word2vec = Vectors('GoogleNews-vectors-negative300.bin.gz.txt')
            self.embedding_vectors = word2vec.vectors
            self.stoi = word2vec.stoi
            self.itos = word2vec.itos
        elif embedding == 'polyglot':
            # *Polyglot
            print(lang)
            polyglot_emb = Embedding.load('embeddings2/%s/embeddings_pkl.tar.bz2' % lang)
            self.embedding_vectors = torch.from_numpy(polyglot_emb.vectors)
            self.stoi = polyglot_emb.vocabulary.word_id
            self.itos = polyglot_emb.vocabulary.id_word
        
        self.word_embedding = nn.Embedding.from_pretrained(self.embedding_vectors, freeze=True, sparse=True)
        self.emb_dim = self.embedding_vectors.size(1)
    def __getitem__(self, index):
        return (torch.tensor([index], dtype=torch.long), self.word_embedding(torch.tensor([index])).squeeze())

    def __len__(self):
        return len(self.itos)
    
    def word2idx(self, c):
        return self.stoi[c]

    def idx2word(self, idx):
        return self.itos[int(idx)]

    def idxs2sentence(self, idxs):
        return ' '.join([self.itos[int(i)] for i in idxs])

    def sentence2idxs(self, sentence):
        word = sentence.split()
        return [self.stoi[w] for w in word]

    def idxs2words(self, idxs):
        '''
        Return tensor of indexes as a sentence
        
        Input:
        idxs = (torch.LongTensor) 1D tensor contains indexes
        '''
        idxs = idxs.squeeze()
        sentence = [self.itos[int(idx)] for idx in idxs]
        return sentence

    def get_word_vectors(self):
        return self.word_embedding