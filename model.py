import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

# *Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mimick(nn.Module):
    def __init__(self, embedding, char_emb_dim,emb_dim, hidden_size):
        super(mimick, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(char_emb_dim, self.hidden_size, 1, bidirectional=True, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
        )

    def forward(self, inputs):
        x = self.embedding(inputs).float()
        _, (hidden_state, _) = self.lstm(x)
        out_cat = (hidden_state[0, :, :] + hidden_state[1, :, :])
        out = self.mlp(out_cat)

        return out

class mimick_cnn(nn.Module):
    def __init__(self, embedding,  char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, random=False, asc=False):
        super(mimick_cnn, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.conv2 = nn.Conv2d(1, num_feature, (2, char_emb_dim), bias=False)
        self.conv3 = nn.Conv2d(1, num_feature, (3, char_emb_dim), bias=False)
        self.conv4 = nn.Conv2d(1, num_feature, (4, char_emb_dim), bias=False)
        self.conv5 = nn.Conv2d(1, num_feature, (5, char_emb_dim), bias=False)
        self.conv6 = nn.Conv2d(1, num_feature, (6, char_emb_dim), bias=False)
        self.conv7 = nn.Conv2d(1, num_feature, (7, char_emb_dim), bias=False)
        self.inputs = None

        self.mlp1 = nn.Sequential(
            nn.Linear(num_feature*6, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.t = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        self.inputs = inputs
        x = self.embedding(self.inputs).float()
        x2 = self.conv2(x).relu().squeeze(-1)
        x3 = self.conv3(x).relu().squeeze(-1)
        x4 = self.conv4(x).relu().squeeze(-1)
        x5 = self.conv5(x).relu().squeeze(-1)
        x6 = self.conv6(x).relu().squeeze(-1)
        x7 = self.conv7(x).relu().squeeze(-1)


        x2_max = F.max_pool1d(x2, x2.shape[2]).squeeze(-1)
        x3_max = F.max_pool1d(x3, x3.shape[2]).squeeze(-1)
        x4_max = F.max_pool1d(x4, x4.shape[2]).squeeze(-1)
        x5_max = F.max_pool1d(x5, x5.shape[2]).squeeze(-1)
        x6_max = F.max_pool1d(x6, x6.shape[2]).squeeze(-1)
        x7_max = F.max_pool1d(x7, x7.shape[2]).squeeze(-1)

        
        maxpoolcat = torch.cat([x2_max, x3_max, x4_max, x5_max, x6_max, x7_max], dim=1)

        out_cnn = self.mlp1(maxpoolcat)
        
        out = self.t(out_cnn) * self.mlp2(out_cnn) + (1 - self.t(out_cnn)) * out_cnn

        return out

class mimick_cnn2(nn.Module):
    def __init__(self, embedding,  char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, random=False, asc=False):
        super(mimick_cnn2, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.conv1 = nn.Conv2d(1, num_feature, (2, char_emb_dim))
        self.conv2 = nn.Conv1d(num_feature, num_feature, 2)
        self.conv3 = nn.Conv1d(num_feature, emb_dim, 2)
        self.conv4 = nn.Conv1d(emb_dim, emb_dim, 2)


        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-3.0, max_val=3.0),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.t = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.embedding(inputs).float()
        x2_conv1 = self.conv1(x).relu().squeeze(-1)
        x2_max1 = F.max_pool1d(x2_conv1, 2).squeeze(-1)
        x2_conv2 = self.conv2(x2_max1).relu()
        x2_max2 = F.max_pool1d(x2_conv2, 2)
        x2_conv3 = self.conv3(x2_max2).relu()
        x2_max3 = F.max_pool1d(x2_conv3, x2_conv3.shape[2]).squeeze(-1)

        # maxpoolcat = torch.cat([x2_max, x3_max, x4_max, x5_max, x6_max, x7_max], dim=2).view(inputs.size(0), -1)

        out_cnn = self.mlp1(x2_max3)

        out = self.t(out_cnn) * self.mlp2(out_cnn) + (1 - self.t(out_cnn)) * out_cnn
        
        return out

class mimick_cnn3(nn.Module):
    def __init__(self, embedding, char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, mtp=1, random=False, asc=False):
        super(mimick_cnn3, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.conv2 = nn.Conv2d(1, num_feature, (2, embedding.embedding_dim), bias=False)
        self.conv3 = nn.Conv2d(1, num_feature, (3, embedding.embedding_dim), bias=False)
        self.conv4 = nn.Conv2d(1, num_feature, (4, embedding.embedding_dim), bias=False)
        self.conv5 = nn.Conv2d(1, num_feature, (5, embedding.embedding_dim), bias=False)
        self.conv6 = nn.Conv2d(1, num_feature, (6, embedding.embedding_dim), bias=False)
        self.conv7 = nn.Conv2d(1, num_feature, (7, embedding.embedding_dim), bias=False)

        self.featloc = nn.Sequential(
            nn.Linear(num_feature*99, emb_dim),
            nn.Sigmoid()
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-mtp*3, max_val=mtp*3),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardtanh(min_val=-mtp*3, max_val=mtp*3),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

        self.t = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.embedding(inputs).float()
        x2 = self.conv2(x).sigmoid().squeeze(-1)
        x3 = self.conv3(x).sigmoid().squeeze(-1)
        x4 = self.conv4(x).sigmoid().squeeze(-1)
        x5 = self.conv5(x).sigmoid().squeeze(-1)
        x6 = self.conv6(x).sigmoid().squeeze(-1)
        x7 = self.conv7(x).sigmoid().squeeze(-1)

        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x4 = x4.view(x4.shape[0], -1)
        x5 = x5.view(x5.shape[0], -1)
        x6 = x6.view(x6.shape[0], -1)
        x7 = x7.view(x7.shape[0], -1)

        concat = torch.cat([x2,x3,x4,x5,x6,x7], dim=1)

        feature_loc = self.featloc(concat)
        
        out_cnn = self.mlp1(feature_loc)
        
        out = self.t(out_cnn) * self.mlp2(out_cnn) + (1 - self.t(out_cnn)) * out_cnn

        return out

class mimick_cnn4(nn.Module):
    def __init__(self, embedding, char_max_len=15, char_emb_dim=300, emb_dim=300, num_feature=100, classif=200, random=False, asc=False):
        super(mimick_cnn4, self).__init__()
        self.embedding = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim)
        self.embedding.weight.data.copy_(embedding.weight.data)
        self.conv2 = nn.Conv2d(1, num_feature, (2, char_emb_dim))
        self.conv3 = nn.Conv2d(1, num_feature, (3, char_emb_dim))
        self.conv4 = nn.Conv2d(1, num_feature, (4, char_emb_dim))
        self.conv5 = nn.Conv2d(1, num_feature, (5, char_emb_dim))
        self.conv6 = nn.Conv2d(1, num_feature, (6, char_emb_dim))
        self.conv7 = nn.Conv2d(1, num_feature, (7, char_emb_dim))

        self.classif = nn.Sequential(
            nn.Linear(num_feature*48, classif),
            nn.LogSoftmax()
        )

        self.regres = nn.Sequential(
            nn.Linear(classif, emb_dim),
            nn.Hardtanh(min_val=-3, max_val=3)
        )

    def forward(self, inputs):
        x2 = self.conv2(inputs).relu().squeeze(-1)
        x3 = self.conv3(inputs).relu().squeeze(-1)
        x4 = self.conv4(inputs).relu().squeeze(-1)
        x5 = self.conv5(inputs).relu().squeeze(-1)
        x6 = self.conv6(inputs).relu().squeeze(-1)
        x7 = self.conv7(inputs).relu().squeeze(-1)


        x2_max = F.max_pool1d(x2, 2).squeeze(-1)
        x3_max = F.max_pool1d(x3, 2).squeeze(-1)
        x4_max = F.max_pool1d(x4, 2).squeeze(-1)
        x5_max = F.max_pool1d(x5, 2).squeeze(-1)
        x6_max = F.max_pool1d(x6, 2).squeeze(-1)
        x7_max = F.max_pool1d(x7, 2).squeeze(-1)

        
        maxpoolcat = torch.cat([x2_max, x3_max, x4_max, x5_max, x6_max, x7_max], dim=2).view(inputs.size(0), -1)

        c = self.classif(maxpoolcat)

        out = self.regres(c)

        return out