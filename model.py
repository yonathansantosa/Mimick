import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    def __init__(self, char_emb_dim, char_emb, emb_dim, n_vocab, num_feature=100):
        super(mimick_cnn, self).__init__()
        self.embed = char_emb
        self.conv2 = nn.Conv2d(1, num_feature, (2, char_emb_dim))
        self.conv3 = nn.Conv2d(1, num_feature, (3, char_emb_dim))
        self.conv4 = nn.Conv2d(1, num_feature, (4, char_emb_dim))
        self.conv5 = nn.Conv2d(1, num_feature, (5, char_emb_dim))
        self.conv6 = nn.Conv2d(1, num_feature, (6, char_emb_dim))

        self.mlp = nn.Sequential(
            nn.Linear(5*num_feature, 450),
            nn.Hardtanh(),
            nn.Linear(450, 400),
            nn.Hardtanh(),
            nn.Linear(400, emb_dim),
            nn.Hardtanh(),
            # nn.Linear(400, 300),
            # nn.Hardtanh()
        )

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = self.embed(inputs)

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