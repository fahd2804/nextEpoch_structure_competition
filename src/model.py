pip install transformers

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class RNA_embedding(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', max_length=512):
        super(RNA_embedding, self).__init__()

        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.max_length = max_length

    def forward(self, x):  # x is (N, L) with L being sequence length
        # Tokenize RNA sequences and get BERT embeddings
        batch = [self.tokenizer.encode(seq, max_length=self.max_length, truncation=True, padding='max_length') for seq in x]
        input_ids = torch.tensor(batch).to(self.bert.device)

        with torch.no_grad():
            outputs = self.bert(input_ids)
        
        # Extract embeddings
        s = outputs.last_hidden_state  # (N, L, embedding_dim)

        # Create matrix representation
        m = s.unsqueeze(2).repeat(1, 1, s.shape[1], 1)  # (N, L, L, embedding_dim)
        m = torch.cat((m, m.permute(0, 2, 1, 3)), dim=-1)  # (N, L, L, 2*embedding_dim)

        return s, m

class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(num_features=in_channel)
        self.batch_norm2 = nn.BatchNorm2d(num_features=in_channel)

        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.batch_norm1(self.conv1(input))
        x = self.relu(x)
        x = self.batch_norm2(self.conv2(x))
        x += input

        return x

class RNA_net(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(RNA_net, self).__init__()

        self.embedding = RNA_embedding(bert_model_name=bert_model_name)

        self.module1 = nn.Sequential(
            ResBlock(768),  # BERT hidden size is 768
            ResBlock(768),
            ResBlock(768),
            ResBlock(768),
        )
        self.conv1 = nn.Conv2d(768, 384, kernel_size=3, padding=1)

        self.module2 = nn.Sequential(
            ResBlock(384),
            ResBlock(384),
            ResBlock(384),
            ResBlock(384),
        )
        self.conv2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        self.module3 = nn.Sequential(
            ResBlock(192),
            ResBlock(192),
            ResBlock(192),
            ResBlock(192),
        )
        self.conv3 = nn.Conv2d(192, 96, kernel_size=3, padding=1)

        self.module4 = nn.Sequential(
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
            ResBlock(96),
        )
        self.conv4 = nn.Conv2d(96, 1, kernel_size=3, padding=1)

    def forward(self, x):
        _, m = self.embedding(x)  # (N, d, L, L)

        m = self.conv1(self.module1(m))
        m = self.conv2(self.module2(m))
        m = self.conv3(self.module3(m))
        m = self.conv4(self.module4(m))

        output = m.squeeze(1)  # output is (N, L, L)
        output = 0.5 * (output.permute(0, 2, 1) + output)

        return output

