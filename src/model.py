import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertModel, BertTokenizer

class RNA_embedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size=5):
        super(RNA_embedding, self).__init__()
        self.table_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_input = nn.Linear(embedding_dim*2, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        s = self.table_embedding(x)
        s = self.dropout(s)
        m = s.unsqueeze(2).repeat(1, 1, s.shape[1], 1)
        m = torch.cat((m, m.permute(0, 2, 1, 3)), dim=-1)
        m = self.fc_input(m)
        m = self.dropout(m)
        m = m.permute(0, 3, 1, 2)
        return s, m

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=in_channel)
        self.batch_norm2 = nn.BatchNorm2d(num_features=in_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        x = self.batch_norm1(self.conv1(input))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch_norm2(self.conv2(x))
        x = self.dropout(x)
        x += input
        return x

class RNA_net(nn.Module):
    def __init__(self, embedding_dim):
        super(RNA_net, self).__init__()
        self.embedding = RNA_embedding(embedding_dim)
        self.transformer = TransformerBlock(embedding_dim)
        
        self.module1 = nn.Sequential(
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
        )
        self.conv1 = nn.Conv2d(embedding_dim, embedding_dim//2, kernel_size=3, padding=1)
        
        self.module2 = nn.Sequential(
            ResBlock(embedding_dim//2),
            ResBlock(embedding_dim//2),
            ResBlock(embedding_dim//2),
            ResBlock(embedding_dim//2),
        )
        self.conv2 = nn.Conv2d(embedding_dim//2, embedding_dim//4, kernel_size=3, padding=1)
        
        self.module3 = nn.Sequential(
            ResBlock(embedding_dim//4),
            ResBlock(embedding_dim//4),
            ResBlock(embedding_dim//4),
            ResBlock(embedding_dim//4),
        )
        self.conv3 = nn.Conv2d(embedding_dim//4, embedding_dim//8, kernel_size=3, padding=1)
        
        self.module4 = nn.Sequential(
            ResBlock(embedding_dim//8),
            ResBlock(embedding_dim//8),
            ResBlock(embedding_dim//8),
            ResBlock(embedding_dim//8),
        )
        self.conv4 = nn.Conv2d(embedding_dim//8, 1, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        _, m = self.embedding(x)
        m = self.transformer(m.permute(0, 2, 3, 1).contiguous())
        m = m.permute(0, 3, 1, 2)
        
        m = self.conv1(self.module1(m))
        m = self.dropout(m)
        m = self.conv2(self.module2(m))
        m = self.dropout(m)
        m = self.conv3(self.module3(m))
        m = self.dropout(m)
        m = self.conv4(self.module4(m))

        output = m.squeeze(1)
        output = 0.5 * (output.permute(0, 2, 1) + output)
        return output

# Example usage
if __name__ == "__main__":
    sequences = ["ACGU", "UGCA"]
    batch_size = len(sequences)
    sequence_length = max(len(seq) for seq in sequences)
    vocab_size = 5

    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'P': 4}
    pad_idx = char_to_idx['P']
    
    batch = [[char_to_idx.get(char, pad_idx) for char in seq] + [pad_idx] * (sequence_length - len(seq)) for seq in sequences]
    batch_tensor = torch.tensor(batch)

    model = RNA_net(embedding_dim=128)
    output = model(batch_tensor)
    print(output)
