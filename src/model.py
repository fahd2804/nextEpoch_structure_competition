import torch
import torch.nn as nn

class RNA_embedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size=5):
        super(RNA_embedding, self).__init__()
        self.table_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_input = nn.Linear(embedding_dim*2, embedding_dim)

    def forward(self, x):
        s = self.table_embedding(x)                         # (N, L, embedding_dim)
        m = s.unsqueeze(2).repeat(1, 1, s.shape[1], 1)      # (N, L, L, embedding_dim)
        m = torch.cat((m, m.permute(0, 2, 1, 3)), dim=-1)   # (N, L, L, 2*embedding_dim)
        m = self.fc_input(m)                                # (N, L, L, embedding_dim)
        m = m.permute(0, 3, 1, 2)                           # (N, embedding_dim, L, L)
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
    def __init__(self, embedding_dim):
        super(RNA_net, self).__init__()
        self.embedding = RNA_embedding(embedding_dim)
        
        self.module1 = nn.Sequential(
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim),
            ResBlock(embedding_dim)
        )
        self.conv1 = nn.Conv2d(embedding_dim, embedding_dim//2, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.module2 = nn.Sequential(
            ResBlock(embedding_dim//2),
            ResBlock(embedding_dim//2),
            ResBlock(embedding_dim//2),
            ResBlock(embedding_dim//2)
        )
        self.conv2 = nn.Conv2d(embedding_dim//2, embedding_dim//4, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.module3 = nn.Sequential(
            ResBlock(embedding_dim//4),
            ResBlock(embedding_dim//4),
            ResBlock(embedding_dim//4),
            ResBlock(embedding_dim//4)
        )
        self.conv3 = nn.Conv2d(embedding_dim//4, embedding_dim//8, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.module4 = nn.Sequential(
            ResBlock(embedding_dim//8),
            ResBlock(embedding_dim//8),
            ResBlock(embedding_dim//8),
            ResBlock(embedding_dim//8)
        )
        self.conv4 = nn.Conv2d(embedding_dim//8, embedding_dim//16, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(embedding_dim//16, embedding_dim//8, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Conv2d(embedding_dim//8, embedding_dim//4, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv7 = nn.Conv2d(embedding_dim//4, embedding_dim//2, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv8 = nn.Conv2d(embedding_dim//2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        _, m = self.embedding(x) # (N, d, L, L)
        
        m = self.pool1(self.conv1(self.module1(m)))
        m = self.pool2(self.conv2(self.module2(m)))
        m = self.pool3(self.conv3(self.module3(m)))
        m = self.pool4(self.conv4(self.module4(m)))
        
        m = self.upsample4(m)
        m = self.conv5(m)
        m = self.upsample3(m)
        m = self.conv6(m)
        m = self.upsample2(m)
        m = self.conv7(m)
        m = self.upsample1(m)
        m = self.conv8(m)

        output = m.squeeze(1) # output is (N, L, L)
        output = 0.5 * (output.permute(0, 2, 1) + output)
        return output
