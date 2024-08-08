import torch
import torch.nn as nn

class RNA_embedding(nn.Module):

    '''
    This class is a simple embedding layer for RNA sequences. 

    input:
    - embedding_dim: int, dimension of the embedding space
    - vocab_size: int, size of the vocabulary (number of different nucleotides)

    output:
    - x: tensor, (N, embedding_dim, L, L), where N is the batch size, L is the length of the sequence 
    '''

    def __init__(self, embedding_dim, vocab_size=5):
        super(RNA_embedding, self).__init__()

        self.table_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_input = nn.Linear(embedding_dim*2, embedding_dim)

    def forward(self, x): # x is (N, L) -> embedded as sequence of integer

        # Sequence representation
        s = self.table_embedding(x)                         # (N, L, embedding_dim)

        # Outer concatenation to get matrix representation
        m = s.unsqueeze(2).repeat(1, 1, s.shape[1], 1)      # (N, L, L, embedding_dim)
        m = torch.cat((m, m.permute(0, 2, 1, 3)), dim=-1)   # (N, L, L, 2*embedding_dim)

        # Bring back to embedding dimension
        m = self.fc_input(m)                                # (N, L, L, embedding_dim)    

        m = m.permute(0, 3, 1, 2)                           # (N, embedding_dim , L, L)   

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

# This is the class to be developed !
class RNA_net(nn.Module):

    def __init__(self, embedding_dim):
        super(RNA_net, self).__init__()

        self.embedding = RNA_embedding(embedding_dim)

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



        # Your layers here

    def forward(self, x):
        # x is (N, L)
        _, m = self.embedding(x) # (N, d, L, L)

        m = self.conv1(self.module1(m)) # (N, 1, L, L)
        m = self.conv2(self.module2(m))
        m = self.conv3(self.module3(m))
        m = self.conv4(self.module4(m))


        output = m.squeeze(1) # output is (N, L, L)
        output = 0.5*(output.permute(0,2,1) + output)
        return output
