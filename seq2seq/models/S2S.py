import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class S2S(nn.Module):

    def __init__(self, num_features=22,
                       size_embedding=21,
                       bidirectional=False,
                       encoder_hidden_size=250,
                       decoder_hidden_size=250):
        
        self.num_features = num_features
        
        super().__init__()
        self.encoder = nn.LSTM(input_size=size_embedding, 
                               hidden_size=encoder_hidden_size, 
                               num_layers=1, 
                               batch_first=True, 
                               bidirectional=bidirectional)
        
        # Concatenate the prev sequence + embedding
        self.decoder = nn.LSTM(input_size=size_embedding * 2,
                               hidden_size=decoder_hidden_size,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=False)
        
        # Embed the one hot vector 22 into 21 -> 21 b/c easier to concatenate w/ the PSSM row this way
        self.embedding = nn.Embedding(num_features, size_embedding)
        
        self.hidden_to_pssm = nn.Linear(decoder_hidden_size, 21)
        

    def forward(self, x, pssm):
        # Convert to non one-hot for embedding
        x = x.argmax(axis=1)
        
        # Embedding layer
        x = self.embedding(x)
        
        # Don't need the singular hidden state
        _, (h, c) = self.encoder(x)
        
        first_seq = self.hidden_to_pssm(h).permute([1, 0, 2])
        seq_holder = torch.zeros_like(pssm).to(x.device)
        
        # Non linearity + PSSM are in sigmoid range in the first place
        seq_holder[:, 0:1, :] = F.sigmoid(first_seq)
        seq_holder[:, 1:, :] = pssm[:, :x.shape[1]-1, :]

        # Teacher force pssm during training
        x = torch.cat([x, pssm], axis=2)
        
        out, _ = self.decoder(x, (h, c))
        out = F.sigmoid(self.hidden_to_pssm(out))

        return out
    
    def gen(self, x):
        # Convert to non one-hot for embedding
        x = x.argmax(axis=1)

        # Embedding layer
        x = self.embedding(x)

        # Don't need the singular hidden state
        _, (h, c) = self.encoder(x)

        gen_seq = []
        ht, ct = h, c
        pred_seq = F.sigmoid(self.hidden_to_pssm(h).permute([1, 0, 2]))
        
        for t in range(x.shape[1]):
            xt = x[:, t:t+1, :]
            xt = torch.cat([xt, pred_seq], axis=2)

            out, (ht, ct) = self.decoder(xt, (ht, ct))
            pred_seq = F.sigmoid(self.hidden_to_pssm(out))
            
            gen_seq.append(pred_seq)

        gen_seq = torch.cat(gen_seq, dim=1)

        return gen_seq