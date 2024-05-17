##Language Encoder
##Language Qunatized Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if(hidden_dim is None):
          hidden_dim = input_dim

        self.language_MLP = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.language_MLP(x)

class State2QuantizedVecEncoder(nn.Module):
    def __init__(self, state_dim, q_hidden_dim,enc_dim):  #quantized_out_dim
        super().__init__()
        self.quantized_layer = nn.ModuleList(MLP(state_dim,q_hidden_dim,hidden_dim=384) for i in range(enc_dim))

    def forward(self, x):
        B,D = x.size()
        quantized_vec = []
        for i in self.quantized_layer:
            quantized_vec.append(i(x).unsqueeze(1))

        ###quantized_vec is now a len(enc_dim) list of (B,1,hidden_dim) neural_network
        return torch.cat(quantized_vec,dim=1) # (B,enc_dim,hidden_dim) output


class State_and_Language_Pair_Encoder(nn.Module):
    def __init__(self, state_dim, lang_dim, enc_dim, q_hidden_dim):
        super().__init__()
        self.language_encoder = MLP(lang_dim,q_hidden_dim)
        self.state_quantizer = State2QuantizedVecEncoder(state_dim,q_hidden_dim,enc_dim)

    def forward(self, state, lang_instructions):
        ##inputs state : (B,D), lang_structions : (B,L)

        lang_emb = self.language_encoder(lang_instructions) # (B,hidden_dim)
        quantized_vecs = self.state_quantizer(state)# (B,enc_dim,hidden_dim)

        cos_sim = torch.cosine_similarity(quantized_vecs,lang_emb.unsqueeze(1),dim=-1) # (B,enc_dim)
        return cos_sim