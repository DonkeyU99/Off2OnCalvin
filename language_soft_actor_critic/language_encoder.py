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

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class State2QuantizedVecEncoder(nn.Module):
    def __init__(self, state_dim, q_hidden_dim,enc_dim,linear=False):  #quantized_out_dim
        super().__init__()
        if(linear):
          self.quantized_layer = nn.ModuleList(nn.Linear(state_dim,q_hidden_dim) for i in range(enc_dim))
        else:
          self.quantized_layer = nn.ModuleList(MLP(state_dim,q_hidden_dim,hidden_dim=128) for i in range(enc_dim))
        

    def forward(self, x):
        B,D = x.size()
        quantized_vec = []
        for i in self.quantized_layer:
            quantized_vec.append(i(x).unsqueeze(1))

        ###quantized_vec is now a len(enc_dim) list of (B,1,hidden_dim) neural_network
        return torch.cat(quantized_vec,dim=1) # (B,enc_dim,hidden_dim) output

class State_and_Language_Pair_Encoder(nn.Module):
    def __init__(self, state_dim,reduced_lang_dim,reduced_lang_embeddings,temp):
        super().__init__()
        self.N_task = 34
        self.state_quantizer = State2QuantizedVecEncoder(state_dim,reduced_lang_dim,self.N_task)
        self.state_dim = state_dim + reduced_lang_dim
        self.reduced_lang_dim = reduced_lang_dim
        self.register_buffer('reduced_lang_embeddings', reduced_lang_embeddings)
        self.temp = temp

    def forward(self, state, task_id, prior):
        ##inputs state : (B,D), task_id : (B,)
        B,D = state.shape
        quantized_vecs = self.state_quantizer(state)# (B,N_task,reduced_lang_dim))
        quantized_vecs = quantized_vecs/torch.norm(quantized_vecs,dim=-1,keepdim=True)
        lang_emb = self.reduced_lang_embeddings[task_id] #(B,reduced_lang_dim)
        cos_sim = torch.cosine_similarity(quantized_vecs,lang_emb.unsqueeze(1),dim=-1) # (B,N_task)
        #prior_softmax = 
        log_task_prob = torch.log_softmax(cos_sim/self.temp,dim=-1) # (B,N_task)
        prior_log_task_prob = F.log_softmax(prior, dim=-1)
        latent_vec = torch.einsum('ijk,ij->ik',quantized_vecs,torch.exp(prior_log_task_prob))
        latent_target = quantized_vecs[torch.arange(B),task_id,:].detach()
        rep_vec = torch.cat([state,latent_vec],dim=-1)

        return rep_vec,log_task_prob,latent_target,latent_vec

    def loss(self, state,task_id,prior,mask): # state : (B, D), prior : (B, 34)
        B,D = state.shape
        _, log_task_prob, latent_target,latent_vec = self(state,task_id,prior) # (BL, enc_dim)
        #prior_softmax = F.log_softmax(prior, dim=-1)
        prob_loss = F.cross_entropy(torch.exp(log_task_prob)*mask, prior*mask,reduction='mean')
        cos_loss = torch.mean(torch.tensor(1.,requires_grad=False)-torch.einsum('ij,ij->i',latent_vec*mask,latent_target*mask))

        ## Contrastive Loss btw 34 vecs
        quantized_vecs = self.state_quantizer(state)
        quantized_vecs = quantized_vecs/torch.norm(quantized_vecs,dim=-1,keepdim=True)
        sim_mat = torch.einsum("ijk,ipk->ijp",quantized_vecs,quantized_vecs)
        mask = torch.triu(torch.ones(sim_mat.shape,dtype=int), diagonal=1).to(sim_mat.device)
        cont_cos = torch.mean(mask*(torch.tensor(1.,requires_grad=False)+sim_mat))*torch.tensor(2.)
        
        return prob_loss,cos_loss+cont_cos