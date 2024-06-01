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

class GoalConditioned_StateEncoder(nn.Module):
  def __init__(self, state_dim,output_dim,tau=0.3,hidden_dim=256,identity=True):
    super().__init__()
    self.identity = identity
    if(identity):
        self.encoder = nn.Identity()
    else:
      self.encoder = MLP(state_dim,output_dim,hidden_dim)
      self.decoder = MLP(output_dim,state_dim,hidden_dim)
      self.output_dim = output_dim
      self.tau = tau
  
  def forward(self,x,goal):
    ## x : (B,L,D)였던 벡터가 ((B*L,D)로 바뀌어서 들어옴)
    ## goal : (B,D)
    if(self.identity):
      x_enc = self.encoder(x)
      g_enc = self.encoder(goal)
      x_recon = None
  
    else:
      x_enc = self.encoder(x) # (B*L, enc_dim)
      g_enc = self.encoder(goal) # (B, enc_dim)
      x_recon = self.decoder(x_enc)
    
    return x_enc,g_enc,x_recon

  def loss(self,x,goal,L,gamma,unbound_mask):
    if(self.identity):
      return torch.tensor(0.)
    ##Contrastive Loss
    B,D = goal.shape
    ## x : (B,L,D)였던 벡터가 ((B*L,D)로 바뀌어서 들어옴)
    ## goal : (B,D)
    x_enc,goal_enc,x_recon = self(x,goal)
    x_enc = (x_enc).view(B,-1,self.output_dim) #(B,L,d_enc).  ###to maskout out of bound components #*unbound_mask
    goal_enc = goal_enc.view(B,-1,self.output_dim).detach() #(B,1,d_enc) #애는 이미 준비 되어있음
    goal_enc = (goal_enc/torch.norm(goal_enc,dim=-1,keepdim=True)).squeeze()#(B,d_enc)
    ##print(goal_enc.shape)
    t = torch.randint(L - 1, (B, 1, 1)).to(goal_enc.device)
    cont_s_t = torch.gather(x_enc, dim=1, index=t.repeat(1, 1, self.output_dim)).squeeze(1) ##x_enc 중에서 time-step t에 해당하는 애들 뽑아온거
    cont_s_t = cont_s_t/torch.norm(cont_s_t,dim=-1,keepdim=True) #(B,d_enc)
    #print(cont_s_t.shape)
    cos_sim = torch.einsum('ij,pj->ip',goal_enc,cont_s_t)

    goal_idx = torch.sum(unbound_mask.reshape(B,L),dim=-1).reshape(t.shape)

    log_p = F.log_softmax(-cos_sim/self.tau, dim=1)
    weights = (gamma ** (goal_idx-t)).view(-1)

    ##Intra-cluster loss
    intra_loss = -(weights * torch.diagonal(log_p)).mean()

    cont_loss = intra_loss #disp_loss

    #cont_mask = torch.triu(torch.ones((log_p.shape),dtype=int), diagonal=1).to(log_p.device)
    #disp_loss = -torch.mean(cont_mask*F.log_softmax(dist, dim=1))
    ##Inter-cluster loss
    #inter_mask = (torch.ones_like(log_p).to(cont_s.device)-torch.eye(B).to(cont_s.device)).detach()

    ##Recon Loss
    recon_loss = F.mse_loss(x.detach(),x_recon)
  
    recon_coef = 0.5

    loss = cont_loss + recon_coef*recon_loss

    return loss



#class State2QuantizedVecEncoder(nn.Module):
#    def __init__(self, state_dim, q_hidden_dim,enc_dim,linear=False):  #quantized_out_dim
#        super().__init__()
#        if(linear):
#          self.quantized_layer = nn.ModuleList(nn.Linear(state_dim,q_hidden_dim) for i in range(enc_dim))
#        else:
#          self.quantized_layer = nn.ModuleList(MLP(state_dim,q_hidden_dim,hidden_dim=128) for i in range(enc_dim))
#        
#
#    def forward(self, x):
#        B,D = x.size()
#        quantized_vec = []
#        for i in self.quantized_layer:
#            quantized_vec.append(i(x).unsqueeze(1))
#
#        ###quantized_vec is now a len(enc_dim) list of (B,1,hidden_dim) neural_network
#        return torch.cat(quantized_vec,dim=1) # (B,enc_dim,hidden_dim) output

#class State_and_Language_Pair_Encoder(nn.Module):
#    def __init__(self, state_dim,reduced_lang_dim,reduced_lang_embeddings,temp):
#        super().__init__()
#        self.N_task = 34
#        self.state_quantizer = State2QuantizedVecEncoder(state_dim,reduced_lang_dim,self.N_task)
#        self.state_dim = state_dim + reduced_lang_dim
#        self.reduced_lang_dim = reduced_lang_dim
#        self.register_buffer('reduced_lang_embeddings', reduced_lang_embeddings)
#        self.temp = temp
#
#    def forward(self, state, task_id, prior):
#        ##inputs state : (B,D), task_id : (B,)
#        B,D = state.shape
#        quantized_vecs = self.state_quantizer(state)# (B,N_task,reduced_lang_dim))
#        quantized_vecs = quantized_vecs/torch.norm(quantized_vecs,dim=-1,keepdim=True)
#        lang_emb = self.reduced_lang_embeddings[task_id] #(B,reduced_lang_dim)
#        cos_sim = torch.cosine_similarity(quantized_vecs,lang_emb.unsqueeze(1),dim=-1) # (B,N_task)
#        #prior_softmax = 
#        log_task_prob = torch.log_softmax(cos_sim/self.temp,dim=-1) # (B,N_task)
#        prior_log_task_prob = F.log_softmax(prior, dim=-1)
#        latent_vec = torch.einsum('ijk,ij->ik',quantized_vecs,torch.exp(prior_log_task_prob))
#        latent_target = quantized_vecs[torch.arange(B),task_id,:].detach()
#        rep_vec = torch.cat([state,latent_vec],dim=-1)
#
#        return rep_vec,log_task_prob,latent_target,latent_vec
#
#    def loss(self, state,task_id,prior,mask): # state : (B, D), prior : (B, 34)
#        B,D = state.shape
#        _, log_task_prob, latent_target,latent_vec = self(state,task_id,prior) # (BL, enc_dim)
#        #prior_softmax = F.log_softmax(prior, dim=-1)
#        prob_loss = F.cross_entropy(torch.exp(log_task_prob)*mask, prior*mask,reduction='mean')
#        cos_loss = torch.mean(torch.tensor(1.,requires_grad=False)-torch.einsum('ij,ij->i',latent_vec*mask,latent_target*mask))
#
#        ## Contrastive Loss btw 34 vecs
#        quantized_vecs = self.state_quantizer(state)
#        quantized_vecs = quantized_vecs/torch.norm(quantized_vecs,dim=-1,keepdim=True)
#        sim_mat = torch.einsum("ijk,ipk->ijp",quantized_vecs,quantized_vecs)
#        mask = torch.triu(torch.ones(sim_mat.shape,dtype=int), diagonal=1).to(sim_mat.device)
#        cont_cos = torch.mean(mask*(torch.tensor(1.,requires_grad=False)+sim_mat))*torch.tensor(2.)
#        
#        return prob_loss,cos_loss+cont_cos