import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pdb, traceback, sys

'''
Aux functions
'''
def _init_matrix(*dims):
   m = torch.Tensor(*dims)
   #from torch.nn.Linear source
   init.kaiming_uniform_(m, a=np.sqrt(5))
   return m

class GELU(nn.Module):
    def forward(self, x):
        return x*torch.sigmoid(1.702*x)


'''
Embeddings
'''

def matrixR(L, d_model):
   inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
   sinusoid_inp = torch.ger(torch.arange(L-1,dtype=torch.float32), inv_freq)
   return torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1).transpose(-2,-1)


''' 
Attention building blocks
'''

class MultiHeadedAttention(nn.Module):
   
   def __init__(self, d_model, h, mask=None, dropout=0.1):
      super(MultiHeadedAttention, self).__init__()
      
      self.h  = h
      self.d_model = d_model
      
      self.Wq = nn.Linear(d_model, d_model)
      self.Wk = nn.Linear(d_model, d_model)
      self.Wv = nn.Linear(d_model, d_model)
      self.Wo = nn.Linear(d_model, d_model)
      self.do = nn.Dropout(p = dropout) if dropout > 0 else None
      self.ln = nn.LayerNorm(d_model)

      self.mask = mask
      
   def forward(self, q, v, mask=None):
      ''' 
         q, k, v ~ (Batch, L, d_model)
      Qh, Kh, Vh ~ (Batch, h, d_model/h, L)
              qk ~ (Batch, h, L, L)
              Oh ~ (Batch, h, d_model/h, L)
               O ~ (Batch, L, d_model)
      '''
      Qh = self.Wq(q).view(q.shape[0], q.shape[1], self.h, -1).transpose(1,2)
      Kh = self.Wk(q).view(q.shape[0], q.shape[1], self.h, -1).transpose(1,2)
      Vh = self.Wk(v).view(q.shape[0], q.shape[1], self.h, -1).transpose(1,2)

      # Scaled Dot Product Attention in h blocks QKh_b = dot(Qh_b^T, Kh_b) for all h (head) and b (batch)
      qk = torch.einsum('ijlk,ijmk->ijlm', (Qh, Kh)) / np.sqrt(self.d_model)

      # Reset mask values to -Inf (softmax prob ~ 0)
      if mask is not None:
         qk = qk.masked_fill(mask == 0, float('-inf'))
      elif self.mask is not None:
         qk = qk.masked_fill(self.mask == 0, float('-inf'))
             
      # Softmax on sample dimension (not d_model)
      p_attn = F.softmax(qk, dim=-1)

      # Dropout to attention probabilities
      if self.do is not None:
         p_attn = self.do(p_attn)

      # Apply attention to Vh -> Oh = dot(p_attn, Vh^T)
      Oh = torch.einsum('ijkl,ijlm->ijkm', (p_attn, Vh))

      # Concatenate attention output
      O = Oh.contiguous().view(q.shape)

      # Layer norm and residual connection
      return self.ln(q + self.Wo(O))


class RelativeMultiHeadedAttention(nn.Module):

   def __init__(self, L, d_model, h, mask=None, dropout=0.1):
      super(RelativeMultiHeadedAttention, self).__init__()
      assert d_model % h == 0
      self.L = L
      self.h = h
      self.d_model = d_model
      self.mask = mask
      self.R = nn.Parameter(matrixR(L+1, d_model))

      # Linear transformations of embeddings
      self.Wq = nn.Parameter(_init_matrix(d_model, d_model))
      self.Wv = nn.Parameter(_init_matrix(d_model, d_model))
      self.Wke = nn.Parameter(_init_matrix(d_model, d_model))
      self.Wkr = nn.Parameter(_init_matrix(d_model, d_model))

      # Position and content biases
      self.cb = nn.Parameter(torch.zeros(d_model)) # Content bias
      self.pb = nn.Parameter(torch.zeros(d_model)) # Position bias

      # Output layers
      self.do = nn.Dropout(p = dropout) if dropout > 0 else None
      self.Wo = nn.Linear(d_model, d_model)
      self.ln = nn.LayerNorm(d_model)

   def _shift_b(self, b):
      # Inspired by https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py#L194
      z = torch.zeros(b.shape[0], b.shape[1], b.shape[2], 1, device=b.device)
      b_pad = torch.cat((z,b), 3).view(b.shape[0], b.shape[1], b.shape[3]+1, b.shape[2])
      return b_pad[:,:,1:].view_as(b)[:,:,:,:b.shape[2]]


   def forward(self, E, Ev, mask=None):
      '''
         Ev, E  ~  (Batch, L, d_model)
             R  ~  (1, d_model, 2L-1)
            Wq  ~  (h, d_model, d_model/h)
    Wv,Wke,Wkr  ~  (h, d_model/h, d_model)
        cb, pb  ~  (1, h, 1, d_model/h)
             q  ~  (Batch, h, L, d_model/h)
          v, k  ~  (Batch, h, d_model/h, L)
             Q  ~  (1, h, d_model/h, 2L-1)
             b  ~  (Batch, h, L, 2L-1)
       A, D, B  ~  (Batch, h, L, L)
            Oh  ~  (Batch, h, d_model/h, L)
             O  ~  (Batch, L, d_model)
      '''
      q = torch.matmul(E, self.Wq).view(E.shape[0], E.shape[1], self.h, -1).transpose(1,2)
      k = torch.matmul(E, self.Wke).view(E.shape[0], E.shape[1], self.h, -1).transpose(1,2)
      v = torch.matmul(Ev, self.Wv).view(E.shape[0], E.shape[1], self.h, -1).transpose(1,2)
      # Not the query, the matrix Q page 12 of Transformer-XL.
      Q = torch.matmul(self.Wkr, self.R).view((1,self.h,-1,E.shape[1])).repeat(E.shape[0],1,1,1)
      B = torch.matmul(q,Q)
      D = torch.matmul(self.pb.view(1,self.h,1,-1).repeat(E.shape[0],1,E.shape[1],1),Q)

      # Attention matrix
      A_a = torch.matmul(q,k.transpose(-2,-1))
      A_b = self._shift_b(B)

      # Compute bias vector and replicate to row dimension L
      A_c = torch.matmul(self.cb.view(1,self.h,1,-1).repeat(E.shape[0],1,E.shape[1],1),k.transpose(-2,-1))
      A_d = self._shift_b(D)

      # Attention matrix
      A = A_a + A_b + A_c + A_d

      if mask is not None:
         A = A.masked_fill(mask == 0, float('-inf'))
      elif self.mask is not None:
         A = A.masked_fill(self.mask == 0, float('-inf'))

      # Attention softmax
      p_attn = F.softmax(A, dim=-1)

      # Dropout to attention probabilities
      if self.do is not None:
         p_attn = self.do(p_attn)

      # Apply attention to v
      Oh = torch.einsum('ijkl,ijlm->ijkm', (p_attn, v))

      # Concatenate attention output
      O = Oh.contiguous().view(E.shape)

      # Layer norm and residual connection
      return self.ln(Ev + self.Wo(O))


   
class FeedForwardNet(nn.Module):
   def __init__(self, d_model, d_ffn):
      super(FeedForwardNet, self).__init__()
      
      self.ff = nn.Sequential(nn.Linear(d_model, d_ffn),
                              GELU(),#nn.ReLU(),
                              nn.Linear(d_ffn, d_model))
      self.ln = nn.LayerNorm(d_model)

   def forward(self, x):
      return self.ln(x + self.ff(x))

   
''' 
Encoder and Decoder Blocks
'''

class EncoderBlock(nn.Module):
   def __init__(self, d_model, d_ffn, h, dropout=0.1):
      super(EncoderBlock, self).__init__()
      
      self.attn      = MultiHeadedAttention(d_model, h, dropout=dropout)
      self.ffn       = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, mask):
      return self.ffn(self.attn(x,x,mask))


class RelativeEncoderBlock(nn.Module):
   def __init__(self, L, d_model, d_ffn, h, dropout=0.1):
      super(RelativeEncoderBlock, self).__init__()
      
      self.attn      = RelativeMultiHeadedAttention(L, d_model, h, dropout=dropout)
      self.ffn       = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, mask):
      return self.ffn(self.attn(x,x,mask))

   
class DecoderBlock(nn.Module):
   def __init__(self, d_model, d_ffn, h, mask, dropout=0.1):
      super(DecoderBlock, self).__init__()
      
      self.attn0      = MultiHeadedAttention(d_model, h, mask=mask, dropout=dropout)
      self.attn1      = MultiHeadedAttention(d_model, h, mask=None, dropout=dropout)
      self.ffn        = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, vk):
      y  = self.attn0(x,x)
      z  = self.attn1(y,vk)
      return self.ffn(z)

class RelativeDecoderBlock(nn.Module):
   def __init__(self, L, d_model, d_ffn, h, mask, dropout=0.1):
      super(RelativeDecoderBlock, self).__init__()
      
      self.attn0      = RelativeMultiHeadedAttention(L, d_model, h, mask=mask, dropout=dropout)
      self.attn1      = RelativeMultiHeadedAttention(L, d_model, h, mask=None, dropout=dropout)
      self.ffn        = FeedForwardNet(d_model, d_ffn)
      
   def forward(self, x, vk):
      y  = self.attn0(x,x)
      z  = self.attn1(y,vk)
      return self.ffn(z)


'''
ALBERT
'''

class Albert(nn.Module):
   def __init__(self, N, E, H, h, d_ffn, L, n_word, dropout=0.1, device=None):
      super(Albert,self).__init__()

      # Model parameters
      self.N = N # Number of encoders
      self.E = E # Embedding size
      self.H = H # Attention hidden size
      self.h = h # MultiAttention head count
      self.d_ffn = d_ffn # FeedForward Network hidden size
      # Text parameters
      self.L = L # Sequence Length
      self.n_word = n_word # Vocabulary size
      # Model device
      self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

      # Text Embedding Transformations
      self.tok_emb = nn.Embedding(n_word, E).to(self.device)
      self.out_b   = nn.Parameter(torch.zeros(n_word).to(self.device))
      self.to_hid = nn.Linear(E,H).to(self.device)
      self.to_emb = nn.Linear(H,E).to(self.device)

      # Sequence Embeddings
      self.seq_emb = nn.Embedding(2, E).to(self.device)

      # Self-attention layers
      self.attn    = [RelativeEncoderBlock(L, H, d_ffn, h, dropout=dropout).to(self.device) for _ in range(N)]

      # CLS Pooling
      self.pooler = nn.Sequential(nn.Linear(H, H), nn.Tanh()).to(self.device)

      # NextSequencePrediction head
      self.osp_head = nn.Linear(H, 2).to(self.device) # No activation, use with nn.CrossEntropyLoss() in optimization

      # MaskedLanguageModel head
      self.mlm_head = nn.Sequential(
         nn.Linear(H,H),
         GELU(),#nn.ReLU(),
         nn.LayerNorm(H)
      ).to(self.device)
      

   def tied_H_to_E(self, h):
      # Tied output embeddings with untied output biases
      return torch.einsum('ixy, jzy -> ixz', self.to_emb(h), self.tok_emb.weight[None,:,:]) + self.out_b

   
   def get_seg_embeds(self, seglens):
      # Returns sequence embeddings, sequence changes at half sequence length (including 'SEP')
      seqA = self.seq_emb(torch.LongTensor([0]).to(self.device))
      seqB = self.seq_emb(torch.LongTensor([1]).to(self.device))
      zero = torch.zeros(1,1,self.E).to(self.device)
      none = torch.Tensor([]).to(self.device)

      
      seg_embeds = [torch.cat((zero,
                               seqA.repeat((1,la+1,1)),
                               seqB.repeat((1,lb+1,1)),
                               zero.repeat(1,self.L-la-lb-3,1) if self.L-la-lb-3 > 0 else none)
                              , -2) for la, lb in seglens]
      
      return torch.stack(seg_embeds)[:,0]

   
   def forward(self, batch, seglens, mode='pretrain', mask=None):
      # Apply word and sequence embeddings
      token_embeddings = self.tok_emb(batch)
      segmt_embeddings = self.get_seg_embeds(seglens)
      x = self.to_hid(token_embeddings + segmt_embeddings)

      # Attention layers
      for att in self.attn:
         x = att(x, mask)

      # Upper head
      if mode == 'raw':
         # No head, return attention output
         return x
      elif mode == 'pretrain':
         # Return output of MLM and OSP heads
         C = self.osp_head(x[:,0,:])
         T = self.tied_H_to_E(self.mlm_head(x[:,1:,:]))
         return C, T
      else:
         return None

      
   def pretrain(self, seqs, epochs, batch_size, lr, lr_warmup=10000):

      # Optimizer (warmup and linear decay or LR)
      opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.01)
      lr_decay = torch.linspace(lr,0,steps=epochs*int(np.ceil(seqs.train_seq.shape[0]/batch_size))-lr_warmup+1)

      # Logs
      train_sop_loss = np.zeros(epochs)
      train_mlm_loss = np.zeros(epochs)
      test_sop_loss  = np.zeros(epochs)
      test_mlm_loss  = np.zeros(epochs)

      # Loss functions
      sop_loss_func = nn.CrossEntropyLoss(reduction='sum')
      mlm_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

      # Test size
      test_batch_ratio = 10
      
      n = 0
      for e in torch.arange(epochs):
         b = 0; mem = 0

         # Train epoch
         train_x, train_l, train_t = seqs.SOP_MLM_batches(batch_size)
         for batch, seglens, targets in zip(train_x, train_l, train_t):
            b += 1; n += 1
            # Update LR
            opt.param_groups[0]['lr'] = lr*n/lr_warmup if n < lr_warmup else lr_decay[n-lr_warmup]
            
            # Compute Albert SOP and MLM outputs
            cls, tokens = self(batch, seglens, mode='pretrain')

            # SOP Loss
            sop_loss = sop_loss_func(cls, targets[:,0])/batch.shape[0]
            
            # MLM Loss
            mlm_loss = mlm_loss_func(tokens.view(-1,self.n_word), targets[:,1:].reshape(-1))/(targets[:,1:] >= 0).sum()

            # Total loss
            loss = sop_loss + mlm_loss
            mem = max(mem, torch.cuda.memory_allocated()/1024/1024)
            
            # Update parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_sop_loss[e] += float(sop_loss.detach().to('cpu'))/len(train_x)
            train_mlm_loss[e] += float(mlm_loss.detach().to('cpu'))/len(train_x)

         # End of epoch, compute test loss on a single batch
         test_x, test_l, test_t  = seqs.SOP_MLM_batches(test_batch_ratio*batch_size, test=True)
         with torch.no_grad():
            # Get random test batch
            idx = np.random.choice(len(test_x))
            batch, seglens, targets = test_x[idx], test_l[idx], test_t[idx]

            # Compute Albert SOP and MLM outputs
            cls, tokens = self(batch, seglens, mode='pretrain')
            
            # SOP Loss
            sop_loss = sop_loss_func(cls, targets[:,0])/batch.shape[0]
            
            # MLM Loss
            mlm_loss = mlm_loss_func(tokens.view(-1,self.n_word), targets[:,1:].reshape(-1))/(targets[:,1:] >= 0).sum()
            
            # Total loss
            test_sop_loss[e] = sop_loss
            test_mlm_loss[e] = mlm_loss

         # Verbose epoch
         print("[epoch {}] train_loss=[SOP:{:.3f} MLM:{:.3f}] test_loss=[SOP:{:.3f} MLM:{:.3f}] memory={:.2f}MB".format(e+1, 100*train_sop_loss[e], 100*train_mlm_loss[e], 100*test_sop_loss[e], 100*test_mlm_loss[e], mem))

         # Save model
         if (e+1)%10 == 0:
            torch.save(self.state_dict(), 'saved_models/albert_statedict_epoch{}.torch'.format(e))

      return train_sop_loss, train_mlm_loss, test_sop_loss, test_mlm_loss
