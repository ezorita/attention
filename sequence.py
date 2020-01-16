import torch
import torch.nn as nn
from torch import LongTensor as lt
import numpy as np

class Sequence():
   def __init__(self, vocab, replace_symbols, fname, test_size=0.2, random_state=None, device=None):

      self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

      self.vocab = vocab
      self.replace_symbols = replace_symbols
      self.vocab_lut = list(self.vocab.keys())
      
      # Number of symbols
      self.n_symbols = len(self.vocab)

      # Preallocate aminoacid table
      to_idx = lambda x: [self.vocab[r] for r in x]

      # Read raw file
      with open(fname) as f:
         raw = f.read().split('\n')
         
      # Convert residues to indices
      seqs = [lt(to_idx(x)) for x in raw][:-1]
      del raw

      # Preserve sequence lengths
      self.lens = lt([len(s) for s in seqs])

      # Pad sequences, convert to tensor and send to device
      self.idseq = nn.utils.rnn.pad_sequence(seqs).transpose(0,1).to(self.device)
      del seqs

      self.seqlen = self.idseq.shape[1]
         
      # Train/test split
      idx = np.arange(self.idseq.shape[0])
      if random_state:
         np.random.seed(random_state)
      np.random.shuffle(idx)
      test_size = int(np.round(self.idseq.shape[0]*test_size))
      # Sequences
      self.test_seq  = self.idseq[idx[:test_size]]
      self.train_seq = self.idseq[idx[test_size:]]
      del self.idseq

      # Sizes
      self.test_len  = self.lens[idx[:test_size]]
      self.train_len = self.lens[idx[test_size:]]
      del self.lens

   def random_train_batch(self, batch_size):
      return self.train_seq[np.random.choice(self.train_seq.shape[0], size=batch_size, replace=False)]

   def random_test_batch(self, batch_size):
      return self.test_seq[np.random.choice(self.test_seq.shape[0], size=batch_size, replace=False)]

   def train_batches(self, batch_size):
      idx = np.arange(self.train_seq.shape[0])
      np.random.shuffle(idx)
      return [self.train_seq[i] for i in np.array_split(idx, self.train_seq.shape[0]//batch_size)]

   def test_batches(self, batch_size):
      idx = np.arange(self.test_seq.shape[0])
      np.random.shuffle(idx)
      return [self.test_seq[i] for i in np.array_split(idx, self.test_seq.shape[0].shape[0]//batch_size)]

   def SOP_MLM_batches(self, batch_size, mlm_mask_p=0.15, sop_flip_p=0.5, test=False):
      # Sequence Order Prediction & Masked Language Model pre-training
      # Applies the following to all sequences:
      # - mlm_mask_p elements of the sequence tokens are masked (masking: 0.8 replace by 'MASK', 0.1 replace at random, 0.1 stay the same)
      # - Sequence Order Prediction: The sequence is split in two and flipped with probability sop_flip_p
      # - Add CLS and SEP symbols
      # Return the modified batches, the targets and a tensor with segment lengths
      
      data = self.train_seq if test is False else self.test_seq
      lens = self.train_len if test is False else self.test_len

      # Select random target positions
      target_pos = torch.rand(size=data.shape).to(self.device)
      target_pos[data == 0] = 0
      target_idx = torch.where(target_pos > 1-mlm_mask_p)
      
      # Filter out 10% positions that will not be replaced (replaced by true value)
      replace_pos = np.random.uniform(0,1,len(target_idx[0]))
      replace_idx = (target_idx[0][replace_pos > 0.1], target_idx[1][replace_pos > 0.1])
      
      # Replace the remaining 90% by 80% [MASK] and 10% random residues
      replace_elem = np.concatenate((self.replace_symbols, np.array([self.vocab['MASK'],])))
      replace_prob = np.concatenate((np.ones(len(self.replace_symbols))*.1/.9/len(self.replace_symbols), np.array([.8/.9,])))
      replace_vals = lt(np.random.choice(replace_elem, size=len(replace_idx[0]), p=replace_prob)).to(self.device)

      # Generate MLM batch
      batch = data.clone()
      batch[replace_idx] = replace_vals

      # Generate targets (Ignore: -1)
      targets = -1*torch.ones_like(data)
      targets[target_idx] = data[target_idx]

      # Generate Sequence Order Position labels (whether swap was performed)
      sop_label = lt(np.random.randint(0,2,data.shape[-2])).to(self.device)

      # Swap SOP sequences add CLS/SEP symbols
      cls_sym = lt([self.vocab['CLS'],]).to(self.device)
      sep_sym = lt([self.vocab['SEP'],]).to(self.device)
      ign_sym = lt([-1,]).to(self.device)

      sop_batch = torch.stack(
         [torch.cat((cls_sym, s[(l//2):l], sep_sym, s[:(l//2)], sep_sym, s[l:]), -1)
          if r else
          torch.cat((cls_sym, s[:(l//2)], sep_sym, s[(l//2):l], sep_sym, s[l:]), -1)
          for s,l,r in zip(batch, lens, sop_label)
         ]
      )
      
      # Swap targets
      sop_targets = torch.stack(
         [torch.cat((r[None], s[(l//2):l], ign_sym, s[:(l//2)], ign_sym, s[l:]), -1)
          if r else
          torch.cat((r[None], s[:(l//2)], ign_sym, s[(l//2):l], ign_sym, s[l:]), -1)
          for s,l,r in zip(targets, lens, sop_label)
         ]
      )

      # Segment lengths
      seg_lens = torch.stack(
         [torch.LongTensor([l-l//2,l//2]) if r else torch.LongTensor([l//2,l-l//2]) for l,r in zip(lens,sop_label)]
      )
      
      # Shuffle data batches
      idx = np.arange(sop_batch.shape[0])
      np.random.shuffle(idx)
      
      # Return list of batches and targets
      batches = [sop_batch[i] for i in np.array_split(idx, sop_batch.shape[0]//batch_size)]
      targets = [sop_targets[i] for i in np.array_split(idx, sop_batch.shape[0]//batch_size)]
      seglens = [seg_lens[i] for i in np.array_split(idx, sop_batch.shape[0]//batch_size)]
      
      return batches, seglens, targets
   


class ProteinSeq(Sequence):
   def __init__(self, fname, test_size=0.2, random_state=None, device=None):
      # Vocabulary
      vocab = {'PAD': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6,
               'H': 7, 'I': 8, 'K': 9, 'L': 10,'M': 11,'N': 12,'P': 13,
               'Q': 14,'R': 15,'S': 16,'T': 17,'V': 18,'W': 19,'Y': 20,
               'CLS': 21, 'SEP': 22, 'MASK': 23
      }
      
      # Vocabulary indices that represent aminoacids (for random replacement)
      replace_symbols = torch.arange(1,21)

      super().__init__(vocab, replace_symbols, fname, test_size, random_state, device)

class DNASeq(Sequence):
   def __init__(self, fname, test_size=0.2, random_state=None, device=None):
      # Vocabulary
      vocab = {'PAD': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4,
               'CLS': 5, 'SEP': 6, 'MASK': 7
      }
      
      # Vocabulary indices that represent aminoacids (for random replacement)
      replace_symbols = torch.arange(1,5)

      super().__init__(vocab, replace_symbols, fname, test_size, random_state, device)




# Applies random deletions, returns new sequence and attention mask
def random_dels(seq, n_del_func):
   # Create list of sequences with deletions
   dseqs = [x[np.sort(np.random.choice(seq.shape[1],size=seq.shape[1] - n_del_func(),replace=False))] for x in seq]
   # Extend maximum length to original
   dseqs.append(torch.zeros(seq.shape[1], dtype=seq.dtype, device=seq.device))
   # Pad again to full length
   return nn.utils.rnn.pad_sequence(dseqs).transpose(0,1)[:-1]

   
# Converts a index sequence to one-hot
def to_onehot(seq, n_symbols, dtype=None):
   seq_oh = torch.zeros((seq.shape[0]*seq.shape[1], n_symbols), dtype=seq.dtype if not dtype else dtype, device=seq.device)
   seq_oh.scatter_(1, seq.view(-1,1), 1)
   return seq_oh.view(seq.shape[0], seq.shape[1], -1)
