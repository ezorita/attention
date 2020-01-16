import sys
import matplotlib.pyplot as plt

# Local modules
import sequence
import attention
from lookahead import Lookahead

# Avoid plot crash when no X-server is available
plt.switch_backend('agg')


if __name__ == "__main__":

   if sys.argv[1] == 'train':
      seqs = sequence.ProteinSeq(sys.argv[2], test_size=0.2, device='cuda')

      albert = attention.Albert(
         N = 6,
         E = 64,
         H = 256,
         h = 8,
         d_ffn = 1024,
         L = seqs.seqlen + 3,
         n_word = seqs.n_symbols
      )

      epochs = 300
      batch_size = 64
      lr = 0.0001
      train_sop_loss, train_mlm_loss, test_sop_loss, test_mlm_loss = albert.pretrain(seqs, epochs=epochs, batch_size=batch_size, lr=lr)
      torch.save(albert.state_dict(), 'albert_model_statedict.torch')
      
      plt.figure()
      plt.plot(train_sop_loss)
      plt.plot(test_sop_loss)
      plt.legend(["Train", "Test"])
      plt.xlabel("epoch")
      plt.ylabel("SOP loss")
      plt.savefig('sop_loss.png')

      plt.figure()
      plt.plot(train_mlm_loss)
      plt.plot(test_mlm_loss)
      plt.legend(["Train", "Test"])
      plt.xlabel("epoch")
      plt.ylabel("MLM loss")
      plt.savefig('mlm_loss.png')


   elif sys.argv[1] == 'eval':
      raise Exception("Not yet implemented")
