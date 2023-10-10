import MultiTaskBERT_
import warnings
import torch

# Driver code
warnings.filterwarnings("ignore")
torch.manual_seed(20)
multi_bert_model_trainer = MultiTaskBERT_.trainer("config_.yaml")
multi_bert_model_trainer.data()
multi_bert_model_trainer.model_init()
multi_bert_model_trainer.train()