# import necessary libraries
from pathlib import Path
import traceback
import argparse
import io
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import evaluate
import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import whisper
import yaml
from yaml.loader import SafeLoader
import pickle
from torcheval.metrics.functional import multiclass_f1_score

# Multitask Classifier BERT model
class Net(nn.Module):
  def __init__(self, label_encoder):
    super(Net, self).__init__()
    # Linear layer for each task
    self.linear1 = nn.ModuleDict()
    for task in label_encoder.keys():
      self.linear1[task] = torch.nn.Linear(768,label_encoder[task].classes_.shape[0])

  def forward(self, x, task):
    out = self.linear1[task](x)
    return out
class Dataset(torch.utils.data.Dataset):
    def __init__(self,encodings, labels, tasks):
      self.encodings = encodings
      self.labels = labels
      self.tasks = tasks

    def __len__(self):
      return len(self.labels['action'])

    def __getitem__(self, idx):
      items ={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
      for task in self.tasks:
        items[task] = torch.tensor(self.labels[task][idx])
      return items

class trainer:
  def __init__(self,config_path):
    # Taking inputs from config.yaml file
    config_file = open(config_path, "r") 
    cfg = yaml.load(config_file,Loader=SafeLoader)
    self.max_epoch      = cfg["Epochs"]
    self.learning_rate  = cfg["learning_rate"]
    self.batch_size     = cfg["Batch_size"]
    self.model_path     = cfg["output_dir"]
    self.train_csv_file = cfg["train_csv"]
    self.val_csv_file   = cfg["valid_csv"]
    self.label_encoder  = {}
    self.trainDataloader= []
    self.valDataloader  = []
    self.tokenizer      = []
    self.tasks          = ['action','object','location']
    self.model          = []
    self.criterion      = []
    self.optimizer      = []
    self.writer         = SummaryWriter()
    # BERT model for feature extraction
    self.BERT_model     = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')

  def data(self):
    # Loading csv files
    print("Loading data...")
    train_data = pd.read_csv(self.train_csv_file)
    train_data = train_data.dropna()
    val_data = pd.read_csv(self.val_csv_file)
    val_data = val_data.dropna()

    # Train dataset
    train_texts           = train_data["transcription"].values.tolist()
    train_action_labels   = train_data["action"].values.tolist()
    train_object_labels   = train_data["object"].values.tolist()
    train_location_labels = train_data["location"].values.tolist()

    # Val dataset
    val_texts             = val_data["transcription"].values.tolist()
    val_action_labels     = val_data["action"].values.tolist()
    val_object_labels     = val_data["object"].values.tolist()
    val_location_labels   = val_data["location"].values.tolist()

    # Train and validation label dictionaries
    train_labels = {"action":train_action_labels,"object":train_object_labels,"location":train_location_labels}
    val_labels = {"action":val_action_labels,"object":val_object_labels,"location":val_location_labels}

    print("Encoding labels...")
    for task in self.tasks:
      # Train label encoder
      label_encoder = preprocessing.LabelEncoder()
      label_encoder.fit(train_labels[task])
      train_labels[task] = label_encoder.transform(train_labels[task])
      # Encode validation labels
      val_labels[task] = label_encoder.transform(val_labels[task])
      self.label_encoder[task] = label_encoder

    #Saving label encoder dict for test.py
    pickle.dump(self.label_encoder, open('label_encoder.pkl', 'wb'))

    # load tokenizer from pretrained model
    print("Loading the tokenizer from pretrained model...")
    self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    # tokenize the text
    print("Tokenizing input...")
    train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = self.tokenizer(val_texts, truncation=True, padding=True)

    # Create a pytorch dataset
    print("Creating a pytorch dataset...")

    train_dataset= Dataset(train_encodings,train_labels,self.tasks)
    self.trainDataloader= DataLoader(train_dataset,batch_size=self.batch_size, shuffle=True,drop_last=True)
    val_dataset = Dataset(val_encodings, val_labels,self.tasks)
    self.valDataloader= DataLoader(val_dataset,batch_size=self.batch_size, shuffle=True,drop_last=True)


  def train_one_epoch(self):
      self.model.train()
      epoch_loss=0
      for i,data in enumerate(tqdm(self.trainDataloader)):
        src = {'input_ids'     :data['input_ids'],
              'token_type_ids' :data['token_type_ids'],
              'attention_mask' :data['attention_mask']}
        bert_feature = bertEmbedding(self.BERT_model, src)
        loss = 0
        for task in self.tasks:
          output = self.model(bert_feature,task)      # Model prediction from BERT feature
          tgt = data[task]                            # Target label
          loss += self.criterion(output.reshape(-1,self.label_encoder[task].classes_.shape[0]),tgt.reshape(-1))
        epoch_loss += loss
        self.optimizer.zero_grad()
        loss.backward()
        # Optimze
        self.optimizer.step()
      return epoch_loss/len(list(self.trainDataloader))

  def model_init(self):
    self.model    = Net(self.label_encoder)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

  def train(self):
    print("Training model...")
    for epoch in range(1,self.max_epoch+1):
      training_loss = self.train_one_epoch()
      torch.save({
              'epoch': epoch,
              'model_state_dict': self.model.state_dict(),
              'optimizer_state_dict': self.optimizer.state_dict(),
              'loss': training_loss,
              }, self.model_path + str(epoch)+".pt")
      validation_loss,f1_validation_score = self.validation(self.model_path + str(epoch)+".pt")
      self.writer.add_scalar("Loss/train", training_loss, epoch)
      self.writer.add_scalar("Loss/valid", validation_loss, epoch)
      self.writer.add_scalar("Action F1 score/valid",f1_validation_score['action'],epoch)
      self.writer.add_scalar("Object F1 score/valid",f1_validation_score['object'],epoch)
      self.writer.add_scalar("Location F1 score/valid",f1_validation_score['location'],epoch)
      print(f"Epoch: {epoch}, Train loss: {training_loss:.5f}, Validation loss: {validation_loss:.5f}, Validation F1 score: {f1_validation_score}")
      self.writer.flush()
    self.writer.close()

  def validation(self, model_path):
    model = Net(self.label_encoder)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    eval_loss =0
    for i,data in enumerate(tqdm(self.valDataloader)):
      src = {'input_ids'      :data['input_ids'],
              'token_type_ids':data['token_type_ids'],
              'attention_mask':data['attention_mask']}
      bert_feature = bertEmbedding(self.BERT_model,src)
      softmax = nn.Softmax()
      logits ={}
      f1_score = {}
      for task in self.tasks:
        # Model output from BERT feature
        output = model(bert_feature,task)
        # One hot vector for prediction
        tgt = data[task]
        # Loss computation
        eval_loss += self.criterion(output.reshape(-1,self.label_encoder[task].classes_.shape[0]),tgt.reshape(-1))
        #F1 score
        logits[task] = softmax(output)
        f1_score[task] = multiclass_f1_score(logits[task], tgt, num_classes=self.label_encoder[task].classes_.shape[0])
    return eval_loss/len(list(self.valDataloader)), f1_score

# To get the embeddings from BERT
def bertEmbedding(BERT_model,inputs):
  output = BERT_model(**inputs,output_hidden_states=True)
  bert_out = output.hidden_states[-1][:,0,:] #BERT embedding from last layer
  return bert_out
# Evaluate model on one custom command

def eval(model_path,Dataloader,label_encoder,BERT_model):
  model = Net(label_encoder)
  model.load_state_dict(torch.load(model_path)['model_state_dict'])
  tasks    = ['action','object','location']
  softmax  = nn.Softmax()
  f1_score = {}
  logits   = {}

  with torch.no_grad():
    for i,data in enumerate(tqdm(Dataloader)):
      src = {'input_ids'      :data['input_ids'],
              'token_type_ids':data['token_type_ids'],
              'attention_mask':data['attention_mask']}
      bert_feature = bertEmbedding(BERT_model,src)
      print("dataloader")
      for task in tasks:
        tgt = []
        output = model(bert_feature,task)
        tgt = data[task]
        logits[task] = softmax(output)

        #F1 score
        f1_score[task] = multiclass_f1_score(logits[task], tgt, num_classes=label_encoder[task].classes_.shape[0])

  return  f1_score

def custom_input(model_path,text,label_encoder,tokenizer,BERT_model):
  model = Net(label_encoder)
  model.load_state_dict(torch.load(model_path)['model_state_dict'])
  tasks    = ['action','object','location']
  softmax  = nn.Softmax()
  logits   = {}
  predicted_label = {}

  with torch.no_grad():
    # input_text = whisper_model.transcribe(input,language="en")["text"]
    input_token = tokenizer(text)
    input_token = {key: torch.tensor(val).unsqueeze(dim=0) for key,val in input_token.items()}
    bert_out = bertEmbedding(BERT_model,input_token)
    
    for task in tasks:
        logits[task] = softmax(model(bert_out,task))
        predicted_class_id = logits[task].argmax().item()
        predicted_label[task] = label_encoder[task].inverse_transform([predicted_class_id])

    return predicted_label