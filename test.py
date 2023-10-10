import MultiTaskBERT_
from MultiTaskBERT_ import *
import warnings
import torch
import argparse
import whisper
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Driver code
warnings.filterwarnings("ignore")
torch.manual_seed(20)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='/speech/vrunda/NER/vrunda/Model_checkpoints/checkpoint_epoch20.pt')
parser.add_argument('--test_csv', default='/speech/vrunda/NER/task_data/test.csv')
args = parser.parse_args()

asr_model     = whisper.load_model("small")
model_path    = args.model_path
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
tokenizer     = AutoTokenizer.from_pretrained('bert-base-cased')
BERT_model    = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
tasks         = ["action","object","location"]
test_texts    = []

print("Loading data...")
test_data = pd.read_csv(args.test_csv)
test_data = test_data.dropna()

# Test dataset
test_audio_paths     = test_data["path"].values.tolist()
test_action_labels   = test_data["action"].values.tolist()
test_object_labels   = test_data["object"].values.tolist()
test_location_labels = test_data["location"].values.tolist()
for path in tqdm(test_audio_paths):    
    test_texts.append(asr_model.transcribe(path,language="en")["text"])
print(test_texts)

test_labels = {"action": test_action_labels,"object": test_object_labels,"location": test_location_labels}
for task in tasks:
    test_labels[task] = label_encoder[task].transform(test_labels[task])

# Dataloader preparation
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset= MultiTaskBERT_.Dataset(test_encodings,test_labels,tasks)
testDataloader= DataLoader(test_dataset,batch_size=len(test_texts))

# Evaluation
f1_score = MultiTaskBERT_.eval(model_path=model_path,Dataloader=testDataloader,label_encoder=label_encoder,BERT_model=BERT_model)

print("Eval_F1_score:",f1_score)