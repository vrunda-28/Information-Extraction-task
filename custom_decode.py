import MultiTaskBERT_
from MultiTaskBERT_ import *
import warnings
import torch
import argparse
import whisper

# Driver code
warnings.filterwarnings("ignore")
torch.manual_seed(20)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='/speech/vrunda/NER/vrunda/Model_checkpoints/checkpoint_epoch20.pt')
parser.add_argument('--speech_file', default='/speech/vrunda/NER/task_data/wavs/speakers/2BqVo8kVB2Skwgyb/ad5b6390-4479-11e9-a9a5-5dbec3b8816a.wav')
args = parser.parse_args()


asr_model     = whisper.load_model("small")
model_path    = args.model_path
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
tokenizer     = AutoTokenizer.from_pretrained('bert-base-cased')
BERT_model    = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
text          = asr_model.transcribe(args.speech_file,language="en")["text"]      

predicted_label = MultiTaskBERT_.custom_input(model_path=model_path,text=text,label_encoder=label_encoder,tokenizer=tokenizer,BERT_model=BERT_model)

print(f"Action: {predicted_label['action'][0]}, Object: {predicted_label['object'][0]},Location: {predicted_label['location'][0]}")
