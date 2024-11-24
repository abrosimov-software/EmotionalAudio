import os
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import HubertForSequenceClassification, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, base_dir, max_length=None):
        self.file_list = file_list
        self.base_dir = base_dir
        self.max_length = max_length  

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        emotion = int(file_name.split("-")[2])
        emotion = emotion - 1  

        input_values = torch.load(os.path.join(self.base_dir, "audiofiles", file_name))
        if self.max_length is None:
            self.max_length = input_values.size(0) // 2
        input_values = input_values[:self.max_length]

        label_tensor = torch.tensor(emotion, dtype=torch.long)

        return {"input_values": input_values, "labels": label_tensor}

def collate_fn(batch):
    input_values = [item["input_values"].float() for item in batch]
    max_length = max([x.size(0) for x in input_values]) 
    input_values = [torch.nn.functional.pad(x, (0, max_length - x.size(0))) for x in input_values]
    input_values = torch.stack(input_values) 
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_values": input_values, "labels": labels}



DATA_DIR = "data/training_data"
audio_dir = os.path.join(DATA_DIR, "audiofiles")
file_list = os.listdir(audio_dir)

emotion_groups = {}
for file_name in file_list:
    emotion = int(file_name.split("-")[2])
    emotion_groups.setdefault(emotion, []).append(file_name)

train_files = []
for emotion, files in emotion_groups.items():
    train_files.extend(files) 

# Prepare training dataset
train_files = train_files[:50] 
train_dataset = AudioDataset(train_files, DATA_DIR)

# Training arguments (shared)
training_args = TrainingArguments(
    max_grad_norm=1.0,  
    output_dir="./results",
    save_strategy="epoch", 
    logging_strategy="epoch",  
    logging_dir="./logs",
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8,  
    num_train_epochs=3,
    learning_rate=1e-5, 
    weight_decay=0.01,
    seed=42,
    full_determinism=True,
    disable_tqdm=True,  
    no_cuda=True,  # Use CPU
    report_to=[],  # Disable integrations
)


MODEL_NAME_DISTILHUBERT = "pollner/distilhubert-finetuned-ravdess"
model_distilhubert = HubertForSequenceClassification.from_pretrained(MODEL_NAME_DISTILHUBERT, num_labels=8)
model_distilhubert.to("cpu") 

print("Training DistilHuBERT...")
trainer_distilhubert = Trainer(
    model=model_distilhubert,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)
try:
    trainer_distilhubert.train()
except RuntimeError as e:
    print(f"Training failed for DistilHuBERT: {e}")

MODEL_DIR_DISTILHUBERT = "models/distilhubert"
os.makedirs(MODEL_DIR_DISTILHUBERT, exist_ok=True)
model_distilhubert.save_pretrained(MODEL_DIR_DISTILHUBERT)
print("DistilHuBERT model fine-tuned and saved successfully.")


MODEL_NAME_WAV2VEC2 = "firdho26/wav2vec2-large-xlsr-53-english-finetuned-ravdess"
model_wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME_WAV2VEC2, num_labels=8)
model_wav2vec2.to("cpu")  

print("Training Wav2Vec2...")
trainer_wav2vec2 = Trainer(
    model=model_wav2vec2,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)
try:
    trainer_wav2vec2.train()
except RuntimeError as e:
    print(f"Training failed for Wav2Vec2: {e}")

MODEL_DIR_WAV2VEC2 = "models/wav2vec2"
os.makedirs(MODEL_DIR_WAV2VEC2, exist_ok=True)
model_wav2vec2.save_pretrained(MODEL_DIR_WAV2VEC2)
print("Wav2Vec2 model fine-tuned and saved successfully.")


