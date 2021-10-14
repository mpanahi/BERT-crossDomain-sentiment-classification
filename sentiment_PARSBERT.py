import pandas as pd

dataset_dataframe=pd.read_csv("cmts_final.csv",usecols=["text","label"])
train_test_ratio=0.8
train_test_sep=int(train_test_ratio*len(dataset_dataframe))
print(train_test_sep)

from  transformers import AutoTokenizer
import torch
train_dataframe=dataset_dataframe[0:train_test_sep]
print(len(train_dataframe))
test_dataframe=dataset_dataframe[train_test_sep:len(dataset_dataframe)]
tokenizer=AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
torch.save(tokenizer, "drive/MyDrive/tokenizer_bert")

logits_ls = []
asl_neg_test = []
asl_pos_test = []
labels_ls = []
device = "cuda"
from transformers import AutoModelForSequenceClassification

model=AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-zwnj-base").to(device)

from sklearn.metrics import classification_report
import numpy as np



# print(np.argmax(logits,axis=-1)[0])

import torch
from torch.utils.data import Dataset, DataLoader

MAX_LEN=32
device="cuda"

class SentimentDataset(Dataset):
  def __init__(self, dataframe):
    self.dataframe=dataframe
  def __len__(self):
    return len(self.dataframe)
  def __getitem__(self,idx):
    df=self.dataframe.iloc[idx]
    text=[df["text"]]
    label=[df["label"]]
    data_t=tokenizer(text,max_length=MAX_LEN,return_tensors="pt",padding="max_length",truncation=True)
    label_t=torch.LongTensor(label)
    return {"input_ids":data_t["input_ids"].squeeze().to(device),"label":label_t.squeeze().to(device),}


train_dataset=SentimentDataset(train_dataframe)

BATCH_SIZE=16
train_dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE)
print((train_dataloader.dataset.dataframe)["label"])

from transformers import AutoModelForSequenceClassification

model=AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-zwnj-base").to(device)


import transformers
from sklearn.metrics import classification_report
import numpy as np
optimizer=transformers.AdamW(model.parameters(),lr=1e-5)

EPOCHS = 3
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
print((train_dataloader.dataset.dataframe)["label"])
# EPOCHS=1
for epoch in range(EPOCHS):
    print("\n****\n epoch=", epoch)
    i = 0
    logits_list = []
    labels_list = []
    for batch in train_dataloader:

        i += 1
        optimizer.zero_grad()

        output_model = model(input_ids=batch["input_ids"], labels=batch["label"])
        loss = output_model.loss
        logits = output_model.logits
        logits_list.append(logits.cpu().detach().numpy())
        labels_list.append(batch["label"].cpu().detach().numpy())
        # print(labels_list)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(loss.item())
    logits_list = np.concatenate(logits_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    logits_list = np.argmax(logits_list, axis=1)
    print(classification_report(labels_list, logits_list))

torch.save(model, "mdl")





