#First we import all relevant packages
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Then we load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
tokenizer = BertTokenizer.from_pretrained(model_name)
#We have a file train.csv containing the headlines and classification of 120000 titles
df_train = pd.read_csv('./train.csv')
df_train.drop(columns=["Description"], inplace=True) #We drop the description column as it is not of great use to us
df_train["Class Index"]=df_train["Class Index"]-1 #Preprocess class indexes to match a list from 0 to 3
num_samples = 5000  # Number of samples we want to take as 120000 are too many

# Creating a subset
df_train_subset = df_train.sample(n=num_samples, random_state=42)
#We initialize the tokenizer
train_inputs = tokenizer(
    df_train_subset["Title"].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
train_labels = torch.tensor(df_train_subset["Class Index"].tolist()) #Creating a torch tensor of class ids

train_dataset = TensorDataset(train_inputs["input_ids"], train_inputs["attention_mask"], train_labels) #using torch.utils to help with dataset creation
print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) #Process in batches
print(len(train_loader))
# Train the model
optimizer = AdamW(model.parameters(), lr=2e-5) #Using a predefined optimizer
for epoch in range(5):  # Adjust the number of epochs
    print(epoch)
    model.train()  #Forward propagation
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) #Initialize the model parameterd
        loss = outputs.loss 
        loss.backward() #Compute the loss
        optimizer.step() #Optimize

# Save the trained model
model.save_pretrained("news_classification_model")
