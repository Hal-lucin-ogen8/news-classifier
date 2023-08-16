#Import all necessary libraries
import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained news classification model
model = BertForSequenceClassification.from_pretrained('./news_classification_model')  # Loading trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def classify_news(headline): #Define a function to classify the news into 4 categories- World Sports Business Science/Technology
    inputs = tokenizer(headline, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()
    categories = ["World", "Sports", "Business", "Science/Technology"]
    category = categories[predicted_label] #Converting predicted label to a string
    return f'{category}'

iface = gr.Interface(
    fn=classify_news,
    inputs=gr.inputs.Textbox(lines=1, label="Enter a news headline"), #Get input
    outputs=gr.outputs.Textbox(label="Predicted Category"), #Give output
    title="News Category Prediction Model",
    description='This is a simple news headline classifer based on Bert and trained on the AG News database. It classifies headlines into one of four types- World, Sports, Business, Science/Technology. Give it a try!'
)

if __name__ == "__main__":
    iface.launch()
