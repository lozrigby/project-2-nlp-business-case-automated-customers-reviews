from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# Load the model and tokenizer
model_path = r'C:\Users\kshno\Desktop\ACR_Transformer\ACRT_launch\Launch\fine_tuned_model'
tokenizer_path = r'C:\Users\kshno\Desktop\ACR_Transformer\ACRT_launch\Launch\fine_tokens_map'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Create Flask app
app = Flask(__name__)

# Define a function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

# Map numerical labels to sentiment
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    predicted_class_id = predict_sentiment(text)
    sentiment = label_map[predicted_class_id]
    
    return jsonify({'text': text, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
