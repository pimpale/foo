import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./finetuned_model/')
model = GPT2LMHeadModel.from_pretrained('./finetuned_model/')

# Set the model to evaluation mode
model.eval()

text = input("Enter some text: ")

encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)