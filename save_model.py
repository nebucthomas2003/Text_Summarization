import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize your model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Save the model's configuration
model.config.save_pretrained('./saved_model')

# Save the model's state_dict (weights)
torch.save(model.state_dict(), './saved_model/pytorch_model.bin')

# Save the tokenizer's vocabulary and configuration
tokenizer.save_pretrained('./saved_model')
