from transformers import BartTokenizer, BartForConditionalGeneration

# Define the model name
model_name = "facebook/bart-large-cnn"

# Download the tokenizer and model
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Save the tokenizer and model locally
save_directory = './saved_model'
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
