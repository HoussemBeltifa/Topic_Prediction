import torch
import torch.nn.functional as F
import nltk
import re
import pickle

# load vocab and vocab_label
with open('vocab_data.pkl', 'rb') as fp:
    vocab = pickle.load(fp)
with open('label_vocab_data.pkl', 'rb') as fp:
    label_vocab = pickle.load(fp)


nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
# Get the list of stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from tokenized data
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Function to remove special characters from a list of strings
def remove_special_characters(texts):
    cleaned_texts = []
    # Define the pattern to match special characters 
    pattern = r'[^a-zA-Z\s]'
    for text in texts:
        # Replace special characters with an empty string and remove leading/trailing whitespaces
        cleaned_text = re.sub(pattern, '', text).strip()
        if cleaned_text:  # Check if the cleaned text is not empty
            cleaned_texts.append(cleaned_text)
    return cleaned_texts

# Function to predict the topic of an article
def predict_topic(model, article, vocab=vocab, label_vocab=label_vocab):
    # Tokenize the article
    text = article.lower()
    text = word_tokenize(text)
    text = remove_stopwords(text)
    text = remove_special_characters(text)

    # Filter out unknown words
    text = [token for token in text if token in vocab]  # Ignore unknown words

    # Convert tokens to numericalized tensor
    numericalized_tokens = torch.tensor(
        [vocab[token] for token in text], dtype=torch.long)

    # Check if there are any tokens left after filtering out unknown words
    if len(numericalized_tokens) == 0:
        raise ValueError("All words in the article are unknown to the model.")

    # Pass the tensor through the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(numericalized_tokens.unsqueeze(0))

    # Interpret the model's output
    predicted_class_index = torch.argmax(output).item()
    predicted_topic = list(label_vocab.keys())[list(
        label_vocab.values()).index(predicted_class_index)]

    return predicted_topic

