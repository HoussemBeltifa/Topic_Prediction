import torch
import torch.nn as nn
from utils import *

# Define the model
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.embedding = nn.EmbeddingBag(
            vocab_size, embedding_dim, mode='mean')
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.ReLU()
        self.fc4 = nn.Dropout(p=0.3)

    def forward(self, text):
        embedded = self.embedding(text)
        x = self.fc1(embedded)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


# Initialize the model
model = Model(vocab_size=30239, embedding_dim=100,
              hidden_dim=128, output_dim=5)

#load the model
model.load_state_dict(torch.load("model_weights.pth"))

def themodel():
    return model