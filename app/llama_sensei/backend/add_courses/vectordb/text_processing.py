import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoModel, AutoTokenizer

nltk.download('punkt')
nltk.download('stopwords')


class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words]

    def chunk_text(self, text, chunk_size=100):
        words = text.split()
        return [
            ' '.join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

    def get_embedding(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
