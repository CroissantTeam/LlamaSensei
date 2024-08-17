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
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        return ' '.join([token for token in tokens if token not in self.stop_words])

    def chunk_text(self, sentences, max_tokens=512):
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_start = None
        chunk_end = None

        for sentence in sentences:
            tokens = self.tokenizer.encode(sentence['text'], add_special_tokens=False)
            if current_tokens + len(tokens) > max_tokens and current_chunk:
                chunks.append(
                    {
                        'text': ' '.join([s['text'] for s in current_chunk]),
                        'start': chunk_start,
                        'end': chunk_end,
                    }
                )
                current_chunk = []
                current_tokens = 0
                chunk_start = None

            current_chunk.append(sentence)
            current_tokens += len(tokens)
            if chunk_start is None:
                chunk_start = sentence['start']
            chunk_end = sentence['end']

        if current_chunk:
            chunks.append(
                {
                    'text': ' '.join([s['text'] for s in current_chunk]),
                    'start': chunk_start,
                    'end': chunk_end,
                }
            )

        return chunks

    def get_embedding(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
