from time import perf_counter
from typing import List
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag

class TextPreprocessor:
    def __init__(self):
        # Ensure necessary NLTK resources are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self,chunks:List[tuple], apply_lemmatize=True, apply_stem=True) -> List[tuple]:
        return [(self._preprocess(
            chunk[0],apply_lemmatize=apply_lemmatize,apply_stem=apply_stem),chunk[1],chunk[2])
            for chunk in chunks]


    def _preprocess(self, text, apply_lemmatize=True, apply_stem=True):
        sentences = sent_tokenize(text)
        words = [word_tokenize(sentence) for sentence in sentences]
        
        words = [[word for word in sentence if word.lower() not in self.stop_words] for sentence in words]
        pos_tagged = [pos_tag(sentence) for sentence in words]
        
        if apply_lemmatize:
            pos_tagged = [[self.lemmatizer.lemmatize(word, pos='v') for word, tag in sentence] for sentence in pos_tagged]
        else:
            pos_tagged = [[word for word, tag in sentence] for sentence in pos_tagged]
        
        if apply_stem:
            pos_tagged = [[self.stemmer.stem(word) for word in sentence] for sentence in pos_tagged]

        preprocessed_text = [' '.join(sentence) for sentence in pos_tagged]
        return ' '.join(preprocessed_text)

    def chunk(self, words, chunk_size=512):
        return [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    @staticmethod
    def merge_text(sentences:List[tuple]) -> tuple:
        if not sentences:
            return None
        
        start, end = sentences[0][1], sentences[-1][2]
        chunk = " ".join([sentence[0] for sentence in sentences])
        return (chunk, start, end)


