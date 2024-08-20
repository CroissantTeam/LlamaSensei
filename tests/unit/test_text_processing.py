# import pytest
# import numpy as np

# def test_preprocess(text_processor):
#     text = "This is a Test Sentence with Some Stop Words."
#     processed = text_processor.preprocess(text)
#     assert processed == "test sentence stop words ."

# def test_text_chunking(text_processor):
#     sentences = [
#         {'text': 'This is the first sentence with more words to increase token count.', 'start': 0, 'end': 5},
#         {'text': 'Here is the second sentence, also longer to ensure we exceed the token limit.', 'start': 6, 'end': 11},
#         {'text': 'Finally, the third sentence adds even more tokens to force a new chunk.', 'start': 12, 'end': 17}
#     ]
    
#     chunks = text_processor.chunk_text(sentences, max_tokens=20)
    
#     assert len(chunks) == 3
#     assert chunks[0]['text'] == 'This is the first sentence with more words to increase token count.'
#     assert chunks[1]['text'] == 'Here is the second sentence, also longer to ensure we exceed the token limit.'
#     assert chunks[2]['text'] == 'Finally, the third sentence adds even more tokens to force a new chunk.'

# def test_get_embedding(text_processor):
#     text = "This is a test sentence."
#     embedding = text_processor.get_embedding(text)
#     assert isinstance(embedding, np.ndarray)
#     assert embedding.shape == (384,)  # The shape depends on the model used

# def test_preprocess_remove_stopwords(text_processor):
#     text = "The quick brown fox jumps over the lazy dog"
#     processed = text_processor.preprocess(text)
#     assert "the" not in processed
#     assert "quick brown fox jumps lazy dog" in processed

# def test_chunk_text_empty_input(text_processor):
#     assert text_processor.chunk_text([]) == []

# def test_chunk_text_single_sentence(text_processor):
#     sentences = [{'text': 'This is a single sentence.', 'start': 0, 'end': 5}]
#     chunks = text_processor.chunk_text(sentences, max_tokens=20)
#     assert len(chunks) == 1
#     assert chunks[0]['text'] == 'This is a single sentence.'

# def test_get_embedding_different_texts(text_processor):
#     text1 = "This is the first sentence."
#     text2 = "This is a completely different sentence."
#     embedding1 = text_processor.get_embedding(text1)
#     embedding2 = text_processor.get_embedding(text2)
#     assert not np.array_equal(embedding1, embedding2)