from llama_sensei.backend.add_courses.embedding.preprocessing_text import TextPreprocessor
from llama_sensei.backend.add_courses.embedding.load_text import TranscriptLoader
from llama_sensei.backend.add_courses.embedding.get_embedding import Embedder

transcriptLoader = TranscriptLoader("stanford_cs229_l1.json")
textPreprocessor = TextPreprocessor()
embedder = Embedder()

chunks = transcriptLoader.load_data()
# chunks = [textPreprocessor.merge_text(sentences[i:min(i+3,len(sentences))]) for i in range(0,len(sentences),3)]
print(len(chunks))

preprocessed_chunks = textPreprocessor.preprocess_text(chunks,apply_lemmatize=True, apply_stem=True)
chunks_with_embed = embedder.embed_chunks(preprocessed_chunks)
print(len(chunks_with_embed))


