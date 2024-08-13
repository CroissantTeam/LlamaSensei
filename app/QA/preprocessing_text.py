import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.chunk import ne_chunk #https://www.nltk.org/api/nltk.chunk.html
from nltk.corpus import stopwords
from nltk.tag import pos_tag


from get_embedding import embedding

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt',quiet=True)
nltk.download('punkt_tab',quiet=True)#sent_tokenize
nltk.download('wordnet',quiet=True)
nltk.download('averaged_perceptron_tagger',quiet=True)
# nltk.download('maxent_ne_chunker') #ne_chunk
nltk.download('words',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('averaged_perceptron_tagger_eng',quiet=True)
# nltk.download('maxent_ne_chunker_tab')  #ne_chunk

def preprocess_transcript(text,apply_lemantize=True,apply_stem=True):
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    
    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    words = [[word for word in sentence if word.lower() not in stop_words] for sentence in words]

    # POS tagging
    pos_tagged = [pos_tag(sentence) for sentence in words]
    print(pos_tagged[0])

    # Lemmatization and stemming
    if apply_lemantize:
        pos_tagged = [[lemmatizer.lemmatize(word, pos='v') for word, tag in sentence] for sentence in pos_tagged]
    else:
        pos_tagged = [[word for word, tag in sentence] for sentence in pos_tagged]
    if apply_stem:
        pos_tagged = [[stemmer.stem(word) for word in sentence] for sentence in pos_tagged]

    # Reconstruct sentences from words
    # print(pos_tagged[0])
    preprocessed_text = [' '.join(sentence) for sentence in pos_tagged]
    return ' '.join(preprocessed_text)

def chunk(words,chunk_size=500):
    # Chunking (Named Entity Recognition)
    # chunked = [ne_chunk(sentence) for sentence in pos_tagged]
    #fixed size chunking
    chunk_size=512
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    return chunks

def group_chunks(chunks_list):
    if chunks_list==[]:return
    
    start,end=chunks_list[0][1],chunks_list[-1][2]
    transcript = " ".join([chunk[0] for chunk in chunks_list])
    return (transcript,start,end)

# chunks_list = [("jkhsfad",0,55),
#                ("yugfdsa",57,78)]

# print(group_chunks(chunks_list=chunks_list))

if __name__=="__main__":
    # Example usage
    with open("text.txt","r") as file:
        transcript = file.readlines()
        transcript= " ".join(transcript)

    
    preprocessed = preprocess_transcript(transcript,False,False)
    chunks = chunk(preprocessed)
    embedded = [embedding(chunk) for chunk in chunks]
    print(embedded)
    with open("lemantized_embedded.txt","w") as file: 
        from time import perf_counter
        tin = perf_counter()  
        for chunk, embed in zip(chunks,embedded):
            file.write(chunk)
            file.write("\n")
            file.write(str(embed.tolist()))
            file.write("\n")
        file.write(f"cost {perf_counter()-tin}s")
    # with open("non_lemantize.txt","w") as file:
    #     file.write(preprocess_transcript(transcript,False,False))
    # with open("lemantize.txt","w") as file:
    #     file.write(preprocess_transcript(transcript))
    # print(len(chunk())
    # print(len(chunk(transcript)))

