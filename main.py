from llama_sensei.backend.add_courses.vectordb.document_processor import DocumentProcessor

def retrieve_dummy(top_k: int = 3):
    return [
        {
            "timestamp": "123456789",
            "text": "When the target variable that we are trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem",
            "similarity_score": 0.32492834,
            "metadata": {"start": 2180, "end": 2300, "video_id": "jGwO_UgTS7I"},
        },
        {
            "timestamp": "131332132",
            "text": "When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.",
            "similarity_score": 0.22333834,
            "metadata": {"start": 1234, "end": 2345, "video_id": "4b4MUYve_U8"},
        },
        {
            "timestamp": "324356733",
            "text": "We will also use X denote the space of input values, and Y the space of output values.",
            "similarity_score": 0.18333834,
            "metadata": {"start": 90, "end": 245, "video_id": "het9HFqo1TQ"},
        },
    ]

dummy = retrieve_dummy()

proc = DocumentProcessor('cs229_stanford', search_only=False)

for sen in dummy:
    proc.process_document(sen['text'], sen['metadata'])

print(proc.vector_db.client.list_collections())