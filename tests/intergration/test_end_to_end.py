from app.backend.add_courses.document.document_processor import DocumentProcessor

def test_end_to_end_processing_and_search():
    doc_processor = DocumentProcessor("integration_test_collection")
    sentences = [
        {'text': 'The quick brown fox jumps over the lazy dog.', 'start': 0, 'end': 10},
        {'text': 'Pack my box with five dozen liquor jugs.', 'start': 11, 'end': 20},
        {'text': 'How vexingly quick daft zebras jump!', 'start': 21, 'end': 30},
    ]
    doc_processor.process_document(sentences)

    results = doc_processor.search("fox jumps", top_k=1)
    assert len(results['ids'][0]) == 1
    assert "fox jumps" in results['documents'][0][0].lower()

    results = doc_processor.search("liquor jugs", top_k=1)
    assert len(results['ids'][0]) == 1
    assert "liquor jugs" in results['documents'][0][0].lower()

    # results = doc_processor.search("nonexistent phrase", top_k=1)
    # assert len(results['ids'][0]) == 0