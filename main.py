from llama_sensei.backend.add_courses.vectordb.document_processor import DocumentProcessor

proc = DocumentProcessor('cs229_stanford', search_only=False)
proc.process_document(path="stanford_cs229_l1.json", metadata={'video_id': "jGwO_UgTS7I"})
# print(proc.vector_db.client.list_collections())