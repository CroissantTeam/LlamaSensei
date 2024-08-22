from llama_sensei.backend.add_courses.vectordb.document_processor import (
    DocumentProcessor,
)

proc = DocumentProcessor('cs229_stanford', search_only=False)
proc.process_document(
    path="data/cs229_stanford/transcript/0rt2CsEQv6U.json",
    metadata={'video_id': "0rt2CsEQv6U"},
)
# print(proc.vector_db.client.list_collections())
