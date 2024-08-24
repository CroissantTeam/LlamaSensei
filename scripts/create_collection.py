import glob
import os

from llama_sensei.backend.add_courses.vectordb.document_processor import (
    DocumentProcessor,
)
from llama_sensei.backend.add_courses.vectordb.vector_db_operations import (
    VectorDBOperations,
)

course_name = "cs229_stanford"
transcript_folder = f"/shared/final/{course_name}/transcript/"
transcript_files = glob.glob(os.path.join(transcript_folder, "*.json"))
ids = [os.path.basename(name)[:-5] for name in transcript_files]
print(transcript_files, ids)

vectordb = VectorDBOperations("data/")
proc = DocumentProcessor(vectordb, course_name, search_only=False)
for path, video_id in zip(transcript_files, ids):
    proc.process_document(path=path, metadata={'video_id': video_id})
# print(proc.vector_db.client.list_collections())
