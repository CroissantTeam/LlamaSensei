import glob
import os

from llama_sensei.backend.add_courses.vectordb.document_processor import (
    DocumentProcessor,
)

course_name = "cs224n_stanford"
transcript_folder = f"/shared/final/{course_name}/transcript/"
transcript_files = glob.glob(os.path.join(transcript_folder, "*.json"))
ids = [os.path.basename(name)[:-5] for name in transcript_files]
print(transcript_files, ids)

proc = DocumentProcessor(course_name, search_only=False)
for path, video_id in zip(transcript_files, ids):
    proc.process_document(path=path, metadata={'video_id': video_id})
# print(proc.vector_db.client.list_collections())
