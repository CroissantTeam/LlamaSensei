import glob
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from schemas import AddCourseRequest, SearchQuery, SearchResponse
from speech_to_text.transcript import DeepgramSTTClient
from vectordb.document_processor import DocumentProcessor
from vectordb.vector_db_operations import VectorDBOperations
from yt_api.audio import YouTubeAudioDownloader
from yt_api.playlist import PlaylistVideosFetcher

DATA_SAVE_DIR = "data/"
load_dotenv()
vectordb = VectorDBOperations()
app = FastAPI(title="LlamaSensei: Course management API")


@app.post("/add_course/")
async def add_course(request: AddCourseRequest):
    try:
        fetcher = PlaylistVideosFetcher()
        video_urls = fetcher.get_playlist_videos(request.playlist_url)
        print(video_urls)

        downloader = YouTubeAudioDownloader(
            output_path=DATA_SAVE_DIR, course_name=request.course_name
        )
        downloader.download_audio(video_urls)
        print('Download success')

        course_audio_dir = os.path.join(
            DATA_SAVE_DIR, request.course_name, "audio/*.wav"
        )
        audio_list = glob.glob(course_audio_dir)
        transcript_dir = os.path.join(DATA_SAVE_DIR, request.course_name, "transcript/")
        deepgram_client = DeepgramSTTClient(output_path=transcript_dir)
        deepgram_client.get_transcripts(audio_list)
        print("Transcript success")

        proc = DocumentProcessor(request.course_name, search_only=False)
        for video_id in os.listdir(transcript_dir):
            proc.process_document(
                path=os.path.join(transcript_dir, video_id),
                metadata={'video_id': video_id.split('.')[0]},
            )

        return {"message": "Success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/", response_model=SearchResponse)
async def search(query: SearchQuery):
    try:
        document_processor = DocumentProcessor(
            collection_name=query.course_name,
            search_only=True,
        )
        result = document_processor.search(
            query=query.text,
            top_k=query.top_k,
        )
        return SearchResponse(
            documents=result['documents'][0], metadatas=result['metadatas'][0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/courses/")
async def get_courses():
    try:
        vectordb = VectorDBOperations()
        available_courses = vectordb.get_collections()
        return JSONResponse(content=available_courses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to LlamaSensei: Course management API"}


def main():
    # Run web server with uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("COURSE_FASTAPI_HOST", "127.0.0.1"),
        port=int(os.getenv("COURSE_FASTAPI_PORT", 8002)),
        # reload=True,  # Uncomment this for debug
        # workers=2,
    )


if __name__ == "__main__":
    main()
