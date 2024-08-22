import asyncio
import glob
import os

import chromadb
import streamlit as st
from llama_sensei.backend.add_courses.document.transcript import DeepgramSTTClient
from llama_sensei.backend.add_courses.vectordb.document_processor import (
    DocumentProcessor,
)
from llama_sensei.backend.add_courses.yt_api.audio import YouTubeAudioDownloader
from llama_sensei.backend.add_courses.yt_api.playlist import PlaylistVideosFetcher

import shutil

if 'list_name' not in st.session_state:
    client = chromadb.PersistentClient(path="data/chroma_db")
    st.session_state.list_name = [x.name for x in client.list_collections()]

st.write("# This is for uploading courses")


@st.dialog("Create new course")
def create_course():
    st.write("Please enter the name of the new course: ")
    name = st.text_input("Example: 'cs229_stanford', 'cs50_harvard'")
    if st.button("Submit"):
        if name not in st.session_state.list_name:
            st.session_state.list_name.append(name)
        st.rerun()


st.write("Choose the course you want to upload:")
course_name = st.selectbox(
    label="You can create new course if it did not exist",
    options=st.session_state.list_name,
)

if st.button("Add new"):
    create_course()

erase_db = st.checkbox("Erase all data currently in course database")

st.write("Paste the youtube link of the course:")
url = st.text_input(
    "Example: https://www.youtube.com/watch?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU"
)


def upload():
    try:
        fetcher = PlaylistVideosFetcher()
        video_urls = fetcher.get_playlist_videos(url)
        print(video_urls)

        downloader = YouTubeAudioDownloader("data/", course_name=course_name)
        downloader.download_audio(video_urls)
        print('Download success')

        audio_list = glob.glob(os.path.join("data", course_name, "audio/*.wav"))
        deepgram_client = DeepgramSTTClient(
            os.path.join("data/transcript", course_name)
        )
        asyncio.run(deepgram_client.get_transcripts(audio_list))
        print("transcript success")

        proc = DocumentProcessor(course_name, search_only=False)
        if erase_db:
            proc.erase_all_data()
        folder_path = f"data/transcript/{course_name}/"
        for video_id in os.listdir(folder_path):
            proc.process_document(
                path=os.path.join(folder_path, video_id),
                metadata={'video_id': video_id.split('.')[0]},
            )
    except Exception as e:
        st.error(f"An error occurred while uploading: {str(e)}")
        st.rerun()
    st.success("Completed upload")

    # Clean up
    shutil.rmtree(os.path.join("data", course_name))
    shutil.rmtree(os.path.join("data/transcript", course_name))

st.button("Upload", on_click=upload)
