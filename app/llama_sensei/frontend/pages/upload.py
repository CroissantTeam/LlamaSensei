import asyncio
import glob
import os
import shutil
import streamlit as st
from llama_sensei.backend.add_courses.document.transcript import DeepgramSTTClient
from llama_sensei.backend.add_courses.vectordb.document_processor import DocumentProcessor
from llama_sensei.backend.add_courses.yt_api.audio import YouTubeAudioDownloader
from llama_sensei.backend.add_courses.yt_api.playlist import PlaylistVideosFetcher

def create_course_ui():
    """Display the UI for creating a new course."""

    if 'course_name' not in st.session_state:
        st.session_state.course_name = None
    if 'erase_db' not in st.session_state:
        st.session_state.erase_db = False
    # Display Back button to return to home view
    st.sidebar.button(label="Back", on_click=lambda: st.session_state.update({'current_view': 'home'}), key="create back")

    st.title("Add a New Course to Database")
    col1, col2 = st.columns(2)
    with col1:
        with st.form(key='create_course_form'):
            name = st.text_input("Enter the name of the new course:", "")
            submit_button = st.form_submit_button(label="Submit")
            if submit_button:
                if name:
                    if name not in st.session_state.list_name:
                        st.session_state.list_name.append(name)
                        st.session_state.course_name = name
                        st.toast(f"✅ Course '{name}' created successfully ✅")
                    elif name in st.session_state.list_name:
                        st.toast(f"⚠️  Course '{name}' already exists ⚠️")
                else:
                    st.toast("❗Course name cannot be empty ❗")
    with col2:
        with st.form(key='upload_form'):
            url = st.text_input(
                    label="New Youtube video for course: ",
                    placeholder="https://www.youtube.com/watch?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU"
                )
            upload_button = st.form_submit_button(label="Upload")
            if upload_button:
                if url:
                    upload_course_data(st.session_state.course_name, url, st.session_state.erase_db)
                else:
                    st.toast("⚠️ YouTube URL cannot be empty. ⚠️")

def choose_course_ui():
    """Display the UI for choosing an existing course."""
    if 'course_name' not in st.session_state:
        st.session_state.course_name = None
    if 'erase_db' not in st.session_state:
        st.session_state.erase_db = False
    # Display Back button to return to home view
    st.sidebar.button(label="Back", on_click=lambda: st.session_state.update({'current_view': 'home'}), key="choose back")

    st.title("Modify Existing Course in Database")
    # Adding custom CSS to style the container

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            #  st.write("Select your desired course")
            st.session_state.course_name = st.selectbox(
                label="Select your desired course",
                options=st.session_state.list_name,
                key="course_selector"
            )

            if st.button("Erase all data currently in course database", key="remove_all"):
                st.toast("✅ All data has been erased ✅")  # Show toast notification
                proc = DocumentProcessor(st.session_state.course_name, search_only=True)
                proc.erase_all_data()

    with col2:
        with st.form(key='upload_form'):
            url = st.text_input(
                    label="New Youtube video for course: ",
                    placeholder="https://www.youtube.com/watch?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU"
                )
            upload_button = st.form_submit_button(label="Upload")
            if upload_button:
                if url:
                    upload_course_data(st.session_state.course_name, url, st.session_state.erase_db)
                else:
                    st.toast("⚠️ YouTube URL cannot be empty. ⚠️")

    st.write("Videos currently in course's database:")

    import chromadb
    client = chromadb.PersistentClient(path="data/chroma_db")
    col = client.get_collection(st.session_state.course_name)
    print(col.count())
    if col.count():
        yad = YouTubeAudioDownloader("data/", course_name=st.session_state.course_name)
        video_set = list({f"https://www.youtube.com/watch?v={x['video_id']}" for x in col.get()['metadatas']})
        table = {'Title': [yad.extract_title(x) for x in video_set], 
                 'URL': video_set}
        st.dataframe(data=table, use_container_width=True, hide_index=True)

def upload_course_data(course_name, url, erase_db=False):
    """Handle the upload process for the selected course."""
    try:
        fetcher = PlaylistVideosFetcher()
        video_urls = fetcher.get_playlist_videos(url)

        downloader = YouTubeAudioDownloader("data/", course_name=course_name)
        titles, ids = downloader.download_audio(video_urls)

        audio_list = glob.glob(os.path.join("data", course_name, "audio/*.wav"))
        deepgram_client = DeepgramSTTClient(
            os.path.join("data/transcript", course_name)
        )
        deepgram_client.get_transcripts(audio_list)

        proc = DocumentProcessor(course_name, search_only=False)
        folder_path = f"data/transcript/{course_name}/"
        for video_id in os.listdir(folder_path):
            proc.process_document(
                path=os.path.join(folder_path, video_id),
                metadata={'video_id': video_id.split('.')[0]},
            )
    except Exception as e:
        st.toast(f"⚠️ An error occurred while uploading: {str(e)} ⚠️")
        print(str(e))
        st.rerun()
    st.toast("✅ Upload completed successfully! ✅")

    # Clean up
    shutil.rmtree(os.path.join("data", course_name))
    shutil.rmtree(os.path.join("data/transcript", course_name))

def show():
    """Main function to display the Streamlit app."""
    if 'list_name' not in st.session_state:
        import chromadb
        client = chromadb.PersistentClient(path="data/chroma_db")
        st.session_state.list_name = [x.name for x in client.list_collections()]
        st.session_state.current_view = "home"


    if st.session_state.current_view == "create":
        create_course_ui()

    # Choose Course View
    elif st.session_state.current_view == "choose":
        choose_course_ui()

    else:
        st.title("Course Upload Tool")
        st.write("Description")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Choose Existing Course", on_click=lambda: st.session_state.update({'current_view': 'choose'}), key="col1"):
                st.rerun() # Ensure the UI updates immediately

        with col2:
            if st.button("Create New Course", on_click=lambda: st.session_state.update({'current_view': 'create'}), key="col2"):
                st.rerun()  # Ensure the UI updates immediately

