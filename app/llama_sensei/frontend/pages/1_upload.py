import streamlit as st
from utils.client import add_course, get_courses

if 'list_name' not in st.session_state:
    st.session_state.list_name = get_courses()

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
        response = add_course(url, course_name)
        if "Success" in response["message"]:
            st.success("Completed upload")
    except Exception as e:
        st.error(f"An error occurred while uploading: {str(e)}")
        st.rerun()


st.button("Upload", on_click=upload)
