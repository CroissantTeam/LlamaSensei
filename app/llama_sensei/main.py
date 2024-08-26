import streamlit as st
import os

script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Define the path to the assets directory relative to the script directory
assets_dir = os.path.join(script_dir, 'assets/')

st.set_page_config(
    page_title="Hello world",              # Title of the web page
    page_icon="chart_with_upwards_trend",  # Icon to display in the browser tab
    layout="wide"                          # Use wide layout
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Questions and Answers"
if "notification" not in st.session_state:
    st.session_state.notification = ""
if "internet_warning" not in st.session_state:
    st.session_state.internet_warning = False

# Function to update page
def set_page():
    st.session_state.page = st.session_state.page_selectbox

# Sidebar navigation
st.sidebar.title("Features")
page_options = ["Questions and Answers", "Upload", "About Us"]
st.sidebar.selectbox(
    "",
    options=page_options,
    key="page_selectbox",
    on_change=set_page
)

# Display the notification
if st.session_state.notification:
    st.sidebar.info(st.session_state.notification)

# Load page functions dynamically
def load_page_functions():
    from llama_sensei.frontend.pages.upload import show as show_upload
    from llama_sensei.frontend.pages.qa import show as show_qa
    from llama_sensei.frontend.pages.about_us import show as show_about_us
    return show_upload, show_qa, show_about_us

show_upload, show_qa, show_about_us = load_page_functions()

# Display the selected page content
if st.session_state.page == "Upload":
    show_upload()

elif st.session_state.page == "Questions and Answers":
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image(assets_dir + "llama.png", width=60)  # Add logo or image here
    with col2:
        st.markdown("<h1 style='font-size: 50px; margin-top: -9px;'>Llama Sensei</h1>", unsafe_allow_html=True)
    
    # Initialize the internet_checkbox state if it doesn't exist
    if 'internet_checkbox_state' not in st.session_state:
        st.session_state.internet_checkbox_state = False
    
    def on_internet_checkbox_change(checked):
        if checked and not st.session_state.internet_checkbox_state:
            st.session_state.show_warning = True
        else:
            st.session_state.show_warning = False
        st.session_state.internet_checkbox_state = checked

    show_qa(on_internet_checkbox_change=on_internet_checkbox_change)
    
    if st.session_state.get('show_warning', False):
        st.toast("Using internet retrieval may affect accuracy and privacy.")

elif st.session_state.page == "About Us":
    show_about_us()
