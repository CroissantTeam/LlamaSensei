import streamlit as st
import time
from llama_sensei.backend.qa.llm_api.generate_answer import GenerateRAGAnswer

def get_courses():
    return ["cs224n_stanford", "cs229_stanford", "cs231n_stanford"]

st.title("Echo Bot")
course_name = st.selectbox(label="Choose the course you want to ask", 
             options=get_courses())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history osn app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamed response emulator
def response_generator(input: str):
    rag_generator = GenerateRAGAnswer(query=prompt, course=course_name)
    response = rag_generator.generate_answer()
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    if prompt == "/clear":
        st.session_state.messages.clear()
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})