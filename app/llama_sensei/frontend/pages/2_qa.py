import time

import chromadb
import requests
import streamlit as st


def get_courses():
    client = chromadb.PersistentClient(path="data/chroma_db")
    return [x.name for x in client.list_collections()]


st.set_page_config(layout="wide")

st.title("Llama Sensei")
course_name = st.selectbox(
    label="Choose the course you want to ask", options=get_courses()
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# display more info about response
def more_info(evidence: dict):
    for ctx in evidence["context_list"]:
        st.markdown(
            f"**Context** (start from [here](https://www.youtube.com/watch?v={ctx['metadata']['video_id']}&t={ctx['metadata']['start']}s)): {ctx['context']}\n"
        )

    st.markdown(f"**Faithfulness Score:** {evidence['f_score']:.4f}\n")
    st.markdown(f"**Answer Relevancy Score:** {evidence['ar_score']:.4f}")


# Display chat messages from history osn app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evidence" in message:
            with st.expander("More information"):
                more_info(message["evidence"])


# Streamed response emulator
def response_generator(input: str):
    chat_query = {"question": input, "course": course_name}
    r = requests.post("http://localhost:8000/generate_answer", json=chat_query)
    print("*" * 20, r.json())
    response = r.json()
    return response["answer"], response["evidence"]


def streaming(input: str):
    for w in input.split(" "):
        yield w + " "
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
            answer, evidence = response_generator(prompt)
            st.write_stream(streaming(answer))
            # print(evidence)
            with st.expander("More information"):
                more_info(evidence)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "evidence": evidence}
            )
