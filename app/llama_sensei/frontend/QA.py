import time

import chromadb
import streamlit as st
from llama_sensei.backend.qa.generate_answer import GenerateRAGAnswer


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
    #print(evidence)
    tabs = st.tabs([str(i + 1) for i in range(len(evidence["context_list"]))])
    for (i, ctx) in enumerate(evidence["context_list"]):
        with tabs[i]:
            if 'link' in ctx['metadata']:
                st.markdown(
                    f"**Context** ([source]({ctx['metadata']['link']})): {ctx['context']}\n"
                )
            else:
                st.markdown(
                    f"**Context** ([source](https://www.youtube.com/watch?v={ctx['metadata']['video_id']}&t={ctx['metadata']['start']}s)): {ctx['context']}\n"
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
def response_generator(input: str, indb: bool, internet: bool):
    rag_generator = GenerateRAGAnswer(query=input, course=course_name)
    answer, evidence = rag_generator.generate_answer(indb, internet)
    return answer, evidence


def streaming(input: str):
    for w in input.split(" "):
        yield w + " "
        time.sleep(0.05)

with st.sidebar:
    st.write("Choose resources to search from: ")
    indb = st.checkbox("Course's record", value=True)
    internet = st.checkbox("Internet")

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
            answer, evidence = response_generator(prompt, indb, internet)
            st.write_stream(streaming(answer))
            # print(evidence)
            with st.expander("More information"):
                more_info(evidence)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "evidence": evidence}
            )