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
    if evidence['context_list'] == []:
        st.markdown("**No evidence found**")
        return
    tabs = st.tabs([str(i + 1) for i in range(len(evidence["context_list"]))])
    for i, ctx in enumerate(evidence["context_list"]):
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
            if "rag_generator" not in st.session_state:
                st.session_state.rag_generator = GenerateRAGAnswer(course=course_name)
            rag_generator = st.session_state.rag_generator
            rag_generator.prepare_context(indb, internet, query=prompt)
            answer = st.write_stream(rag_generator.generate_llm_answer())
            evidence = rag_generator.cal_evidence(answer)
            with st.expander("More information"):
                more_info(evidence)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "evidence": evidence}
            )
