import streamlit as st
from utils.client import evaluate_evidence, get_courses, response_generator

st.set_page_config(layout="wide")

st.title("Llama Sensei")
course_name = st.selectbox(
    label="Choose the course you want to ask", options=get_courses()
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []


# display more info about response
def more_info(evidence: list):
    if not evidence:
        st.markdown("**No evidence found**")
        return
    tabs = st.tabs([str(i + 1) for i in range(len(evidence))])
    for i, ctx in enumerate(evidence):
        with tabs[i]:
            if 'link' in ctx['metadata']:
                st.markdown(
                    f"**Context** ([source]({ctx['metadata']['link']})): {ctx['text']}\n"
                )
            else:
                st.markdown(
                    f"**Context** ([source](https://www.youtube.com/watch?v={ctx['metadata']['video_id']}&t={ctx['metadata']['start']}s)): {ctx['text']}\n"
                )


def show_score(scores: dict):
    st.markdown(f"**Faithfulness Score:** {scores['f_score']}\n")
    st.markdown(f"**Answer Relevancy Score:** {scores['ar_score']}")


# Display chat messages from history osn app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evidence" in message:
            with st.expander("More information"):
                more_info(message["evidence"])
        if "score" in message:
            show_score(message["score"])


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
            answer = st.write_stream(
                response_generator(prompt, course_name, indb, internet)
            )
            evidence = st.session_state.contexts
            with st.expander("More information"):
                more_info(evidence)

            scores = evaluate_evidence(
                query=prompt,
                answer=answer,
                contexts=st.session_state.contexts,
                course=course_name,
            )
            show_score(scores)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "evidence": evidence,
                    "score": scores,
                }
            )
