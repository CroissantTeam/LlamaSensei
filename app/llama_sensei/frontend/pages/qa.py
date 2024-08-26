import chromadb
import streamlit as st
from llama_sensei.backend.qa.generate_answer import GenerateRAGAnswer

def get_courses():
    client = chromadb.PersistentClient(path="data/chroma_db")
    return [x.name for x in client.list_collections()]

def show(on_internet_checkbox_change=None):
    # Sidebar for course selection
    st.sidebar.header("Course Selection")
    course_name = st.sidebar.selectbox(
        label="Choose the course you want to ask",
        options=get_courses()
    )
    
    # Initialize chat history and RAG generator
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_generator" not in st.session_state:
        st.session_state.rag_generator = GenerateRAGAnswer(course=course_name)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "evidence" in message:
                with st.expander("More information"):
                    more_info(message["evidence"])

    # Accept user input
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            rag_generator = st.session_state.rag_generator
            rag_generator.prepare_context(indb=True, internet=True, query=prompt)
            
            message_placeholder = st.empty()
            full_response = ""
            for response in rag_generator.generate_llm_answer():
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
            #  evidence = rag_generator.cal_evidence(full_response)
            #  with st.expander("More information"):
            #      more_info(evidence)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response, })
        #"evidence": evidence})

    # More info function (unchanged)
    def more_info(evidence: dict):
        if not evidence['context_list']:
            st.markdown("**No evidence found**")
            return
        tabs = st.tabs([str(i + 1) for i in range(len(evidence["context_list"]))])
        for i, ctx in enumerate(evidence["context_list"]):
            with tabs[i]:
                link = f"https://www.youtube.com/watch?v={ctx['metadata']['video_id']}&t={ctx['metadata']['start']}s"
                st.markdown(f"**Context** ([source]({link})): {ctx['context']}\n")
                col1, col2 = st.columns([3, 3])
                with col1:
                    st.video(data=link, start_time=ctx['metadata']['start'], end_time=ctx['metadata']['end'])
                f_score = ctx['f_score']
                cr_score = ctx['cr_score']
                st.markdown(f"**Faithfulness Score:** {f_score * 100:.2f}%")
                st.markdown(f"**Context Relevancy Score:** {cr_score * 100:.2f}%")

    # Internet resources checkbox (unchanged)
    if on_internet_checkbox_change:
        with st.sidebar:
            indb = st.checkbox("Course's record", value=True)
            internet = st.checkbox("Internet", value=False)
            on_internet_checkbox_change(internet)

