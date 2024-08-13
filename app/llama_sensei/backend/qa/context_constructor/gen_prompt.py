def retrieve(query: str, top_k: int = 3):
    
    return [
        {
            "timestamp": "123456789",
            "text": "When the target variable that we are trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem", 
            "similarity_score": 0.32492834,
            "metadata": {"start": 1.23, "end": 2.45}
        },
        {
            "timestamp": "324356789",
            "text": "When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.", 
            "similarity_score": 0.22333834,
            "metadata": {"start": 1.23, "end": 2.45}
        },
        {
            "timestamp": "324356733",
            "text": "We will also use X denote the space of input values, and Y the space of output values.", 
            "similarity_score": 0.18333834,
            "metadata": {"start": 1.23, "end": 2.45}
        },
    ]

def gen_prompt(query: str, course: str) -> str:
    result  = retrieve(query)
    context_list = [val["text"] for val in result]
    context = "\n\n".join(context_list)

    prompt_template = (
        f"""You are a teaching assistant. Given a set of relevant information (delimited by <info></info>) extracted from a document, please compose an answer to the question of a student. Ensure that the answer is accurate, has a friendly tone, and sounds helpful. If you can not answer, ask the student to clarify the question.
            <info>
            {context}
            </info>
            Question: {query}
            Answer: """
    )

    return prompt_template

def answer(query: str, course: str):
    from langchain_groq import ChatGroq
    model = ChatGroq(model="llama3-70b-8192", temperature=0)
    
    res = model.invoke(gen_prompt(query, course))
    return res.content

if __name__ == '__main__':
    result = answer(query="What method do we use if we want to predict house price in an area ?", 
           course="cs229_stanford")
    print(result)