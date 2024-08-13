from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import faithfulness

MODEL = "llama3-70b-8192"


def retrieve(query: str, top_k: int = 3):
    return [
        {
            "timestamp": "123456789",
            "text": "When the target variable that we are trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem",
            "similarity_score": 0.32492834,
            "metadata": {"start": 2180, "end": 2300, "link": "https://www.youtube.com/watch?v=jGwO_UgTS7I"},
        },
        {
            "timestamp": "131332132",
            "text": "When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.",
            "similarity_score": 0.22333834,
            "metadata": {"start": 1234, "end": 2345, "link": "https://www.youtube.com/watch?v=jGwO_UgTS7I"},
        },
        {
            "timestamp": "324356733",
            "text": "We will also use X denote the space of input values, and Y the space of output values.",
            "similarity_score": 0.18333834,
            "metadata": {"start": 90, "end": 245, "link": "https://www.youtube.com/watch?v=jGwO_UgTS7I"},
        },
    ]


class GenerateRAGAnswer:
    def __init__(self, query: str, course: str, model=MODEL):
        self.query = query
        self.course = course
        self.model = ChatGroq(model=model, temperature=0)
        self.contexts = None  # To store the retrieved contexts

    def retrieve_contexts(self):
        result = retrieve(self.query)
        self.contexts = [
            {"text": val["text"], "metadata": val["metadata"]} for val in result
        ]  # Store contexts for use in prompt generation and evaluation
        return self.contexts

    def gen_prompt(self) -> str:
        # Extract the 'text' field from each context dictionary
        context_texts = [f"{ctx['text']}" for ctx in self.contexts]

        # Join the extracted text with double newlines
        context = "\n\n".join(context_texts)

        prompt_template = f"""
            You are a teaching assistant. Given a set of relevant information (delimited by <info></info>) extracted from a document,
            please compose an answer to the question of a student. Ensure that the answer is accurate, has a friendly tone, and sounds
            helpful. If you cannot answer, ask the student to clarify the question.
            If no context is available in the system, please answer that you can not find the relevant context in the system.
            <info>
            {context}
            </info>
            Question: {self.query}
            Answer: """

        return prompt_template

    def generate_answer(self) -> str:
        self.retrieve_contexts()
        final_prompt = self.gen_prompt()
        res = self.model.invoke(final_prompt)

        llm_answer = res['content'] if isinstance(res, dict) else res.content

        # Calculate score
        faithfulness_score = self.calculate_faithfulness(llm_answer)
        answer_relevancy_score = self.calculate_answer_relevancy(llm_answer)

        context_list = [{"context": ctx["text"], "metadata": ctx["metadata"]} for ctx in self.contexts]

        #evidence = (
        #    f"**Retrieved Contexts:**\n{context_str}\n\n"
        #    f"**Faithfulness Score:** {faithfulness_score:.4f}\n"
        #    f"**Answer Relevancy Score:** {answer_relevancy_score:.4f}"
        #)

        evidence = {"context_list": context_list, "f_score": faithfulness_score, "ar_score": answer_relevancy_score}

        return llm_answer, evidence

    def calculate_faithfulness(self, generated_answer: str) -> float:
        if not self.contexts:
            raise ValueError(
                "Contexts have not been retrieved. Ensure contexts are retrieved before this method is called."
            )

        model = self.model

        # Prepare the dataset for evaluation
        data_samples = {
            'question': [self.query],
            'answer': [generated_answer],
            'contexts': [[val["text"] for val in self.contexts]],
        }
        dataset = Dataset.from_dict(data_samples)

        # Evaluate faithfulness
        score = evaluate(dataset, metrics=[faithfulness], llm=model)

        # Convert score to pandas DataFrame and get the first score
        score_df = score.to_pandas()
        return score_df['faithfulness'].iloc[0]

    def calculate_answer_relevancy(self, generated_answer: str) -> float:
        if not self.contexts:
            raise ValueError(
                "Contexts have not been retrieved. Ensure contexts are retrieved before this method is called."
            )

        model = self.model

        # Prepare the dataset for evaluation
        data_samples = {
            'question': [self.query],
            'answer': [generated_answer],
            'contexts': [[val["text"] for val in self.contexts]],
        }
        dataset = Dataset.from_dict(data_samples)

        # Evaluate faithfulness
        score = evaluate(dataset, metrics=[faithfulness], llm=model)

        # Convert score to pandas DataFrame and get the first score
        score_df = score.to_pandas()
        return score_df['faithfulness'].iloc[0]

# Example usage
if __name__ == "__main__":
    prompt = "What method do we use if we want to predict house price in an area?"
    course = "cs229_stanford"

    # Create an instance of GenerateRAGAnswer with the query and course
    rag_generator = GenerateRAGAnswer(query=prompt, course=course)

    # Generate and print the answer along with its embedded faithfulness score
    answer = rag_generator.generate_answer()
    print(answer)
