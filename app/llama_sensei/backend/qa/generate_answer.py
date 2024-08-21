from datetime import datetime

from datasets import Dataset
from langchain_groq import ChatGroq
from llama_sensei.backend.add_courses.vectordb.document_processor import (
    DocumentProcessor,
)
from llama_sensei.backend.add_courses.embedding.get_embedding import Embedder
from ragas import evaluate
from ragas.metrics import faithfulness

import time

MODEL = "llama3-70b-8192"


class GenerateRAGAnswer:
    def __init__(self, query: str, course: str, model=MODEL):
        self.query = query
        self.course = course
        self.model = ChatGroq(model=model, temperature=0)
        self.contexts = None  # To store the retrieved contexts

    def retrieve_contexts(self):
        processor = DocumentProcessor(self.course, search_only=True)
        result = processor.search(self.query)
        # print(result)
        self.contexts = [
            {"text": text, "metadata": metadata}
            for text, metadata in zip(result['documents'][0], result['metadatas'][0])
        ]  # Store contexts for use in prompt generation and evaluation
        return self.contexts

    def gen_prompt(self) -> str:
        # Extract the 'text' field from each context dictionary
        context_texts = [f"{ctx['text']}" for ctx in self.contexts]

        # Join the extracted text with double newlines
        context = "\n\n".join(context_texts)

        prompt_template = (
            """
            You are a teaching assistant.
            Given a set of relevant information from teacher's recording during the lesson """
            """(delimited by <info></info>), please compose an answer to the question of a student.
            Ensure that the answer is accurate, has a friendly tone, and sounds helpful.
            If you cannot answer, ask the student to clarify the question.
            If no context is available in the system, """
            f"""please answer that you can not find the relevant context in the system.
            <info>
            {context}
            </info>
            Question: {self.query}
            Answer: """
        )

        return prompt_template

    def external_search(self):
        pass

    def prepare_context(self, indb: bool, internet: bool):
        before = datetime.now()
        if internet == True:
            search_results = self.external_search()
            # Populate self.contexts with 'text' and 'embedding'
            embedder = Embedder()  # Initialize the embedder
            self.contexts = [
                {
                    "text": result['snippet'],
                    "metadata": {"link": result['link']},
                    "embedding": embedder.embed(result['snippet'])
                }
                for result in search_results
            ]

        if indb == True:
            self.retrieve_contexts()

        print(f"Retrieve context time: {datetime.now() - before} seconds")

    def generate_llm_answer(self):
        final_prompt = self.gen_prompt()
        for chunk in self.model.stream(final_prompt):
            yield chunk.content
            time.sleep(0.05)

    def cal_evidence(self, llm_answer) -> str:
        # Calculate score
        before = datetime.now()
        faithfulness_score = self.calculate_faithfulness(llm_answer)
        answer_relevancy_score = self.calculate_answer_relevancy(llm_answer)
        print(f"Eval answer time: {datetime.now() - before} seconds")

        context_list = [
            {"context": ctx["text"], "metadata": ctx["metadata"]}
            for ctx in self.contexts
        ]

        # evidence = (
        #    f"**Retrieved Contexts:**\n{context_str}\n\n"
        #    f"**Faithfulness Score:** {faithfulness_score:.4f}\n"
        #    f"**Answer Relevancy Score:** {answer_relevancy_score:.4f}"
        # )

        evidence = {
            "context_list": context_list,
            "f_score": faithfulness_score,
            "ar_score": answer_relevancy_score,
        }

        return evidence

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
