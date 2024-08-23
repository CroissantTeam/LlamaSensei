import time
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_groq import ChatGroq
from llama_sensei.backend.add_courses.vectordb.document_processor import (
    DocumentProcessor,
)
from ragas import evaluate
from ragas.metrics import faithfulness
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "llama3-70b-8192"
EMBEDDING_LLM = "all-MiniLM-L12-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GenerateRAGAnswer:
    def __init__(self, course: str, model=MODEL):
        self.query = ""
        self.course = course
        self.embedder = SentenceTransformer(EMBEDDING_LLM, trust_remote_code=True).to(
            DEVICE
        )
        self.model = ChatGroq(model=model, temperature=0)
        self.contexts = []  # To store the retrieved contexts

    def retrieve_contexts(self):
        processor = DocumentProcessor(self.course, search_only=True)
        result = processor.search(self.query)

        for text, metadata, embedding in zip(
            result['documents'][0], result['metadatas'][0], result['embeddings'][0]
        ):
            self.contexts.append(
                {"text": text, "metadata": metadata, "embedding": embedding}
            )  # Store contexts for use in prompt generation and evaluation

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

    def external_search(self) -> dict:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        results = wrapper.results(self.query, max_results=5)

        return results

    def calculate_context_relevancy(self) -> float:
        # Embed the query
        embedded_query = self.embedder.encode(self.query)

        # Retrieve the embeddings
        similarity_scores = []

        for context in self.contexts:
            context_embedding = np.asarray(context["embedding"]).reshape(1, -1)

            # Calculate the cosine similarity
            dot_product = np.dot(context_embedding, embedded_query.T).reshape(-1)
            norm_product = np.linalg.norm(context_embedding) * np.linalg.norm(
                embedded_query
            )
            similarity = dot_product / norm_product

            similarity_scores.append(similarity.item())

        # Calculate and return the mean of the similarity scores
        if similarity_scores:
            return np.mean(similarity_scores)
        else:
            return 0.0

    def calculate_score(self, generated_answer: str) -> float:
        if not self.contexts:
            print(
                "Contexts have not been retrieved. Ensure contexts are retrieved before this method is called."
            )
            return {'faithfulness': 0, 'answer_relevancy': 0}

        model = self.model

        # Prepare the dataset for evaluation
        data_samples = {
            'question': [self.query],
            'answer': [generated_answer],
            'contexts': [[val["text"] for val in self.contexts]],
        }
        dataset = Dataset.from_dict(data_samples)

        # Evaluate faithfulness
        score = evaluate(
            dataset, metrics=[faithfulness], llm=model, embeddings=self.embedder
        )

        # Calculate context relevancy
        relevancy_score = self.calculate_context_relevancy()

        # Convert score to pandas DataFrame and get the first score
        score_df = score.to_pandas()
        f_score = score_df[['faithfulness']].iloc[0, 0]

        result = {'faithfulness': f_score, 'answer_relevancy': relevancy_score}

        return result

    def rank_and_select_top_contexts(self, top_n=5):
        # Extract the embeddings
        all_embeddings = [context['embedding'] for context in self.contexts]

        # Compute cosine similarity between all contexts
        similarity_matrix = cosine_similarity(all_embeddings, all_embeddings)

        # Rank each context by summing its similarities with all other contexts
        similarity_sums = similarity_matrix.sum(axis=1)

        # Get the indices of the top N contexts based on similarity sum
        top_indices = similarity_sums.argsort()[-top_n:][::-1]

        # Collect the top contexts based on the computed indices, including their text and metadata
        top_contexts = [self.contexts[index] for index in top_indices]

        return top_contexts

    def prepare_context(self, indb: bool, internet: bool, query: str) -> str:
        self.query = query
        self.contexts = []
        before = datetime.now()
        if internet:
            search_results = self.external_search()
            # Populate self.contexts with 'text' and 'embedding'
            self.contexts = [
                {
                    "text": result['snippet'],
                    "metadata": {"link": result['link']},
                    "embedding": self.embedder.encode(result['snippet']),
                }
                for result in search_results
            ]

        if indb:
            self.retrieve_contexts()

        if indb or internet:
            self.contexts = self.rank_and_select_top_contexts(top_n=5)

        print(f"Retrieve context time: {datetime.now() - before} seconds")

    def generate_llm_answer(self):
        final_prompt = self.gen_prompt()
        for chunk in self.model.stream(final_prompt):
            yield chunk.content
            time.sleep(0.05)

    def cal_evidence(self, llm_answer) -> str:
        # Calculate score
        before = datetime.now()

        context_list = [
            {"context": ctx["text"], "metadata": ctx["metadata"]}
            for ctx in self.contexts
        ]

        # Calculate score
        score = self.calculate_score(llm_answer)

        evidence = {
            "context_list": context_list,
            "f_score": score['faithfulness'],
            "ar_score": score['answer_relevancy'],
        }

        print(f"Eval answer time: {datetime.now() - before} seconds")
        return evidence
