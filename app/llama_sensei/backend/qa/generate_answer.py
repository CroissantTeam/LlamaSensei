from datasets import Dataset
from langchain_groq import ChatGroq
from llama_sensei.backend.add_courses.vectordb.document_processor import DocumentProcessor
from ragas import evaluate
from ragas.metrics import (answer_relevancy, faithfulness)
import torch
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.embeddings import SentenceTransformerEmbeddings
import re
import numpy as np
from llama_sensei.backend.add_courses.embedding.get_embedding import Embedder
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "llama3-70b-8192"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# EMBEDDING_LLM = SentenceTransformer("all-MiniLM-L12-v2", trust_remote_code=True).to(device)
EMBEDDING_LLM = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L12-v2", model_kwargs={"trust_remote_code":True})


class GenerateRAGAnswer:
    def __init__(self, query: str, course: str, model=MODEL):
        self.query = query
        self.course = course
        self.model = ChatGroq(model=model, temperature=0)
        self.contexts = []  # To store the retrieved contexts

    def retrieve_contexts(self):
        processor = DocumentProcessor(self.course, search_only=True)
        result = processor.search(self.query)
        
        #print(result)
        for text, metadata, embedding in zip(result['documents'][0], result['metadatas'][0], result['embeddings'][0]):
            self.contexts.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "embedding": embedding
                })  # Store contexts for use in prompt generation and evaluation
            
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
        results = wrapper.results(self.query, max_results = 5)
        
        return results
    
    def calculate_context_relevancy(self) -> float:
        # Embed the query
        embedder = Embedder()
        embedded_query = embedder.embed(self.query)
        
        # Retrieve the embeddings
        similarity_scores = []

        for context in self.contexts:
            context_embedding = np.asarray(context["embedding"]).reshape(1, -1)

            # Calculate the cosine similarity
            dot_product = np.dot(context_embedding, embedded_query.T).reshape(-1)
            norm_product = np.linalg.norm(context_embedding) * np.linalg.norm(embedded_query)
            similarity = dot_product / norm_product

            similarity_scores.append(similarity.item())

        # Calculate and return the mean of the similarity scores
        if similarity_scores:
            return np.mean(similarity_scores)
        else:
            return 0.0

    def calculate_score(self, generated_answer: str) -> float:
        if not self.contexts:
            raise ValueError(
                "Contexts have not been retrieved. Ensure contexts are retrieved before this method is called."
            )

        model = self.model

        # Prepare the dataset for evaluation
        data_samples = {
            'question': [self.query],
            'answer': [generated_answer],
            'contexts': [[val["text"] for val in self.contexts]]
        }
        dataset = Dataset.from_dict(data_samples)

        # Evaluate faithfulness
        score = evaluate(dataset, metrics=[faithfulness], llm = model, embeddings = EMBEDDING_LLM)
        
        # Calculate context relevancy
        relevancy_score = self.calculate_context_relevancy()

        # Convert score to pandas DataFrame and get the first score
        score_df = score.to_pandas()
        f_score = score_df[['faithfulness']].iloc[0, 0]
        
        result = {
            'faithfulness': f_score,
            'answer_relevancy': relevancy_score
        }
        
        return result
    
    def rank_and_select_top_contexts(internet_contexts, database_contexts, top_n=5):
        # Combine both contexts into one list for comparison
        all_contexts = internet_contexts + database_contexts
        
        # Extract the embeddings
        all_embeddings = [context['embedding'] for context in all_contexts]
        
        # Compute cosine similarity between all contexts
        similarity_matrix = cosine_similarity(all_embeddings, all_embeddings)
        
        # Rank each context by summing its similarities with all other contexts
        similarity_sums = similarity_matrix.sum(axis=1)
        
        # Get the indices of the top N contexts based on similarity sum
        top_indices = similarity_sums.argsort()[-top_n:][::-1]
        
        # Collect the top contexts based on the computed indices, including their text and metadata
        top_contexts = [all_contexts[index] for index in top_indices]

        return top_contexts
    
    def generate_answer(self, indb: bool, internet: bool) -> str:
        if internet:
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

        if indb:
            self.retrieve_contexts()
            
        if internet and indb:
            search_results = self.external_search()
            # Populate self.contexts with 'text' and 'embedding'
            embedder = Embedder()  # Initialize the embedder
            internet_contexts = [
                {
                    "text": result['snippet'],
                    "metadata": {"link": result['link']},
                    "embedding": embedder.embed(result['snippet'])
                }
                for result in search_results
            ]
            
            database_contexts = self.retrieve_contexts()
            self.contexts = self.rank_and_select_top_contexts(internet_contexts, database_contexts, top_n=5)
        
        final_prompt = self.gen_prompt()
        res = self.model.invoke(final_prompt)
        # llm_answer = res['content'] if isinstance(res, dict) else res.content
        llm_answer = res['content'] if isinstance(res, dict) else res.content

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

        return llm_answer, evidence

# Example usage
if __name__ == "__main__":
    prompt = "What method do we use if we want to predict house price in an area?"
    course_name = "cs229_stanford"

    # Create an instance of GenerateRAGAnswer with the query and course
    rag_generator = GenerateRAGAnswer(query=prompt, course=course_name)

    # Generate and print the answer along with its embedded faithfulness score
    answer, evidence = rag_generator.generate_answer()
    print("answer:\n" + answer)
    #print(evidence)
