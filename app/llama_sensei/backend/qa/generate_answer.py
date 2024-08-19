from datasets import Dataset
from langchain_groq import ChatGroq
from llama_sensei.backend.add_courses.vectordb.document_processor import (
    DocumentProcessor,
)
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from sentence_transformers import SentenceTransformer # embedding llm
import torch
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.embeddings import SentenceTransformerEmbeddings
import re
import numpy as np
from llama_sensei.backend.add_courses.embedding.get_embedding import Embedder

MODEL = "llama3-70b-8192"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# EMBEDDING_LLM = SentenceTransformer("all-MiniLM-L12-v2", trust_remote_code=True).to(device)
EMBEDDING_LLM = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L12-v2", model_kwargs={"trust_remote_code":True})


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
            {
                "text": text,
                "metadata": metadata,
                "embedding": embedding
            }
            for text, metadata, embedding in zip(result['documents'][0], result['metadatas'][0], result['embeddings'][0])
        ]  # Store contexts for use in prompt generation and evaluation

        return self.contexts


    def gen_prompt(self, context_texts = None) -> str:
        # Extract the 'text' field from each context dictionary
        if context_texts == None:
            context_texts = [f"{ctx['text']}" for ctx in self.contexts]
        else:
            pass

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
    
    # def parse_results_string(self, results_str):
    #     # Regular expression to match the pattern for snippet, title, and link
    #     pattern = r"\[snippet:\s*(.*?),\s*title:\s*(.*?),\s*link:\s*(.*?)\]"
        
    #     # Use re.findall to extract all matches
    #     matches = re.findall(pattern, results_str)
        
    #     # Convert matches into a list of dictionaries
    #     results = []
    #     for match in matches:
    #         result_dict = {
    #             "snippet": match[0].strip(),
    #             "title": match[1].strip(),
    #             "link": match[2].strip()
    #         }
    #         results.append(result_dict)
        
    #     return results
    
    def external_search(self) -> dict:
        # search_tool = DuckDuckGoSearchResults(max_results=5)
        # result = search_tool.invoke(f"{self.query}")
        
        # # Parse the result from DuckDuckGo
        # parsed_results = self.parse_results_string(result)
        
        # # Optionally, store or process the parsed results as needed
        # return parsed_results
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
        results = search_tool.results(self.query, max_results = 5)
        
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
        faithfulness = evaluate(dataset, metrics=[faithfulness], llm = model, embeddings = EMBEDDING_LLM)
        
        # Calculate context relevancy
        relevancy_score = self.calculate_context_relevancy()

        # Convert score to pandas DataFrame and get the first score
        score_df = faithfulness.to_pandas()
        f_score = score_df[['faithfulness']].iloc[0]
        
        result = {
            'faithfulness': f_score,
            'answer_relevancy': relevancy_score
        }
        
        return result
    
    def generate_answer(self, mode = None) -> str:
        if mode == "external":
            search_results = self.external_search()
            context = [val['snippet'] for val in search_results]
            final_prompt = self.gen_prompt(context_texts = context)
            llm_answer = res['content'] if isinstance(res, dict) else res.content
            
            embedder = Embedder()  # Initialize the embedder

            # Populate self.contexts with 'text' and 'embedding'
            self.contexts = [
                {
                    "text": result['snippet'],
                    "embedding": embedder.embed(result['snippet'])[0]
                }
                for result in results
            ]

            # Calculate score
            score = self.calculate_score(llm_answer)

            context_list = [
                {"context": ctx["text"], "metadata": ctx["metadata"]}
                for ctx in self.contexts
            ]

            evidence = {
                "context_list": context_list,
                "f_score": score['faithfulness'],
                "ar_score": score['answer_relevancy'],
            }

            return llm_answer, evidence
        
        else:
            self.retrieve_contexts()
            final_prompt = self.gen_prompt(context_texts = self.contexts)
            res = self.model.invoke(final_prompt)
            # llm_answer = res['content'] if isinstance(res, dict) else res.content
            llm_answer = res['content'] if isinstance(res, dict) else res.content
            
            # Calculate score
            score = self.calculate_score(llm_answer)

            context_list = [
                {"context": ctx["text"], "metadata": ctx["metadata"]}
                for ctx in self.contexts
            ]

            evidence = {
                "context_list": context_list,
                "f_score": score['faithfulness'],
                "ar_score": score['answer_relevancy'],
            }

            return llm_answer, evidence
        
        # # Concatenate the external search result with the LLM's answer
        # if external_result:
        # # Assuming external_result is a list of dictionaries, we concatenate the snippets with links
        #     external_info = "\n".join(
        #         f"External Source {i + 1}: {item['snippet']}\nRead more: {item['link']}"
        #         for i, item in enumerate(external_result)
        #     )
        #     llm_answer = f"{llm_answer}\n\nAdditional Information:\n{external_info}"

# Example usage
if __name__ == "__main__":
    prompt = "What method do we use if we want to predict house price in an area?"
    course_name = "cs229_stanford"

    # Create an instance of GenerateRAGAnswer with the query and course
    rag_generator = GenerateRAGAnswer(query=prompt, course=course_name)

    # Generate and print the answer along with its embedded faithfulness score
    answer, evidence = rag_generator.generate_answer()
    print("answer:\n" + answer + "\nevidence:\n" + evidence)
