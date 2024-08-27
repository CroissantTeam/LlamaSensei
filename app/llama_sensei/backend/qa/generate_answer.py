import json
from datetime import datetime

import numpy as np
import requests
import torch
from datasets import Dataset
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import faithfulness
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "llama3-70b-8192"
EMBEDDING_LLM = "all-MiniLM-L12-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GenerateRAGAnswer:
    """
    Provides a mechanism to generate answers to queries using a retrieval-augmented generation approach.
    This class supports both internet-based and internal database context retrieval, which are then used
    to generate responses through a large language model.

    Attributes:
        course (str): Course identifier for context retrieval from internal databases.
        model (str): Identifier for the large language model used for generating answers.
        embedder (SentenceTransformer): Transformer model used to compute embeddings for context relevance.
        model (ChatGroq): Instance of the large language model initialized with specified model parameters.
        contexts (list): List storing retrieved contexts along with metadata and embeddings.

    Methods:
        retrieve_contexts: Retrieves contexts from internal database based on the current query.
        gen_prompt: Generates a prompt for the language model using retrieved or searched contexts.
        external_search: Performs an internet search to retrieve contexts when internal data is insufficient.
        calculate_context_relevancy: Calculates the relevancy of retrieved contexts to the query using cosine similarity.
        calculate_score: Evaluates the generated answer based on faithfulness and relevance.
        rank_and_select_top_contexts: Selects the top relevant contexts based on their similarity scores.
        prepare_context: Prepares the necessary contexts for answering a query, from either internal or external sources.
        generate_llm_answer: Generates an answer from the language model using the prepared prompt.
        cal_evidence: Compiles evidence of the generated answer's quality and relevancy.
    """

    def __init__(self, course: str, context_search_url: str, model=MODEL):
        """
        Initializes the GenerateRAGAnswer instance with specified course and model settings.

        Parameters:
            course (str): The identifier for the course to retrieve contextual data from.
            model (str): The model identifier for the language model used in answer generation.
        """
        self.query = ""
        self.contexts = []  # To store the retrieved contexts
        self.course = course
        self.context_search_url = context_search_url
        self.embedder = SentenceTransformer(EMBEDDING_LLM, trust_remote_code=True).to(
            DEVICE
        )
        self.model = ChatGroq(model=model, temperature=0)

    def retrieve_contexts(self, top_k=5):
        search_query = {
            "course_name": self.course,
            "text": self.query,
            "top_k": top_k,
        }
        try:
            r = requests.post(url=self.context_search_url, json=search_query)
            response = r.json()

            self.contexts.extend(
                [
                    {
                        "text": text,
                        "metadata": metadata,
                        "embedding": embedding,
                        "is_internal": True,
                    }
                    for text, metadata, embedding in zip(
                        response['documents'],
                        response['metadatas'],
                        response['embeddings'],
                    )
                ]
            )
            return self.contexts
        except Exception:
            raise

    def gen_prompt(self) -> str:
        """
        Constructs a detailed prompt from the retrieved or searched contexts to be processed by the language model.

        Returns:
            str: A formatted string that serves as a prompt for the language model.
        """
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
        """
        Conducts an internet search using DuckDuckGo API to find relevant contexts when internal data is insufficient.

        Returns:
            dict: A dictionary containing snippets of text and metadata from the search results.
        """
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        results = wrapper.results(self.query, max_results=5)

        return results

    def calculate_context_relevancy(self) -> float:
        """
        Calculates the average cosine similarity between the query's embedding and the embeddings of retrieved contexts.

        Returns:
            float: The average cosine similarity score indicating relevancy of contexts to the query.
        """
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

        return similarity_scores

    def calculate_score(self, generated_answer: str) -> float:
        """
        Computes the faithfulness and relevancy scores for the generated answer based on provided contexts.

        Parameters:
            generated_answer (str): The answer generated by the language model to evaluate.

        Returns:
            dict: A dictionary containing 'faithfulness' and 'answer_relevancy' scores.
        """
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

        # Convert score to pandas DataFrame and get the mean score
        score_df = score.to_pandas()
        f_score = score_df['faithfulness'].tolist()

        result = {'faithfulness': f_score, 'context_relevancy': relevancy_score}

        return result

    def rank_and_select_top_contexts(self, top_n=5):
        """
        Ranks and selects the top contexts based on their cosine similarity scores.

        Parameters:
            top_n (int): Number of top contexts to select.

        Returns:
            list: The top contexts selected based on their overall relevance.
        """

        # Embed the query and reshape it to 2D array
        embedded_query = self.embedder.encode(self.query).reshape(1, -1)

        # Extract the embeddings of the contexts and ensure they are in a 2D array
        all_embeddings = [
            np.array(context['embedding']).reshape(1, -1) for context in self.contexts
        ]
        all_embeddings = np.vstack(
            all_embeddings
        )  # Stack to create a 2D array of all embeddings

        # Compute cosine similarity between the query embedding and each context embedding
        similarity_scores = cosine_similarity(embedded_query, all_embeddings)[0]

        # Get the indices of the top N contexts based on similarity scores
        top_indices = similarity_scores.argsort()[-top_n:][::-1]

        # Collect the top contexts based on the computed indices, including their text and metadata
        top_contexts = [self.contexts[index] for index in top_indices]

        return top_contexts

    def prepare_context(self, indb: bool, internet: bool, query: str) -> str:
        """
        Prepares necessary contexts by either retrieving from the internal database or searching on the internet.

        Parameters:
            indb (bool): Flag to indicate if contexts should be retrieved from the internal database.
            internet (bool): Flag to indicate if contexts should be searched on the internet.
            query (str): The query for which contexts are being prepared.
        """
        self.query = query
        self.contexts = []
        before = datetime.now()
        if internet:
            search_results = self.external_search()
            # Populate self.contexts with 'text' and 'embedding'
            self.contexts.extend(
                [
                    {
                        "text": result['snippet'],
                        "metadata": {"link": result['link']},
                        "embedding": self.embedder.encode(result['snippet']).tolist(),
                        "is_internal": False,
                    }
                    for result in search_results
                ]
            )

        if indb:
            self.retrieve_contexts()

        if indb or internet:
            self.contexts = self.rank_and_select_top_contexts(top_n=5)

        print(f"Retrieve context time: {datetime.now() - before} seconds")

    async def generate_llm_answer(self):
        """
        Generates an answer from the language model by streaming content based on the prepared prompt.

        Yields:
            str: Each content chunk generated by the language model.
        """
        final_prompt = self.gen_prompt()
        is_sent_context = True
        for chunk in self.model.stream(final_prompt):
            if is_sent_context:  # only send context along with first token
                context = self.contexts
                is_sent_context = False
            else:
                context = None
            json_text = json.dumps({"token": chunk.content, "context": context})
            yield json_text + "\n"

    def cal_evidence(self, llm_answer) -> str:
        """
        Calculates and compiles evidence regarding the quality and relevancy of the language model's answer.

        Parameters:
            llm_answer (str): The generated answer to evaluate.

        Returns:
            dict: A dictionary containing the list of contexts used and their respective quality scores.
        """
        # Calculate score
        before = datetime.now()

        scores = self.calculate_score(llm_answer)

        evidence_list = [
            {
                "context": ctx["text"],
                "metadata": ctx["metadata"],
                "is_internal": ctx["is_internal"],
                "f_score": scores['faithfulness'][0],
                "cr_score": scores['context_relevancy'][i],
            }
            for i, ctx in enumerate(self.contexts)
        ]

        evidence = {
            "context_list": evidence_list,
            "f_scores": [
                str(round(evidence["f_score"] * 100, 2)) for evidence in evidence_list
            ],
            "cr_scores": [
                str(round(evidence["cr_score"] * 100, 2)) for evidence in evidence_list
            ],
        }

        print(f"Eval answer time: {datetime.now() - before} seconds")
        return evidence

    def run_evaluation(self, query, answer, contexts):
        self.query = query
        self.contexts = contexts
        evidence = self.cal_evidence(answer)
        del evidence["context_list"]
        return evidence
