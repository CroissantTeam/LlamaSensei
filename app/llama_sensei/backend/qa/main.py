import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from generate_answer import GenerateRAGAnswer
from schemas import ChatResponse, EvaluationRequest, EvaluationResponse, Question

load_dotenv()
CONTEXT_SEARCH_API_URL = f'{os.getenv("COURSE_API_URL")}/search'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
app = FastAPI(title="LlamaSensei: Chat API")


@app.post("/generate_answer", response_model=ChatResponse)
async def api_generate_answer(question: Question):
    rag_chain = GenerateRAGAnswer(
        course=question.course,
        context_search_url=CONTEXT_SEARCH_API_URL,
        groq_api_key=GROQ_API_KEY,
    )
    rag_chain.prepare_context(
        indb=question.indb,
        internet=question.internet,
        query=question.question,
    )
    return StreamingResponse(rag_chain.generate_llm_answer())


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_answer(request: EvaluationRequest):
    rag_chain = GenerateRAGAnswer(
        course=request.course_name,
        context_search_url=CONTEXT_SEARCH_API_URL,
        groq_api_key=GROQ_API_KEY,
    )
    evidence = rag_chain.run_evaluation(
        query=request.query,
        answer=request.answer,
        contexts=request.contexts,
    )
    return EvaluationResponse(**evidence)


@app.get("/")
async def root():
    return {"message": "Welcome to LlamaSensei: Chat API"}


def main():
    # Run web server with uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("CHAT_FASTAPI_HOST", "127.0.0.1"),
        port=int(os.getenv("CHAT_FASTAPI_PORT", 8001)),
        # reload=True,  # Uncomment this for debug
        # workers=2,
    )


if __name__ == "__main__":
    main()
# Add more endpoints as needed
