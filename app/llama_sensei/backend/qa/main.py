import os

import uvicorn
from fastapi import FastAPI
from generate_answer import GenerateRAGAnswer
from schemas import ChatResponse, Question

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Llama Sensei API"}


@app.post("/generate_answer", response_model=ChatResponse)
async def api_generate_answer(question: Question):
    query = GenerateRAGAnswer(question.question, question.course)
    print((question.question, question.course))
    # time.sleep(20)
    answer, evidence = query.generate_answer()
    return ChatResponse(answer=answer, evidence=evidence)


def main():
    # Run web server with uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("GEN_ANSWER_FASTAPI_HOST", "127.0.0.1"),
        port=int(os.getenv("GEN_ANSWER_FASTAPI_PORT", 8000)),
        reload=True,  # Uncomment this for debug
        # workers=2,
    )


if __name__ == "__main__":
    main()
# Add more endpoints as needed
