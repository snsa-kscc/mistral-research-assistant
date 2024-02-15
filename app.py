from fastapi import FastAPI
from research_agent import flatten_chain as chain
from langserve import add_routes
import uvicorn

app = FastAPI(
    title="LangChain research assistant",
    version="1.0",
    description="A simple api server using LangChain",
)

add_routes(
    app,
    chain,
    path="/assistant",
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
