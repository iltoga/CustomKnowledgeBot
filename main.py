import json
import os
import re
import sys
from typing import Optional

import dotenv
from icecream import ic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.document_loaders.pdf import OnlinePDFLoader, PyPDFDirectoryLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

dotenv.load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "zephyr")
PROMPt_TEMPLATE = os.getenv("PROMPT_TEMPLATE", "prompts/zephyr_ayu.txt")


class GenerationStatisticsCallback(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(response.generations[0][0].generation_info)


class PreciseOllama(Ollama):
    def __init__(self, stop: Optional[str] = None, **kwargs):
        super().__init__(
            **kwargs,
            temperature=0.2,
            top_k=10,
            repeat_penalty=1.5,
            top_p=0.6,
            num_ctx=4096,
        )
        self.stop = stop


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


if __name__ == "__main__":
    stop_word = "terminate"
    # load the pdf and split it into chunks
    loader = PyPDFDirectoryLoader(path="./knowledge_base", recursive=True)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    # all_splits = loader.load_and_split(text_splitter=text_splitter)
    all_splits = loader.load_and_split()

    with SuppressStdout():
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

    while True:
        query = input("\nQuery: ")
        if query == stop_word:
            print(f"Terminating as '{stop_word}' was entered.")
            break
        if query.strip() == "":
            continue

        # Prompt (load the template from a file: prompts/zephyr.txt)
        file_path = "prompts/zephyr_ayu.txt"

        # Open the file using 'with' to ensure proper resource management
        with open(file_path, encoding="utf-8") as file:
            template = file.read()
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        callback_manager = CallbackManager([StdOutCallbackHandler()])

        # llm = PreciseOllama(
        #     stop=stop_word,
        #     base_url="http://localhost:11434",
        #     model=LLM_MODEL,
        #     verbose=True,
        #     callback_manager=callback_manager,
        # )
        llm = Ollama(model=LLM_MODEL, num_ctx=4096)

        qa_chain = RetrievalQA.from_chain_type(
            llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        res = qa_chain({"query": query})
        result = res["result"]
        print(result)
