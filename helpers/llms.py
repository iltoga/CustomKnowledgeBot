import os
from typing import Dict, Union

from dotenv import load_dotenv
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_groq import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()


def get_llms() -> Dict[str, Union[Ollama, ChatGroq, ChatOpenAI]]:
    return {
        "ollama_local": Ollama(
            model=os.getenv("OLLAMA_MODEL", "zephyr"),
            num_ctx=4096,
            temperature=0.1,
            repeat_penalty=1.5,
            top_k=10,
            top_p=0.6,
        ),
        # "ollama_local": Ollama(
        #     model=os.getenv("OLLAMA_MODEL", "zephyr"),
        #     num_ctx=4096,
        #     temperature=0.1,
        #     repeat_penalty=1.5,
        #     top_k=10,
        #     top_p=0.6,
        # ),
        "ollama_local_uncensored": Ollama(model="dolphin-llama3:8b-256k", num_ctx=4096, temperature=0.8),
        "groq_big": ChatGroq(
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
        ),
        "groq_small": ChatGroq(
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192",
        ),
        "openai_small": ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            # temperature=0.3,
        ),
        "openai_big_old": ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-"),
            # temperature=0.3,
        ),
        "openai_big": ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            # temperature=0.3,
        ),
        "openai_embeddings": OpenAIEmbeddings(model="text-embedding-3-large", show_progress_bar=True),
        "ollama_embeddings": OllamaEmbeddings(model="mxbai-embed-large", show_progress=True),
    }
