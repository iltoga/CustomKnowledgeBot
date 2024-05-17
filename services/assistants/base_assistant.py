import abc
import logging
import os
import shutil
from calendar import c
from datetime import datetime
from textwrap import dedent
from typing import List

import chromadb
import dotenv
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import Runnable

from helpers.llms import get_llms

dotenv.load_dotenv()


# class SuppressStdout:
#     def __enter__(self):
#         self._original_stdout = sys.stdout
#         self._original_stderr = sys.stderr
#         # sys.stdout = open(os.devnull, "w")
#         # sys.stderr = open(os.devnull, "w")

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.stdout.close()
#         sys.stdout = self._original_stdout
#         sys.stderr = self._original_stderr


class BaseAssistant(abc.ABC):

    def __init__(
        self,
        embeddings_collection_name: str,
        embeddings_llm: str,
        default_llm: str,
        small_llm: str,
        translation_llm: str,
        reset_db=False,
    ):
        self.logger = logging.getLogger(__name__)
        self.embeddings_collection_name = embeddings_collection_name
        self.stop_word = "|||terminate|||"
        self.language = "English"
        self.all_llms = get_llms()
        self.default_llm: BaseLLM = self.all_llms.get(default_llm)
        if not self.default_llm:
            raise ValueError(f"Invalid Default LLM name: {default_llm}")
        self.small_llm: BaseLLM = self.all_llms.get(small_llm)
        if not self.small_llm:
            raise ValueError(f"Invalid Small LLM name: {small_llm}")
        self.translation_llm: BaseLLM = self.all_llms.get(translation_llm)
        if not self.translation_llm:
            raise ValueError(f"Invalid Translation LLM name: {translation_llm}")
        self.embeddings = self.all_llms.get(embeddings_llm)
        self.embeddings_dir = os.getenv("EMBEDDING_DIR", "./chroma")
        self.memory = self.initialize_memory()
        self.all_splits = self.load_and_split_documents()
        self.vectorstore = self.initialize_vectorstore(
            collection_name=embeddings_collection_name,
            refresh_data=reset_db,
        )
        self.retrieval_chain = self.initialize_retrieval_chain()

    @abc.abstractmethod
    def load_and_split_documents(
        self,
        pdf_dir: str = None,
        md_dir: str = None,
        json_dir: str = None,
        websites: List[str] = None,
    ) -> List[Document]:
        pass

    @abc.abstractmethod
    def get_assistant_prompt(self):
        pass

    @abc.abstractmethod
    def initialize_retrieval_chain(self) -> Runnable:
        pass

    def translate_query(self, query: str, language: str = "English") -> str:
        """Translate the query to English."""
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    dedent(
                        """
                    Your task is to Translate user's query to {language} follolwing the rules below:

                    <rules>
                    If the language is {language}, just correct the grammar if necessary.
                    If the language is not {language}, translate the query to {language}.
                    If the query contains curse words or inappropriate content, answer with '|||invalid_content|||' (without quotes).
                    </rules>

                    Output the translated query and nothing else:
                    """
                    )
                ),
                HumanMessagePromptTemplate.from_template(query),
            ]
        )
        compiled_prompt = prompt.format(language=language)
        response = self.translation_llm.invoke(compiled_prompt)
        # in case the translation_llm is an instance of Ollama, instead of getting response.content, get response
        response_content = response.content if not isinstance(response, Ollama) else response
        self.logger.info(f"Translated query: {response_content}")
        # check if the response contains inappropriate content and return the original query
        if "|||invalid_content|||" in response_content:
            return query
        return response_content

    def enhance_query(self, query: str) -> str:
        """Enhance the query to optimize for retrieval."""
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    dedent(
                        """
                        Your task is to Enhance the user's query to optimize for retrieval, following the rules below:

                        <rules>
                        Enhance the query to make it more specific and clear, without changing the intention, meaning and goal of the original query.
                        If possible, add synonyms to increase the chances of finding the right answer.
                        If it is a follow-up question, use the chat history to enhance the query within the context of the conversation.
                        Output the enhanced query and nothing else.
                        </rules>

                        <chat_history>
                        {chat_history}
                        </chat_history>
                    """
                    )
                ),
                HumanMessagePromptTemplate.from_template(query),
            ]
        )
        response = self.small_llm.invoke(prompt.format(chat_history=self.memory.chat_memory.messages))
        response_content = response.content if hasattr(response, "content") else response
        self.logger.info(f"Enhanced query: {response_content}")
        return response_content

    @abc.abstractmethod
    def preprocess_query(self, query: str) -> str:
        pass

    @abc.abstractmethod
    def respond(self, query: str) -> str:
        pass

    def initialize_memory(self):
        # pylint: disable=maybe-no-member
        mem = ConversationTokenBufferMemory(
            memory_key="chat_history",
            input_key="input",
            llm=self.default_llm,
            max_token_limit=1000,
        )
        dt = datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
        mem.chat_memory.add_ai_message(
            f"Hi, I am Ayu 2.0, your AI assistant ready to answer your questions about RevisBali and its services. Now in Bali is {dt}. How can I help you?"
        )
        return mem

    def initialize_vectorstore(
        self,
        collection_name: str,
        refresh_data: bool = False,
    ):
        if refresh_data and os.path.exists(self.embeddings_dir):
            shutil.rmtree(self.embeddings_dir)
        os.makedirs(self.embeddings_dir, exist_ok=True)

        persistent_client = chromadb.PersistentClient(path=self.embeddings_dir)
        persistent_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        langchain_chroma = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.embeddings_dir,
            collection_metadata={"hnsw:space": "cosine"},
            client_settings={"anonymized_telemetry": False},
        )

        if refresh_data:
            return langchain_chroma.from_documents(
                client=persistent_client,
                collection_name=collection_name,
                documents=self.all_splits,
                embedding=self.embeddings,
                persist_directory=self.embeddings_dir,
                collection_metadata={"hnsw:space": "cosine"},
            )

        return langchain_chroma
