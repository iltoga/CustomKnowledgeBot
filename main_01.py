import os
import shutil
import sys
from datetime import datetime
from textwrap import dedent
from typing import List

import dotenv
import nest_asyncio
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from helpers.md_directoryloader import MdDirectoryLoader
from helpers.md_textsplitter import MdTextSplitter
from helpers.nltk_textsplitter import CustomNLTKTextSplitter

dotenv.load_dotenv()


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


def load_pdf_splits(knowledge_dir: str = "./knowledge_base/knowledge_pdf_files") -> List:
    """
    Load PDFs from a directory and print their content split by pages.

    Args:
        knowledge_dir (str): The directory to load PDFs from. Defaults to "./knowledge_base/revisbali".

    Returns:
        List: A list of all splits.
    """
    loader = PyPDFDirectoryLoader(path=knowledge_dir, recursive=True)
    splits = loader.load_and_split()

    # for split in splits:
    #     print(f"{split.page_content}\n\n")

    return splits


from typing import List


def load_web_content(knowledge_url: List[str] = ["https://www.revisbali.com"]) -> List:
    """
    Load web content from a URL and print its content.

    Args:
        knowledge_url (str): The URL to load web content from. Defaults to "http://example.com".

    Returns:
        List: A list of all content.
    """
    loader = WebBaseLoader(knowledge_url)
    splits = loader.load_and_split()

    # for split in splits:
    #     print(f"{split.page_content}\n\n")

    return splits


def load_md_splits(knowledge_dir: str = "./knowledge_base/knowledge_markdown_files") -> List:
    """
    Load web content from a URL and print its content.

    Args:
        knowledge_url (str): The URL to load web content from. Defaults to "http://example.com".

    Returns:
        List: A list of all content.
    """
    loader = MdDirectoryLoader(knowledge_dir)
    splits = loader.load_and_split(text_splitter=MdTextSplitter())

    # for split in splits:
    #     print(f"{split.page_content}\n\n")

    return splits


if __name__ == "__main__":
    stop_word = "terminate"
    language = "English"

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "zephyr")
    llm_local = Ollama(model=OLLAMA_MODEL, num_ctx=409, temperature=0.1, repeat_penalty=1.5, top_k=10, top_p=0.6)

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    llm_local_uncensored = Ollama(model="dolphin-llama3:8b-256k", num_ctx=4096, temperature=0.8)
    llm_groq_big = ChatGroq(
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
    )
    llm_groq_small = ChatGroq(
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
    )
    llm_openai_small = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0.1,
    )
    llm_openai_big_old = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
        temperature=0.1,
    )
    llm_openai_big = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0,
    )
    selected_llm = llm_groq_big

    # delete the existing embedding directory
    embedding_dir = os.getenv("EMBEDDING_DIR", "./chroma")
    if os.path.exists(embedding_dir):
        shutil.rmtree(embedding_dir)
    # embedding = GPT4AllEmbeddings()
    # embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    # embedding = OllamaEmbeddings(model="mxbai-embed-large")
    # embedding = OllamaEmbeddings(model="nomic-embed-text")
    # embedding = OllamaEmbeddings(model="snowflake-arctic-embed")
    all_splits = []
    KNOWLEDGE_MD_DIR = os.getenv("KNOWLEDGE_MD_DIR", "./knowledge_base/knowledge_markdown_files")
    # all_splits = load_pdf_splits(KNOWLEDGE_DIR)
    # KNOWLEDGE_URLS = ["https://www.revisbali.com"]
    # all_splits.extend(load_web_content(KNOWLEDGE_URLS))
    all_splits.extend(load_md_splits(KNOWLEDGE_MD_DIR))

    with SuppressStdout():
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=embedding_dir)

    # retrieval_qa_chat_prompt: ChatPromptTemplate = hub.pull("langchain-ai/retrieval-qa-chat")
    retrieval_qa_chat_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                dedent(
                    """
                        Act as Auy, AI RevisBali's customer service assistant.
                        Your area of expertise is law and regulations in Indonesia, specifically in the context of immigration, visa, company establishment and management, real estate and related services.
                        Your task is to help customers with their questions in your area of expertise, while marketing at your best ReviseBali and its services, when appropriate and applicable.

                        Follow the rules below to generate the best answers for the user's questions:
                        <rules>
                        - You can only answer in {language} language (don't change language even if the user asks questions in a different one).
                        - You can only answer questions based on the context (knowledge) provided.
                        - Answer in the most concise way possible unless specifically asked by the user to elaborate on something.
                        - Provide the most relevant information first, then expand if necessary or specifically asked.
                        - When providing multiple options or steps, always use numbered lists. If there are multiple lists, add a title to each list to differentiate them.
                        - Add link to the official RevisBali website, email and whatsapp number when appropriate.
                        - For complex or generic questions, think step-by-step and try to breakdown the question in steps to clarify the user's needs, by asking the user provide more details.
                        - For straighforward and simple questions, answer directly, if you have what you need in the knowledge.
                        - Try to not repeat yourself and keep the conversation flow natural (look at the chat_history to avoid repetition).
                        - When the conversation is over, answer with "{stop_word}" and nothing else.
                        </rules>

                        Use the chat history below to keep track of the conversation and keep a natural conversation flow:
                        <chat_history>
                        {chat_history}
                        </chat_history>

                        Answer any question based SOLELY on the knowledge below. Don't use your internal training to answer and don't try to make the answer up:
                        <knowledge>
                        {context}
                        </knowledge>

                        Keep it short and simple and straight to the point. don't repeat yourself and don't provide extra information not strictly relevant to the question.
                        If the question is not related to the context in above knowledge or your area of expertise, just answer: "I'm sorry, I don't have that information. Please try rephrasing your question. I remind you I can only answer to questions related to RevisBali and its services." (in {language} language).
                        """
                ),
            ),
            # The `variable_name` here is what must align with memory
            # MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    # retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             dedent(
    #                 """
    #                     Act as Auy, a polite and helpful customer service assistance working for RevisBali Visa Agency.
    #                     Use the shortest anwer possible and use two sentences maximum. Always Keep the answer as concise as possible.
    #                     Use ONLY the following pieces of context to answer the user's question and never fabricate them.
    #                     If you don't know the answer, just say so, don't try to make up an answer.
    #                     For simple questions, just answer with a few words.
    #                     Don't disclose that you are a machine, AI, or a chatbot.
    #                     Answer any use questions based solely on the context below:
    #                     <context>
    #                     {context}
    #                     </context>

    #                     This is the chat history so far. you can use this to generate a better response:
    #                     <chat>
    #                     {memory}
    #                     </chat>
    #                     """
    #             ),
    #         ),
    #         ("human", "Hello, how are you doing?"),
    #         ("ai", "I'm doing well, thanks!"),
    #         ("human", "{question}"),
    #     ]
    # )

    # # Prompt (load the template from a file: prompts/zephyr.txt)
    # file_path = "prompts/zephyr_ayu.txt"
    # # Open the file using 'with' to ensure proper resource management
    # with open(file_path, encoding="utf-8") as file:
    #     template = file.read()
    # QA_CHAIN_PROMPT = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template=template,
    # )

    # memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
    memory = ConversationTokenBufferMemory(
        memory_key="chat_history",
        input_key="input",
        llm=selected_llm,
        max_token_limit=1000,
    )
    dt = datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
    memory.chat_memory.add_ai_message(
        f"Hi, I am Ayu 2.0, your AI assistant ready to answer your questions about RevisBali and its services. Now in Bali is {dt}. How can I help you?"
    )

    print(memory.buffer)
    while True:
        query = input("\nQuery: ")
        if query == stop_word:
            print(f"Terminating as '{stop_word}' was entered.")
            break
        if query.strip() == "":
            continue

        # Update memory with the new user query
        memory.chat_memory.add_user_message(query)

        # callback_manager = CallbackManager([StdOutCallbackHandler()])
        # llm = PreciseOllama(
        #     stop=stop_word,
        #     base_url="http://localhost:11434",
        #     model=OLLAMA_MODEL,
        #     verbose=True,
        #     callback_manager=callback_manager,
        # )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 80},
        )
        combine_docs_chain = create_stuff_documents_chain(selected_llm, retrieval_qa_chat_prompt)

        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        res = retrieval_chain.invoke(
            {
                "input": query,
                "chat_history": memory.chat_memory.messages,
                "language": language,
                "stop_word": stop_word,
                # "context": all_splits,
            }
        )
        if res is not None and len(res["context"]) > 0:
            print("Context:\n")
            for i in range(len(res["context"])):
                print(f"Document {i}: {res['context'][i]}\n")
            # summarize the result
            # summary = llm_groq_small.invoke(
            #     input=dedent(
            #         f"""
            #         Summarize and rephrase the context focusing on punctually answering the question, avoiding extra information that are not strictly relevant to the question: {query}
            #         <context>
            #         {res["answer"]}
            #         </context>

            #         Output the answer to the question and nothing else.
            #         """
            #     ),
            #     stop=stop_word,
            # )
            # memory.chat_memory.add_ai_message(summary.content)
            # print(summary.content)
            memory.chat_memory.add_ai_message(res.get("answer"))
            print("\n\nAnswer:\n")
            print(memory.buffer)

        # qa_chain = RetrievalQA.from_chain_type(
        #     llm_local_uncensored, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        # )

        # res = qa_chain({"query": query})
        # result = res["result"]
        # print(result)
