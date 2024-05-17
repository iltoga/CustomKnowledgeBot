import os
from textwrap import dedent
from typing import List

import dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from helpers.knowledge_loader import KnowledgeLoader
from services.assistants.base_assistant import BaseAssistant

dotenv.load_dotenv()


class RevisBaliAssistant(BaseAssistant):

    def __init__(
        self,
        reset_db=False,
        default_llm="openai_big",
        small_llm="openai_small",
        translation_llm="openai_small",
        embeddings_llm="openai_embeddings",
    ):
        super().__init__(
            embeddings_collection_name="revisbali",
            reset_db=reset_db,
            default_llm=default_llm,
            small_llm=small_llm,
            translation_llm=translation_llm,
            embeddings_llm=embeddings_llm,
        )

    def load_and_split_documents(
        self,
        pdf_dir: str = None,
        md_dir: str = None,
        json_dir: str = None,
        websites: List[str] = None,
    ) -> List[Document]:
        if websites is None:
            websites = []
        splits = []

        if pdf_dir is None:
            pdf_dir = "./knowledge_base/knowledge_pdf_files"
        docs = KnowledgeLoader.load_pdf_splits(pdf_dir)
        splits.extend(docs)

        if websites:
            docs = KnowledgeLoader.load_web_content(websites)
            splits.extend(docs)

        if md_dir is None:
            md_dir = "./knowledge_base/knowledge_markdown_files"
        docs = KnowledgeLoader.load_md_splits(md_dir)
        splits.extend(docs)

        if json_dir is None:
            json_dir = "./knowledge_base/knowledge_json_files"
        docs = KnowledgeLoader.load_json_splits(json_dir)
        splits.extend(docs)

        self.logger.info(f"Loaded {len(splits)} documents")
        # if logging level is debug, print the content of the loaded documents

        if self.logger.level == 10:
            for i in enumerate(len(splits)):
                self.logger.debug(f"Document {i}: {splits[i]}")

        return splits

    def get_assistant_prompt(self):
        return ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    dedent(
                        """
                            Act as Ayu, AI RevisBali's customer service assistant.
                            Your expertise is Law and Regulations in Indonesia (especially about immigration, visa, real estate, and related services in Bali and Indonesia)
                            and business services (company registration, business consulting, and related services in Bali and Indonesia).
                            Your Task is to help customers with their questions, while marketing RevisBali and its services when appropriate.

                            Follow these rules for generating the best answer to the user's question:
                            <rules>
                            - Answer only in {language} (don't change even if asked in a different language).
                            - Answer based only on the provided context (knowledge below) and not from internal training data.
                            - Try to answer the user's question in one sentence, unless the question requires multiple steps or options.
                            - Be concise and just answer the user's question with the most relevand information first without adding extra information, unless the user asks for elaboration or a detailed answer.
                            - Provide the most relevant information first, then ask if the user needs more details.
                            - Use numbered lists for multiple options or steps, and add titles to differentiate multiple lists.
                            - Add a link to the official RevisBali website, email, and WhatsApp number when appropriate, but don't repeat them during the conversation, unless strictly necessary.
                            - For complex or generic questions, break down the question and ask the user for more details if needed.
                            - For straightforward questions, answer directly if the knowledge is sufficient.
                            - Avoid repetition and maintain a natural conversation flow by referring to the chat history.
                            - If the user wants to speak to a human, provide RevisBali's phone number and email and end the conversation.
                            - If there are no more details to add to your answer (based on the context), ask the user if he needs help with anything else.
                            - If the conversation is over, answer politely and end with "{stop_word}".
                            </rules>

                            Use the chat history to keep the conversation natural:
                            <chat_history>
                            {chat_history}
                            </chat_history>

                            Answer questions based solely on the knowledge provided. Do not use internal training data or fabricate answers:
                            <knowledge>
                            {context}
                            </knowledge>

                            Keep it short, simple, and to the point. Avoid repetition and irrelevant information.
                            If the question is outside the provided knowledge or expertise, respond: "I'm sorry, I don't have that information. Please try rephrasing your question. I remind you I can only answer questions related to RevisBali and its services." (in {language}).
                            If above knowledge is empty, respond: "I'm sorry, I don't have that information. Please try rephrasing your question or change topic." (in {language}).
                            Format your answer as telegram message, with a beautiful and readable style, adding emojis and links when appropriate.
                            Answer in one short paragraph, unless the question requires multiple steps or options.
                        """
                    ),
                ),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

    def initialize_retrieval_chain(self):
        retrieval_qa_chat_prompt = self.get_assistant_prompt()
        # retriever = self.vectorstore.as_retriever(
        #     search_type="mmr",
        #     search_kwargs={"k": 5, "fetch_k": 50},
        # )
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 40, "fetch_k": 60},
        )
        # Reraank the documents
        compressor = FlashrankRerank(top_n=10, model=os.getenv("FLASHRANK_MODEL", "ms-marco-MultiBERT-L-12"))
        # compressor = CohereRerank(
        #     cohere_api_key=os.getenv("COHERE_API_KEY"),
        #     top_n=5,
        # )
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        combine_docs_chain = create_stuff_documents_chain(self.default_llm, retrieval_qa_chat_prompt)
        ret_chain = create_retrieval_chain(compression_retriever, combine_docs_chain)
        return ret_chain

    def preprocess_query(self, query: str) -> str:
        res = self.translate_query(query, "English")
        # res = self.enhance_query(res)
        return res

    def respond(self, query: str, optimize_query=True) -> str:
        # pylint: disable=maybe-no-member
        if optimize_query:
            query = self.preprocess_query(query)
        self.memory.chat_memory.add_user_message(query)

        res = self.retrieval_chain.invoke(
            {
                "input": query,
                "chat_history": self.memory.chat_memory.messages,
                "language": self.language,
                "stop_word": self.stop_word,
                # "context": self.all_splits,
            }
        )

        if res is not None and len(res["context"]) > 0:
            self.logger.info("Context:\n")
            for i in range(len(res["context"])):
                self.logger.info(f"Document {i}: {res['context'][i]}\n")
            self.memory.chat_memory.add_ai_message(res.get("answer"))
            # print(self.memory.buffer)
            ai_answer = res.get("answer")
            return ai_answer

        return "Mmmh, something went wrong and I wasn't able to elaborate the answer. Please try again. Try changing the question or rephrasing it."
