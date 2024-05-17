from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_core.documents import Document

from helpers.json_directoryloader import JsonDirectoryLoader
from helpers.md_directoryloader import MdDirectoryLoader
from helpers.md_textsplitter import MdTextSplitter


class KnowledgeLoader:
    @staticmethod
    def load_pdf_splits(knowledge_dir: str = None, split=True) -> List[Document]:
        if knowledge_dir is None:
            return []
        loader = PyPDFDirectoryLoader(path=knowledge_dir, recursive=True)
        if split:
            return loader.load_and_split()
        return loader.load()

    @staticmethod
    def load_web_content(knowledge_url: List[str] = None, split=True) -> List[Document]:
        if knowledge_url is None:
            return []

        loader = WebBaseLoader(knowledge_url)
        if split:
            return loader.load_and_split()
        return loader.load()

    @staticmethod
    def load_md_splits(knowledge_dir: str, split=True) -> List[Document]:
        loader = MdDirectoryLoader(knowledge_dir)
        if split:
            return loader.load_and_split(text_splitter=MdTextSplitter())
        return loader.load()

    @staticmethod
    def load_json_splits(knowledge_dir: str) -> List[Document]:
        jq_schema = """
{
  questions_and_answers: .questions_and_answers[] | {
    page_content: ("Q: " + .question + "\nA: " + .answer)
  }
}
"""
        loader = JsonDirectoryLoader(
            knowledge_dir,
            content_key="questions_and_answers",
            jq_schema=jq_schema,
        )
        return loader.load()
