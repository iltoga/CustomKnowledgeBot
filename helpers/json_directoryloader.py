import logging
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_core.documents import Document


class JsonDirectoryLoader(BaseLoader):
    """Load a directory with `JSON` files and process them according to a schema.

    Loader also stores file paths in metadata.
    """

    def __init__(
        self,
        path: Union[str, Path],
        glob: str = "**/[!.]*json",
        silent_errors: bool = False,
        load_hidden: bool = False,
        recursive: bool = False,
        jq_schema: str = ".[]",
        content_key: str = None,
    ):
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.silent_errors = silent_errors
        self.logger = logging.getLogger(__name__)
        self.jq_schema = jq_schema
        self.content_key = content_key

    @staticmethod
    def _is_visible(path: Path) -> bool:
        return not any(part.startswith(".") for part in path.parts)

    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        loader = JSONLoader(
                            str(i), jq_schema=self.jq_schema, content_key=self.content_key, text_content=False
                        )
                        sub_docs = loader.load()
                        for doc in sub_docs:
                            doc.metadata["source"] = str(i)
                        docs.extend(sub_docs)
                    except Exception as ex:
                        if self.silent_errors:
                            self.logger.warning(ex)
                        else:
                            raise ex
        return docs


if __name__ == "__main__":
    # Example usage of the JsonDirectoryLoader
    jq_schema = """
    {
    questions_and_answers: .questions_and_answers[] | {
        page_content: ("Q: " + .question + "\nA: " + .answer)
    }
    }
    """
    loader_1 = JsonDirectoryLoader(
        path="test_data/json_files",
        content_key="questions_and_answers",
        jq_schema=jq_schema,
    )
    documents = loader_1.load()
    for doc1 in documents:
        print(doc1)
        print("-----")
