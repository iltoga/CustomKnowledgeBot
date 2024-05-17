import logging
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document


class MdDirectoryLoader(BaseLoader):
    """Load a directory with `PDF` files using `pypdf` and chunks at character level.

    Loader also stores page numbers in metadata.
    """

    def __init__(
        self,
        path: Union[str, Path],
        glob: str = "**/[!.]*.md",
        silent_errors: bool = False,
        load_hidden: bool = False,
        recursive: bool = False,
    ):
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.silent_errors = silent_errors
        self.logger = logging.getLogger(__name__)

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
                        loader = TextLoader(str(i), encoding="utf-8")
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
