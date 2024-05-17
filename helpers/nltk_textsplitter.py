from typing import Iterable, List

import nltk
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

# Ensure the required packages for sentence tokenization are downloaded
nltk.download("punkt")


class CustomNLTKTextSplitter(TextSplitter):
    def __init__(self, max_chunk_length: int = 600):
        """Initialize the NLTKTextSplitter with a specific maximum chunk length."""
        super().__init__(chunk_size=max_chunk_length)
        self.max_chunk_length = max_chunk_length

    def split_text(self, text: str) -> List[Document]:
        """
        Splits a given text into chunks without breaking sentences and paragraphs,
        ensuring no empty chunks.

        Args:
            text (str): The input text to split.

        Returns:
            List[Document]: A list of document chunks, guaranteed no empty chunks.
        """
        # Tokenize the text into sentences
        sentences = nltk.tokenize.sent_tokenize(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Strip and check if sentence is empty
            sentence = sentence.strip()
            if not sentence:  # Skip empty sentences
                continue

            # Check if adding this sentence would exceed the max chunk length
            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_length:
                if current_chunk:  # Ensure current chunk is not empty
                    chunks.append(Document(page_content=current_chunk))  # Wrap in Document
                current_chunk = sentence  # Start new chunk with the current sentence
            else:
                # Add the sentence to the current chunk
                current_chunk += (" " + sentence) if current_chunk else sentence

        # Add the last chunk if it contains any text and is not just whitespace
        if current_chunk.strip():
            chunks.append(Document(page_content=current_chunk))  # Wrap in Document

        return chunks

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Leverage the split_text method to split each document into smaller documents."""
        result = []
        for document in documents:
            text_chunks = self.split_text(document.page_content)
            result.extend(text_chunks)
        return result
