import re
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter


class MdTextSplitter(TextSplitter):
    """
    A text splitter that splits markdown text into chunks based on '##' headings."""

    def __init__(self):
        """Initialize the MdTextSplitter."""
        # No need to specify a chunk size here since the delimiter is based on markdown structure.
        super().__init__()

    def split_text(self, text: str) -> List[Document]:
        """
        Splits a given markdown text into chunks based on '##' headings.

        Args:
            text (str): The input markdown text to split.

        Returns:
            List[Document]: A list of document chunks based on markdown headings.
        """
        # Split the text into sections based on '##' headings
        sections = re.split(r"\n## ", text)  # Note: this assumes that headings are formatted as '## Heading'

        # Initialize an empty list to hold the document chunks
        chunks = []

        # Iterate over the sections to create Document objects
        for section in sections:
            # Ensure the section is not just whitespace
            if section.strip():
                # If the section does not start with '##', add it as it's the first section
                if not section.startswith("##"):
                    section = "## " + section
                chunks.append(Document(page_content=section.strip()))

        return chunks

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Leverage the split_text method to split each markdown document into smaller documents based on headings."""
        result = []
        for document in documents:
            text_chunks = self.split_text(document.page_content)
            result.extend(text_chunks)
        return result


if __name__ == "__main__":
    # Example usage of the MdTextSplitter
    text = """
## Contact Information

Keywords: contact details, WhatsApp, email, business hours

For quick inquiries or appointment bookings, please contact us via WhatsApp at +6282147209978. For more detailed questions and service assistance, email us at info@revisbali.com. Our team is available to respond to WhatsApp messages from 10 AM to 6 PM, monday to sunday, and emails are answered within 24 hours during business hours.

## Office Hours and Visit Information

Keywords: office visit, office hours, coffee, conversation

Visit our office Monday through Friday, between 10 AM and 6 PM, to discuss your visa and administrative needs over a warm cup of coffee. We are located on the main road from Pandawa to Gunung Payung in Nusa Dua, in front of Warung Local Price. We value good conversations and are committed to making your processes as smooth as possible.

## Company Registration Details

Keywords: company name, tax identification, professionalism, PT. BALI SHANKARA KONSULTAN

RevisBali operates under the registered name PT. BALI SHANKARA KONSULTAN. Our tax identification number (NPWP) is 40.932.205.4-908.000. Trust our professional team to handle your visa and administrative concerns with the highest standards of care and expertise.

## Location and Contact Information

Keywords: office location, directions, document service, Bali, contact details

Our office is situated in the Bukit Peninsula, on the main road between Pandawa and Gunung Payung, in front of Warung Local Price, 15-minute drive away from the Ngurah Rai Immigration Office. If you’re unable to visit our office, we offer a document pickup and drop-off service once your visa is processed, available on appointment.

## Full Address and Contact Details

Keywords: full address, contact information, Bali, post code

For those planning a visit, the full address is Jl. Raya Nusa Dua Selatan Ruko F, Desa Kutuh, Kecamatan Kuta Selatan, Badung- Bali, Indonesia. The postal code for our location is 80361. This information is useful for mailing, navigation, and scheduling appointments at our office.

"""
    splitter = MdTextSplitter()
    documents = splitter.split_text(text)
    for doc in documents:
        print(doc.page_content)
        print("-----")
