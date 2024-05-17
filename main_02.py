import logging
import re

import dotenv

from services.assistants.revisbali_assistant import RevisBaliAssistant

dotenv.load_dotenv()
# Initialize logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    reset_db = False
    # default_llm = "ollama_local"
    # small_llm = "ollama_local"
    # translation_llm = "ollama_local"
    # embeddings_llm = "ollama_embeddings"

    default_llm = "groq_big"
    small_llm = "groq_big"
    translation_llm = "groq_big"
    # embeddings_llm = "ollama_embeddings"

    # default_llm = "openai_big"
    # small_llm = "openai_small"
    # translation_llm = "openai_small"
    embeddings_llm = "openai_embeddings"

    assistant = RevisBaliAssistant(
        reset_db=reset_db,
        default_llm=default_llm,
        small_llm=small_llm,
        translation_llm=translation_llm,
        embeddings_llm=embeddings_llm,
    )
    # res = assistant.respond("Qual'e' il miglior visto per un soggiorno di 20 giorni a Bali?")
    res = assistant.respond("Qual'e' il miglior visto per un soggiorno di 40 giorni a Bali?")
    # res = assistant.respond("Where is your office?")
    # res = assistant.respond("What is the address of your office?")
    # res = assistant.respond("I want to move to bali with my family")

    print(res)
