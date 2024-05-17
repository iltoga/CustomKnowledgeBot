# factory class for creating assistant objects

from services.assistants.base_assistant import BaseAssistant
from services.assistants.revisbali_assistant import RevisBaliAssistant


class AssistantFactory:
    @staticmethod
    def create_assistant(
        assistant_type: str = None,
        default_llm="openai_big",
        small_llm="openai_small",
        translation_llm="openai_small",
        embeddings_llm="openai_embeddings",
        reset_db=True,
    ) -> BaseAssistant:
        if assistant_type is None:
            assistant_type = "revisbali_cs"

        if assistant_type == "revisbali_cs":
            return RevisBaliAssistant(
                reset_db=reset_db,
                default_llm=default_llm,
                small_llm=small_llm,
                translation_llm=translation_llm,
                embeddings_llm=embeddings_llm,
            )
        else:
            raise ValueError("Invalid assistant type")
