import unittest

from services.assistants.revisbali_assistant import RevisBaliAssistant


class TestRevisBaliAssistant(unittest.TestCase):

    def test_load_and_split_documents(self):
        assistant = RevisBaliAssistant()
        documents = assistant.load_and_split_documents()
        self.assertIsInstance(documents, list)

    def test_respond(self):
        assistant = RevisBaliAssistant(reset_db=True, default_llm="groq_big")
        res = assistant.respond("What is the best visa for a 20 days stay in Bali?")
        print(res)
        # self.assertTrue(re.search(r"\bevoa\b|\be-voa\b", res.lower()), "Response does not contain 'evoa' or 'e-voa'")


if __name__ == "__main__":
    unittest.main()
