import asyncio
import logging
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from services.assistants.assistant_factory import AssistantFactory
from services.assistants.base_assistant import BaseAssistant


class TelegramBot:

    def __init__(
        self,
        assistant_type=None,
        default_llm="openai_big",
        small_llm="openai_small",
        translation_llm="openai_small",
        embeddings_llm="openai_embeddings",
        reset_db=True,
    ):
        self.sessions = {}
        self.assistant_type = assistant_type
        self.application = None
        self.logger = self._initialize_logger()
        self.default_llm = default_llm
        self.small_llm = small_llm
        self.translation_llm = translation_llm
        self.embeddings_llm = embeddings_llm
        self.reset_db = reset_db

    def _initialize_logger(self):
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
        return logging.getLogger(__name__)

    def get_assistant(self, user_id):
        if user_id not in self.sessions:
            self.sessions[user_id] = AssistantFactory.create_assistant(
                assistant_type=self.assistant_type,
                default_llm=self.default_llm,
                small_llm=self.small_llm,
                translation_llm=self.translation_llm,
                embeddings_llm=self.embeddings_llm,
                reset_db=self.reset_db,
            )
        return self.sessions[user_id]

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.message.from_user is None:
            return
        user_id = update.message.from_user.id
        self.get_assistant(user_id)
        await update.message.reply_text(
            "Hi! I am Ayu 2.0, your AI assistant ready to answer your questions about RevisBali and its services. How can I help you today?"
        )

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.message.from_user is None:
            return
        user_id = update.message.from_user.id
        if user_id in self.sessions:
            del self.sessions[user_id]
            await update.message.reply_text(
                "Your session has been closed. If you need further assistance, feel free to start a new session anytime with `/start` command."
            )
        else:
            await update.message.reply_text("No active session found to end.")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        await update.message.reply_text(
            "You can ask me anything about RevisBali services, and I will try my best to assist you."
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.message.from_user is None:
            return
        user_id = update.message.from_user.id
        assistant = self.get_assistant(user_id)
        user_message = update.message.text
        response = await self.respond(assistant, user_message)

        stop_word = assistant.stop_word
        if stop_word in response:
            response = response.replace(stop_word, "").strip()
            await update.message.reply_text(response)
            await self.stop(update, context)
        else:
            await update.message.reply_text(response)

    async def respond(self, assistant: BaseAssistant, user_message: str) -> str:
        retries = 3
        for attempt in range(retries):
            try:
                response = assistant.respond(user_message)
                return response
            except Exception as ex:
                self.logger.warning(f"Attempt {attempt + 1} failed: {ex}")
                if attempt + 1 == retries:
                    return "Sorry, I'm having trouble connecting to the service. Please try again later."
            await asyncio.sleep(2)

    async def error(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.logger.warning('Update "%s" caused error "%s"', update, context.error)

    async def run(self) -> None:
        self.application = (
            ApplicationBuilder()
            .token(os.getenv("TELEGRAM_TOKEN"))
            .read_timeout(600)
            .get_updates_read_timeout(600)
            .write_timeout(600)
            .get_updates_write_timeout(600)
            .pool_timeout(600)
            .get_updates_pool_timeout(600)
            .connect_timeout(600)
            .get_updates_connect_timeout(600)
            .build()
        )

        # Add command and message handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Log all errors
        self.application.add_error_handler(self.error)

        # Start the Bot
        await self.application.initialize()
        await self.application.start()
