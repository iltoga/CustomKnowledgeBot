import asyncio
import logging
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI, Request
from telegram import Update

from services.telegram_bot import TelegramBot

bot = TelegramBot(
    assistant_type="revisbali_cs",
    default_llm="groq_big",
    small_llm="groq_big",
    translation_llm="groq_big",
    embeddings_llm="openai_embeddings",
    reset_db=True,
)
dotenv.load_dotenv()
# Initialize logging
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(bot.run())
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/webhook")
async def webhook(request: Request):
    if bot.application is None:
        return {"error": "Bot not initialized"}, 500
    update = Update.de_json(await request.json(), bot.application.bot)
    await bot.application.process_update(update)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6000)
