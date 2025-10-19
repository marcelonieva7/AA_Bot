import asyncio
import time
from enum import Enum

from openai import OpenAI, AsyncOpenAI

from src.config.envs import settings

client_nvidia = OpenAI(
  base_url = settings.NVIDIA_URL,
  api_key = settings.NVIDIA_API_KEY,
)

def chat(system="", user="", model="openai/gpt-oss-120b"):
	response = client_nvidia.chat.completions.create(
		messages=[
		{
			"role": "system",
			"content": system
		},
		{
			"role": "user",
			"content": user
		}
		],
		model=model,
		temperature= 1,
		max_tokens= 2500,
		#extra_body={"chat_template_kwargs": {"thinking":False}},
	)

	return response.choices[0].message.content


class Models(Enum):
	gpt_20b = 'openai/gpt-oss-20b'
	kimi_k2 = 'moonshotai/kimi-k2-instruct'
	llama4_scout = 'meta/llama-4-scout-17b-16e-instruct'

async_client_nvidia = AsyncOpenAI(
    base_url=settings.NVIDIA_URL,
    api_key=settings.NVIDIA_API_KEY,
)

MAX_CALLS_PER_MINUTE = 40
_rate_lock = asyncio.Semaphore(MAX_CALLS_PER_MINUTE)
_call_timestamps: list[float] = []

async def _rate_limiter():
    global _call_timestamps

    async with _rate_lock:
        now = time.time()

        _call_timestamps = [t for t in _call_timestamps if now - t < 60]

        if len(_call_timestamps) >= MAX_CALLS_PER_MINUTE:
            wait_time = 60 - (now - _call_timestamps[0])
            await asyncio.sleep(wait_time)

        _call_timestamps.append(time.time())


async def async_chat(system: str = "", user: str = "", model: str = "qwen/qwen3-235b-a22b", thinking: bool = False) -> str:
    await _rate_limiter()

    response = await async_client_nvidia.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=1,
        max_tokens=2500,
        extra_body={"chat_template_kwargs": {"thinking":thinking}},
    )

    return response.choices[0].message.content

