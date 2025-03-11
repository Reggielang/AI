from langchain_ollama import OllamaLLM
import asyncio

model = OllamaLLM(model="deepseek-r1:14b",
                verbose=True)

async def async_stream():
    events = []
    async for event in model.astream_events("hello",version='v2'):
        events.append(event)
    print(events)



asyncio.run(async_stream())
