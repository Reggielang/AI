from langchain_ollama import OllamaLLM
import asyncio




async def task1():
    model = OllamaLLM(model="deepseek-r1:14b",
                      verbose=True)
    chunks= []
    async for chunk in model.astream("天空是什么颜色的？"):
        chunks.append(chunk)
        if len(chunks) ==2:
            print(chunks[1])
        print(chunk,end='|',flush=True)

async def task2():
    model = OllamaLLM(model="deepseek-r1:14b",
                      verbose=True)
    chunks= []
    async for chunk in model.astream("讲个笑话？"):
        chunks.append(chunk)
        if len(chunks) ==2:
            print(chunks[1])
        print(chunk,end='|',flush=True)

async def main():
    #同步调用
    await task1()
    await task2()

    #异步调用 并发运行两个任务
    await asyncio.gather(task1(),task2())


asyncio.run(main())