import time
import os

# pip install -qU langchain-openai
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

# 创建LLM实例
llm = ChatOpenAI(model="gpt-4")
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def measure_invoke_time(llm, prompt):
    # 记录开始时间
    start_wall_time = time.time()
    start_cpu_times = os.times()

    # 调用LLM
    response = llm.invoke(prompt)

    # 记录结束时间
    end_wall_time = time.time()
    end_cpu_times = os.times()

    # 计算经过的时间
    wall_time = end_wall_time - start_wall_time
    user_time = end_cpu_times.user - start_cpu_times.user
    sys_time = end_cpu_times.system - start_cpu_times.system
    total_cpu_time = user_time + sys_time

    return response, wall_time, user_time, sys_time, total_cpu_time

# 第一次调用
response1, wall_time1, user_time1, sys_time1, total_cpu_time1 = measure_invoke_time(llm, "给我讲个笑话")
print("First call response:", response1)
print(f"First call CPU times: user {user_time1 * 1000:.0f} ms, sys: {sys_time1 * 1000:.0f} ms, total: {total_cpu_time1 * 1000:.0f} ms")
print(f"First call Wall time: {wall_time1 * 1000:.0f} ms")

# 第二次调用
response2, wall_time2, user_time2, sys_time2, total_cpu_time2 = measure_invoke_time(llm, "给我讲个笑话")

print("Second call response:", response2)
print(f"Second call CPU times: user {user_time2 * 1000:.0f} ms, sys: {sys_time2 * 1000:.0f} ms, total: {total_cpu_time2 * 1000:.0f} ms")
print(f"Second call Wall time: {wall_time2 * 1000:.0f} ms")