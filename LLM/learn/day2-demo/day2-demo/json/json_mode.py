from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "你是一个助手，请用中文输出JSON"},
        {"role": "user", "content": "2022世界杯冠军是那个队伍?"}
    ]
)
print(response.choices[0].message.content)
