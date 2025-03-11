class LocalModel:
    def generate(self, prompt):
        # 在这里实现本地模型的生成逻辑
        # 例如，使用 transformers 库加载本地模型并生成文本
        from transformers import pipeline

        generator = pipeline('text-generation', model=r'E:\AI\vllm-project\models\DeepSeek-R1-Distill-Qwen-7B')
        response = generator(prompt, max_length=50)
        return response[0]['generated_text']