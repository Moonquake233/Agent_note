import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
model_id = "Qwen/Qwen1.5-0.5B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device:{device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成")

messages = [
    {"role":"system","content": "You are a helpful assistant."},
    {"role":"user","content":"你好，请问你能干什么"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True
)

model_inputs = tokenizer([text],return_tensors="pt").to(device)  #return_tensors="pt"：返回 PyTorch 张量（pt 就是 PyTorch）

print("编码后的输入文本：")
print(model_inputs)

generated_ids = model.generate(  #model.generate 生成回答 
    model_inputs.input_ids,     #model_inputs.input_ids：喂进去的 prompt token id
    max_new_tokens = 512
)                               #返回值 generated_ids 是包含 prompt 在内的完整序列

generated_ids = [
    output_ids[len(input_ids):] for input_ids ,output_ids in zip(model_inputs.input_ids,generated_ids)
]

response = tokenizer.batch_decode(generated_ids,skip_special_tokens = True)[0]

print("\n模型的回答:")
print(response)