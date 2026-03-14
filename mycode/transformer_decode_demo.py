from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'Qwen/Qwen3-0.6B'

# 准备工作

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
model.eval()


# 开始生成llm的输入
prompt = 'hello'

messages = [
    {'role': 'user', 'content': prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generate_prmpt = True,
)
print(f"text: {repr(text)}")
print()

inputs = tokenizer(
    text,
    return_tensors = 'pt'
)
print(f'inputs: {inputs}')
print()


# 开始推理
gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
outputs = model.generate(**inputs, **gen_kwargs)

print(f"output: {outputs}")
print()

outputs = tokenizer.decode(
    outputs
)
print(f'outputs_decode: {repr(outputs)}')
print()