参考
https://github.com/datawhalechina/happy-llm/blob/main/Extra-Chapter/why-fine-tune-small-large-language-models/why-fine-tune-small-large-language-models.ipynb


# lora的目的

省流: 小调节, 变量矩阵是低秩的, 将低秩拆解为两个矩阵的乘积

首先, 我们是准备小参数调, 不想过多的影响原始的大模型(保持原有的性能, 不出现大量的遗忘, 同时, 我想要的能力又能得到增强)

我们假设, 我们只进行小的调节, 这个时候的增量大概率不是满秩的. 也就是我们不需要学习矩阵中的所有内容(因为他们不是线性独立的)

低秩矩阵可以写成两个矩阵之和的形式

W(100*100) = A(100*k) * B(k*100), 秩最多是 k

# 源代码跑不通, 这里进行了调整

## tokenizer.pad_token_id

### 改变内容

改成了 tokenizer.eos_token_id, 这样微调后的模型可以快速输出. 

tokenizer.pad_token = tokenizer.eos_token

### 效果分析

不确定, 但是确实在 peft 模型的情况下, 使用 eos 可以加速输出

## 取消手动输入<start>等

### 改变内容

直接使用 apply_prompt_template

### 效果分析

不确定, 未评测

## 加入model.eval()

### 改变内容

在测试前直接加入 model.eval()

### 效果分析

直接解决问题

## 默认不输出json

### 改变内容

训练数据集中, system_prompt不加入输出为 json, 但是答案是 json. 

### 效果分析

无 prompt 提示的情况下, 大大提高模型输出 json 的概率
