import torch# 导入必要的库
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')# 加载预训练模型tokenizer (vocabulary)

text = "What is the fastest car in the"
indexed_tokens = tokenizer.encode(text)# 对文本输入进行编码
tokens_tensor = torch.tensor([indexed_tokens])# 在PyTorch张量中转换indexed_tokens

# 加载预训练模型 (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')


model.eval()#将模型设置为evaluation模式，关闭DropOut模块

# 如果你有GPU，把所有东西都放在cuda上
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

# 预测所有的tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
# 得到预测的单词
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
# 打印预测单词
print(predicted_text)