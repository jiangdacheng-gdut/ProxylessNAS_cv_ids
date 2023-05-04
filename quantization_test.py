import torch
from proxyless_nas import ProxylessNASNets

# 加载模型
model = torch.load('best_model.pth')
model.eval()
model.to('cpu')

# 转换为 TorchScript
scripted_model = torch.jit.script(model)

# 保存 TorchScript 模型
torch.jit.save(scripted_model, "scripted_model.pt")
