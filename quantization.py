import torch
import torch.quantization as quantization

# 加载训练好的 PyTorch 模型
model = torch.load('best_model.pth')

# 将模型切换到评估模式
model.eval()

# 定义一个仿射量化模型（QuantizedModel）
quantized_model = quantization.QuantWrapper(model)

# 创建一个 torch.quantization.QuantStub 实例，用于在模型输入和输出之间插入量化和反量化操作
quant_stub = quantization.QuantStub()
dequant_stub = quantization.DeQuantStub()

# 将模型和量化/反量化操作放入 nn.Sequential 中，以便能够调用 torch.quantization.quantize_static 函数
quantized_model = torch.nn.Sequential(quant_stub, model, dequant_stub)

# 进行静态量化
quantized_model = quantization.quantize_static(quantized_model, qconfig_dict={torch.nn.Linear: torch.quantization.default_qconfig}, dtype=torch.qint8)

# (可选) 查看量化后的模型
print(quantized_model)

# 将量化后的模型保存
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
