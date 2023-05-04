import torch
import torch.quantization
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.quantization import QConfig
from torch.quantization import default_histogram_observer, default_per_channel_weight_observer

# 定义数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

# 加载训练数据集
train_dataset = datasets.ImageFolder('/root/dacheng/proxylessnas/cv_ids_25/test', transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

# 加载测试数据集
val_dataset = datasets.ImageFolder('/root/dacheng/proxylessnas/cv_ids_25/test', transform=data_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

# class ProxylessNAS_model(nn.Module.proxylessnas):


# 加载训练好的模型
model = torch.load('best_model.pth')
model.eval()

# 将模型权重移到设备上
model.to('cpu')

# 定义量化配置并将其应用于模型
# qconfig = torch.quantization.get_default_qconfig("fbgemm")
qconfig = QConfig(
    activation=default_histogram_observer.with_args(quant_min=0, quant_max=255),
    weight=default_per_channel_weight_observer.with_args(quant_min=-128, quant_max=127)
)
model.qconfig = qconfig

# 准备模型进行量化
model_prepared = torch.quantization.prepare(model)

# 使用代表性数据收集统计信息
# 假设 `train_loader` 是一个 PyTorch 数据加载器，用于加载代表性数据
with torch.no_grad():
    for data, _ in tqdm(train_loader, desc="Quantization"):
        # print(data)
        data = data.to('cpu')
        model_prepared(data)

# 转换模型为量化版本
quantized_model = torch.quantization.convert(model_prepared)

# 进行测试，计算精确度
quantized_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        quantized_model = quantized_model.to('cpu')
        outputs = quantized_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print('Accuracy of the quantized model on the test dataset: {:.2f}%'.format(100 * accuracy))




# 保存量化后的模型
# torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pt')
