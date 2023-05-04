import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from proxyless_nas import proxyless_cpu
from tqdm import tqdm   # 展示信息


device = "cuda"

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

val_dataset = datasets.ImageFolder('/root/dacheng/proxylessnas/cv_ids_25/test', transform=data_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

model = torch.load('best_model.pth')

for epoch in range(8):
    model.eval()  # 设置模型为评估模式
    correct, total = 0, 0
    total_time = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            start_time = time.time()  # 记录推理开始时间
            outputs = model(inputs)  # 前向传播
            end_time = time.time()  # 记录推理结束时间
            inference_time = end_time - start_time  # 计算推理时间
            total_time += inference_time  # 累加推理时间
            _, predicted = outputs.max(1)  # 获取预测结果
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_inference_time = total_time / len(val_loader)  # 计算平均推理时间
    print()
    print('Epoch %d, Accuracy: %.4f%%, Average Inference Time: %.4f seconds' % (epoch, acc, avg_inference_time))  # 打印当前精度和平均推理时间