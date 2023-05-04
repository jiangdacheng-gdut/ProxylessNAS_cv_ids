import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from proxyless_nas import proxyless_cpu

from tqdm import tqdm   # 展示信息

# 定义用于训练的设备
device = "cuda"

# Step 1: Prepare dataset
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

# 网络流量图片数据集
train_dataset = datasets.ImageFolder('/root/dacheng/proxylessnas/cv_ids_25/train', transform=data_transforms)
val_dataset = datasets.ImageFolder('/root/dacheng/proxylessnas/cv_ids_25/test', transform=data_transforms)

# Step 2: Load dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

# Step 3: Define model
model = proxyless_cpu()
model = torch.load('best_model.pth')

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00001)

# Step 4: Define loss function and optimizer

best_loss = float('inf')    # 初始化最佳损失为正无穷
best_model = None

for epoch in range(10):
    model.train()  # 设置模型为训练模式
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader)  # 使用tqdm包装训练数据加载器以输出进度条
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()  # 清空梯度
        inputs, targets = inputs.to(device), targets.to(device)
        model = model.to(device)  # 将模型转换为 CUDA 模型
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        epoch_loss += loss.item()   
        progress_bar.update()  # 更新进度条
        progress_bar.set_description('Epoch {}, Loss: {:.15f}'.format(epoch, loss.item()))  # 输出当前epoch的损失
        progress_bar.set_postfix({'loss': loss.item()})  # 更新进度条位置


        # 检查当前损失值是否比最低值更低
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model, "best_model.pth")  # 保存当前模型
            print("第%d次的第%d次训练，损失值为：%f，保存最佳模型" % (epoch, i, loss.item()))
        # else:
        #     print("第%d次的第%d次训练，损失值为：%f" % (epoch, i, loss.item()))
                    
    # epoch_loss = epoch_loss / len(train_loader.dataset)  # 计算当前epoch的平均损失
    # print('Epoch Loss: {:.8f}'.format(epoch_loss))

    # 在每个 epoch 结束后，量化模型
    torch.quantization.convert(model, inplace=True)

    model.eval()  # 设置模型为评估模式
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # 前向传播
            _, predicted = outputs.max(1)  # 获取预测结果
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    # print()
    # print('Epoch %d, Accuracy: %.2f%%' % (epoch, acc))  # 打印当前精度
    print('Accuracy: {:.4f}%'.format(acc))  # 打印当前精度