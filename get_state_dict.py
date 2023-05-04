import torch

model = torch.load('best_model.pth')  # 从路径加载预训练模型

#打印模型的状态字典
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
    # print(param_tensor,"\t",model.state_dict()[param_tensor].size())

# PATH="/root/dacheng/proxylessnas/model_state_dict.pth"
# torch.save(model.state_dict(), PATH)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

print(model.name_modules())