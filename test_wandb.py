import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

# 初始化 wandb
wandb.init(project="pytorch-example")

# 配置超参数
wandb.config.epochs = 5
wandb.config.batch_size = 64
wandb.config.learning_rate = 0.01

# 数据加载
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)

# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate)

# 训练模型
for epoch in range(wandb.config.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 记录损失
        if batch_idx % 100 == 0:
            wandb.log({"loss": loss.item()})

# 保存模型
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

# 结束 wandb 运行
wandb.finish()
