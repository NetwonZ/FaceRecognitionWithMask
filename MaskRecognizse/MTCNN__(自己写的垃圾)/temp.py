import cv2
import torch
import numpy as np
from torch import nn
rectangle = torch.tensor([747.0000, 544.0000, 839.0000, 642.0000,   0.9975])

class SG(nn.model):
    def __init__(self):
        super(SG, self).__init__()
        self.f = nn.sequential(
            nn.Linear(2, 2),
            nn.sigmoid(),
            nn.Linear(2, 2),
            nn.sigmoid()
        )
    def forward(self, x):
        return self.f(x)

model = SG()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()
for i in range(100):
    #前向传播
    x = torch.tensor([1, 1], dtype=torch.float32)
    y = model(x)
    #计算损失
    loss = criterion(y, x)
    #反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #打印损失
    print(loss.item())

