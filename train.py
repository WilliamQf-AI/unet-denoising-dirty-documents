import torch
import torch.nn as nn
from torch import optim
import os
from unet import UNet
from datasets import UNetDataset
import transforms as Transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

if not os.path.exists('./weight'):
    os.mkdir('./weight')
LR = 1e-5
EPOCH = 50
BATCH_SIZE = 64
weight = './weight/weight.pth'
weight_with_optimizer = './weight/weight_with_optimizer.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():

    # dataset
    transforms = [
        Transforms.ToGray(),
        Transforms.RondomFlip(),
        Transforms.RandomRotate(15),
        Transforms.RandomCrop(48, 48),
        Transforms.Log(0.5),
        # Transforms.EqualizeHist(0.5),
        # Transforms.Blur(0.2),
        Transforms.ToTensor()
    ]
    dataset = UNetDataset('./data/train/', './data/train_cleaned/', transform=transforms)
    dataLoader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # init model
    net = UNet(1, 2).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().to(device)

    # load weight
    if os.path.exists(weight_with_optimizer):
        checkpoint = torch.load(weight_with_optimizer)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('load weight')

    # train
    for epoch in range(EPOCH):
        # train
        for step, (batch_x, batch_y) in enumerate(dataLoader):
            # import cv2
            # import numpy as np
            # display = np.concatenate(
            #     (batch_x[0][0].numpy(), batch_y[0][0].numpy().astype(np.float32)),
            #     axis=1
            # )
            # cv2.imshow('display', display)
            # cv2.waitKey()
            batch_x = batch_x.to(device)
            batch_y = batch_y.squeeze(1).to(device)
            output = net(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: %d | loss: %.4f' % (epoch, loss.data.cpu()))

        # save weight
        if (epoch + 1) % 10 == 0:
            torch.save({
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, weight_with_optimizer)
            torch.save({
                'net': net.state_dict()
            }, weight)
            # 这是保存为ONNX的方法:
            # 由于PyTorch的模型,是动态调整大小的,这里需要初始化一个指定格式的数据,用来调整模型大小
            # 就是和你训练模型的时候用的数据一样的格式就行
            dummy_input = Variable(torch.randn(1, 1, 28, 28)).to(device)
            # 保存模型
            torch.onnx.export(net, dummy_input, "torch.onnx")
            print('saved')


if __name__ == '__main__':
    train()
