import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
total_img = 38562
train_img = round(total_img * 0.8)
valid_img = total_img - train_img
img_size = 256
batch_size = 32

#Code template taken from the provided "Transfer Learning to Birds with ResNet18" tutorial

def get_bird_data():
    transform_train = transforms.Compose([transforms.Resize(img_size),
                                          # Take 128x128 crops from padded images
                                          transforms.RandomCrop(img_size, padding=8, padding_mode='edge'),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([transforms.GaussianBlur(5)]),
                                          # transforms.RandomApply([transforms.ColorJitter(3, 3, 3, 0.5)]),
                                          transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(root='./birds/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root='./birds/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    classes = open("./birds/names.txt").read().strip().split("\n")

    # Backward mapping to original class ids (from folder names) and species name (from names.txt)
    class_to_idx = trainset.class_to_idx
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()}
    idx_to_name = {k: classes[v] for k, v in idx_to_class.items()}
    return {'train': trainloader, 'test': testloader, 'to_class': idx_to_class, 'to_name': idx_to_name}


def train(net, train_dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.005,
          verbose=1, print_every=10, state=None, schedule={}, checkpoint_path=None):
    net.to(device)
    net.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    # train_dataloader, val_dataloader = torch.utils.data.random_split(train_dataloader, [train_img, valid_img],
    #                                                                  generator=torch.Generator().manual_seed(42))

    # Load previous training state
    if state:
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        losses = state['losses']

    # Fast forward lr schedule through already trained epochs
    for epoch in range(start_epoch):
        if epoch in schedule:
            print("Learning rate: %f" % schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

    for epoch in range(start_epoch, epochs):
        sum_loss = 0.0
        correct = 0
        # Update learning rate when scheduled
        if epoch in schedule:
            print("Learning rate: %f" % schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

        for i, batch in enumerate(train_dataloader, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            _, pred = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step()  # takes a step in gradient direction

            losses.append(loss.item())
            sum_loss += loss.item()
            correct += torch.sum(pred == labels).item()
            if i % print_every == print_every - 1:  # print every 10 mini-batches
                if verbose:
                    print('[%d, %5d] loss: %.3f' % (epoch, i + 1, sum_loss / print_every))
                sum_loss = 0.0
        print('Training Acc: %.3f' % (correct / total_img))
        if (epoch + 1) % 5 == 0 and checkpoint_path:
            state = {'epoch': epoch + 1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}
            torch.save(state, checkpoint_path + 'chkpnt-%d.pkl' % (epoch + 1))
            # validation(net, val_dataloader)
    return losses


def validation(net, dataloader):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    sum_loss = 0.0
    losses = []
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader, 0)):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, pred = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            sum_loss += loss.item()
            correct += torch.sum(pred == labels).item()
        print('loss: %.3f' % sum_loss)
        print('Validation Acc: %.3f' % (correct / valid_img))


def predict(net, dataloader, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    net.to(device)
    net.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            if i % 100 == 0:
                print(i)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            fname, _ = dataloader.dataset.samples[i]
            out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
    out.close()


if __name__ == "__main__":
    data = get_bird_data()
    resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
    num_ft = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ft, 555)
    state_dict = torch.load('./test/chkpnt-175.pkl')
    resnet.load_state_dict(state_dict["net"])
    losses = train(resnet, data['train'], state=state_dict,
                   epochs=300, lr=.00001, print_every=50, checkpoint_path='./test/')
    predict(resnet, data['test'], "preds.csv")
