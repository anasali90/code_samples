import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0')

# Hyper parameters
num_epochs = 10
num_classes = 2
batch_size = 25
learning_rate = 0.0001

def load_dataset():
    transform = transforms.Compose(
        [transforms.Resize([64, 64]),
         
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data_path = 'data/dataset/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root='data/test/',
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=500,
        num_workers=0,
        shuffle=True
    )
    return train_loader,test_loader


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=4, stride=1, padding=0))

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        return out

model = ConvNet(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


train_loader = load_dataset()[0] 
total_step = len(train_loader)
test_loader = load_dataset()[1] 
for epoch in range(num_epochs):
    for i,  (images, labels) in enumerate(load_dataset()[0]):
        images = images.to(device)
        labels = labels.to(device)
        
    
        outputs = model(images)
        loss = criterion(outputs, labels)
        
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


model.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy  %'.format(100 * correct / total))


torch.save(model.state_dict(), 'torch_weld_det.ckpt')
