import torch 
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
# Device configuration
device = torch.device('cuda:0')

# Hyper parameters
num_epochs = 10
num_classes = 2
batch_size = 25
learning_rate = 0.0001
#transforms.Scale(imsize),
imsize = 64
loader = transforms.Compose([transforms.Scale([300,900]), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    return image



class ConvNet(nn.Module):
    def __init__(self, num_classes):
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
        #self.fc = nn.Linear(4*4*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #print(out)
        #out = out.reshape(out.size(0), -1)
        return out


model = ConvNet(num_classes)
model.load_state_dict(torch.load('torch_weld_det.ckpt'))
#model.eval()
#print(model.eval())
name = 'w562.JPG'
original_img = cv2.imread(name)
img = image_loader(name)#/iali/welds/data/test/weld_seam
output0 = (model(img)[0][0]).detach().numpy()
output1 = (model(img)[0][1]).detach().numpy()
yy = np.greater(output1, 0.2*np.ones_like(output1))
output = np.less(output0, output1)
output = output*yy
plt.imshow(output*yy)
plt.show()
labeld_img = label(output)
props = regionprops(labeld_img)
count = 0
largest_r = 0

for idx, region in enumerate(props) : 
    if region.bbox_area > largest_r : 
       count = idx
       largest_r = region.bbox_area 
print(props[count].bbox)
pt0 = (int((original_img.shape[1]/output.shape[1])*props[count].bbox[1]),int((original_img.shape[0]/output.shape[0])*props[count].bbox[0]))
pt1 = (int((original_img.shape[1]/output.shape[1])*props[count].bbox[3]),int((original_img.shape[0]/output.shape[0])*props[count].bbox[2]))
cc = cv2.rectangle(original_img, pt0,pt1,(123,231,2), 20)
#cv2.imshow('asdf',original_img)
#cv2.waitKey(20)
plt.imshow(cc)
plt.show()
