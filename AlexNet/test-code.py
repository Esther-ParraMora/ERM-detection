"""
Created on Sun Jun 27 12:08:03 2021

@author: Esther Parra
"""
# Test AlexNet for ERM detection in a single image
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_path):    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(device)

from net import AlexNetERM
model = AlexNetERM()
model.load_state_dict(torch.load('weigths'))
model.to(device)  
model.eval()
image_path = '../noERM-sample.png'
res = np.argmax(model(image_loader(image_path)).cpu().detach().numpy())
print('ERM' if res==0 else 'no ERM')

