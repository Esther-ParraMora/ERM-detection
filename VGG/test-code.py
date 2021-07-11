"""
Created on Sun Jun 27 12:08:03 2021

@author: Esther Parra
"""
# Test AlexNet for ERM detection in a single image
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from net import VGGERM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to load the image
def image_loader(image_path):    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(device)

model = VGGERM()
model.load_state_dict(torch.load('weights'))
model.to(device)  
model.eval()
# the path of the image
image_path = '../ERM-sample.png'
res = np.argmax(model(image_loader(image_path)).cpu().detach().numpy())
print('ERM' if res==0 else 'no ERM')

