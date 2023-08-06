import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import matplotlib.patches as patches
import os

def create_bbox(obj):
    return [int(obj.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]

def get_label_code(obj):
    name = obj.find('name').text
    return 1 if name == "with_mask" else 2 if name == "mask_weared_incorrect" else 0

def build_target(img_id, xml_file):
    with open(xml_file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        boxes = [create_bbox(o) for o in objects]
        labels = [get_label_code(o) for o in objects]
        
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

image_files = sorted(os.listdir("/kaggle/input/face-mask-detection/images/"))
annotation_files = sorted(os.listdir("/kaggle/input/face-mask-detection/annotations/"))

class FaceMaskDataset(object):
    def __init__(self, transform):
        self.transform = transform
        self.imgs = image_files

    def __getitem__(self, idx):
        img_file = 'maksssksksss' + str(idx) + '.png'
        label_file = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join("/kaggle/input/face-mask-detection/images/", img_file)
        label_path = os.path.join("/kaggle/input/face-mask-detection/annotations/", label_file)
        img = Image.open(img_path).convert("RGB")
        target = build_target(idx, label_path)
        return self.transform(img), target if self.transform else img, target

    def __len__(self):
        return len(self.imgs)

preprocess = transforms.Compose([transforms.ToTensor()])
dataset = FaceMaskDataset(preprocess)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=lambda x: tuple(zip(*x)))

def build_model(num_classes):
    base_model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return base_model

model = build_model(3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training parameters
num_epochs = 25
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
data_loader_length = len(data_loader)
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model([images[0]], [targets[0]])
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    print(total_loss)

# Evaluate the model
model.eval()
for images, targets in data_loader:
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    break

predictions = model(images)

# Function to plot image
def display_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data
    ax.imshow(img.permute(1, 2, 0))
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# Display predictions
print("Prediction")
display_image(images[2], predictions[2])
print("Target")
display_image(images[2], targets[2])

# Save the model
torch.save(model.state_dict(), 'model.pt')

# Load the model
loaded_model = build_model(3)
loaded_model.load_state_dict(torch.load('model.pt'))
loaded_model.eval()
loaded_model.to(device)

# Test the loaded model
loaded_predictions = loaded_model(images)
print("Prediction with loaded model")
display_image(images[3], loaded_predictions[3])
