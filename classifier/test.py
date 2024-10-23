import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
from dataset import EMDiffusenDataset
from config import NoiseConfig as config

# f1 score and confusion matrix
from sklearn.metrics import f1_score, confusion_matrix

test_dataset = EMDiffusenDataset(data_root=config.test_data_root, corrupt=config.task)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=4)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

checkpoint = torch.load(f"{config.task}_best.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

testing_loss = 0.0
truths = []
predictions = []

model.eval()
with torch.no_grad():
    for j, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, truth = torch.max(labels.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        testing_loss += loss.item()
        truths.append(truth)
        predictions.append(predicted)
        
acc = torch.sum(torch.cat(truths) == torch.cat(predictions)).item() / len(test_dataset) * 100
f1 = f1_score(torch.cat(truths).cpu(), torch.cat(predictions).cpu(), average="macro")
print(f'Epoch test: {testing_loss / (j + 1)}, acc: {acc}%, f1: {f1}, len: {len(test_dataset)}')

# confusion matrix
cm = confusion_matrix(torch.cat(truths).cpu(), torch.cat(predictions).cpu())

import seaborn as sns
import matplotlib.pyplot as plt

if config.task == 'noise':
    labels = ['20', '30', '40', '50', '100']
    name = 'Noise'
else:
    labels = ['3', '5', '7', '9', '11']
    name = 'Blur'

# set font size
sns.set_theme(font_scale=1.5, font='Arial')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel(f'Predicted {name} Level')
plt.ylabel(f'Truth {name} Level')

# show acc f1
plt.title(f'Acc: {acc:.4f}%, F1: {f1:.4f}')

plt.savefig(f"{config.task}_confusion_matrix.pdf", bbox_inches='tight')