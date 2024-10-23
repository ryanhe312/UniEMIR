import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
from dataset import EMDiffusenDataset
from config import NoiseConfig as config

if os.path.exists('model') == False:
    os.mkdir('model')

if os.path.exists(fr'model/{config.task}') == False:
    os.mkdir(fr'model/{config.task}')

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5) 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = EMDiffusenDataset(data_root=config.train_data_root, corrupt=config.task)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4)
test_dataset = EMDiffusenDataset(data_root=config.test_data_root, corrupt=config.task)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

best_ac = 0.0
best_loss = 100.0

for epoch in range(config.epoch): 
    running_loss = 0.0
    testing_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 0:
            print(f'Epoch {epoch + 1} / {i}, loss: {running_loss / (i + 1)}')

    model.eval()
    with torch.no_grad():
        for j, data in enumerate(test_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, truth = torch.max(labels.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            testing_loss += loss.item()
            total_predictions += labels.size(0)
            correct_predictions += (predicted == truth).sum().item()
            
    print(f'Epoch {epoch + 1}, loss: {running_loss / (i + 1)}, test: {testing_loss / (j + 1)}, ac: {correct_predictions / total_predictions * 100: .2f}%')
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': testing_loss,
        }, fr'model/{config.task}/model_{epoch+1}.pth')
    if correct_predictions / total_predictions > best_ac and  testing_loss / (j + 1) < best_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': testing_loss,
        }, fr'model/{config.task}/model_best.pth')
        best_ac = correct_predictions / total_predictions
        best_loss = testing_loss / (j + 1)
        
print('Finished Training')