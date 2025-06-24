# -*- coding: utf-8 -*-


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import copy
import timm  # yeni

# Yapılandırma
BASE_DIR = r'C:\Users\Derin\PycharmProjects\LungClassification\Datasets\IQ-OTHNCCD_K-fold'

SAVE_BASE_DIR = r'D:\Grup-1-LungClassification\Models\ConvNeXtV2-IQ-QTHNCCD-Dataset'
RESULTS_DIR = os.path.join(SAVE_BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FOLDS = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 50
PATIENCE = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Eğitim fonksiyonu 
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    counter = 0
    epoch_durations = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 30)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)
        print(f'Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds')

        if counter >= patience:
            print(f'\nEarly stopping activated. Validation loss did not improve for {patience} epochs.')
            break

    end_time = time.time()
    total_time = end_time - start_time
    print(f'\nTotal training time: {total_time:.2f} seconds')

    model.load_state_dict(best_model_wts)
    return model, history, total_time, epoch_durations

# Ana döngü
fold_accuracies = []
all_cm = None
all_reports = []
all_y_true, all_y_pred = [], []
total_all_training_time = 0.0

for fold in FOLDS:
    print(f'\nStarting {fold}...')
    train_dir = os.path.join(BASE_DIR, fold, 'train')
    test_dir = os.path.join(BASE_DIR, fold, 'test')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    class_names = full_dataset.classes

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    }

    # Model (timm kullanarak ConvNeXtV2)
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=True, num_classes=len(class_names))
    model = model.to(device)

    # Dondurulacak layer'lar var ise burada ayarlayın (Opsiyonel)
    # Örneğin tüm parametreleri açık bırakmak için bu kısmı yorum satırı yapabilirsiniz
    # for param in model.parameters():
    #     param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=1e-4)

    model, history, total_train_time, epoch_durations = train_model(model, dataloaders, criterion, optimizer, num_epochs=EPOCHS, patience=PATIENCE)
    total_all_training_time += total_train_time

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true_labels = [class_names[i] for i in y_true]
    y_pred_labels = [class_names[i] for i in y_pred]

    all_y_true.extend(y_true_labels)
    all_y_pred.extend(y_pred_labels)

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)
    if all_cm is None:
        all_cm = np.zeros_like(cm)
    all_cm += cm

    report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, digits=4, output_dict=False)
    report_dict = classification_report(y_true_labels, y_pred_labels, target_names=class_names, digits=4, output_dict=True)
    all_reports.append(report_dict)

    accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    fold_accuracies.append(accuracy)
    print(f'\n{fold} Accuracy: {accuracy:.2f}%')

    fold_dir = os.path.join(SAVE_BASE_DIR, fold)
    os.makedirs(fold_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{fold.upper()} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    with open(os.path.join(fold_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_true_labels, y_pred_labels, target_names=class_names, digits=4))
        f.write(f"\n{fold} Training time: {total_train_time:.2f} seconds\n")
        f.write(f"Epochs run: {len(history['train_loss'])}\n")
        for idx, duration in enumerate(epoch_durations, start=1):
            f.write(f"Epoch {idx} duration: {duration:.2f} seconds\n")

    plt.figure()
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title(f'{fold} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(fold_dir, 'accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{fold} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(fold_dir, 'loss.png'))
    plt.close()

overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=class_names)

plt.figure(figsize=(10, 8))
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Overall Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(RESULTS_DIR, 'overall_confusion_matrix.png'), dpi=300)
plt.close()

print(f'\nAverage accuracy over all folds: {np.mean(fold_accuracies):.2f}%')
print(f'Total training time over all folds: {total_all_training_time:.2f} seconds')