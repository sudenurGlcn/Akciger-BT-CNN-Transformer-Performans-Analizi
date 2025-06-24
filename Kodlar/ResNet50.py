
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Kullanılan cihaz: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

root_dir = r'C:\Users\Derin\PycharmProjects\LungClassification\Datasets\IQ-OTHNCCD_K-fold'
output_base = r'D:\Grup-1-LungClassification\Models\ResNetModels-IQ-OTHNCCD_K-fold'
os.makedirs(output_base, exist_ok=True)
result_dir = os.path.join(output_base, 'result')
os.makedirs(result_dir, exist_ok=True)

folds = [f"fold_{i}" for i in range(1, 6)]
all_metrics = []
all_train_losses = []
all_test_losses = []
all_train_accuracies = []
all_test_accuracies = []

for fold in folds:
    print(f"\n🔁 {fold.upper()} işleniyor...")

    fold_output = os.path.join(output_base, fold)
    os.makedirs(fold_output, exist_ok=True)

    train_path = os.path.join(root_dir, fold, "train")
    test_path = os.path.join(root_dir, fold, "test")

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.fc.parameters(), lr=0.0001)

    num_epochs = 50
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    epoch_times = []

    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        running_loss, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(correct / len(train_dataset))

        model.eval()
        test_loss, correct = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        epoch_test_loss = test_loss / len(test_loader)
        test_losses.append(epoch_test_loss)
        test_accuracies.append(correct / len(test_dataset))

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f} | Time: {epoch_duration:.2f}s")

        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1} (no improvement in test loss)")
                break

    total_time = sum(epoch_times)
    with open(os.path.join(fold_output, 'training_time.txt'), 'w') as f:
        for i, t in enumerate(epoch_times):
            f.write(f"Epoch {i+1}: {t:.2f} seconds\n")
        f.write(f"\nToplam süre: {total_time:.2f} seconds\n")

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = test_accuracies[-1]

    all_metrics.append({
        'fold': fold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    })

    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)
    all_train_accuracies.append(train_accuracies)
    all_test_accuracies.append(test_accuracies)

    print(f"\n📊 {fold} sonuçları:")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Grafikler
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.title("Train Loss & Accuracy")
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(fold_output, 'train_loss_accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(test_losses, label='Test Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title("Test Loss & Accuracy")
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(fold_output, 'test_loss_accuracy.png'))
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {fold}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_output, 'confusion_matrix.png'))
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(fold_output, 'classification_report.txt'), 'w') as f:
        f.write(report)

    model_path = os.path.join(fold_output, f'model_{fold}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model kaydedildi: {model_path}")

# Genel Confusion Matrix
print("\n📦 Genel Confusion Matrix hazırlanıyor...")
all_y_true, all_y_pred = [], []

for fold in folds:
    test_path = os.path.join(root_dir, fold, "test")
    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(os.path.join(output_base, fold, f'model_{fold}.pt')))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Genel Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'avg_confusion_matrix.png'))
plt.close()

avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
avg_precision = np.mean([m['precision'] for m in all_metrics])
avg_recall = np.mean([m['recall'] for m in all_metrics])
avg_f1 = np.mean([m['f1'] for m in all_metrics])

avg_report = f"""Ortalama Metrikler (5 Fold):

Accuracy:  {avg_accuracy*100:.2f}%
Precision: {avg_precision:.4f}
Recall:    {avg_recall:.4f}
F1 Score:  {avg_f1:.4f}
"""

with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
    f.write(avg_report)

avg_train_losses = np.mean([np.pad(losses, (0, 50 - len(losses)), constant_values=np.nan) for losses in all_train_losses], axis=0)
avg_test_losses = np.mean([np.pad(losses, (0, 50 - len(losses)), constant_values=np.nan) for losses in all_test_losses], axis=0)
avg_train_accuracies = np.mean([np.pad(accs, (0, 50 - len(accs)), constant_values=np.nan) for accs in all_train_accuracies], axis=0)
avg_test_accuracies = np.mean([np.pad(accs, (0, 50 - len(accs)), constant_values=np.nan) for accs in all_test_accuracies], axis=0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(avg_train_losses, label='Avg Train Loss')
plt.plot(avg_test_losses, label='Avg Test Loss')
plt.title("Ortalama Loss Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(avg_train_accuracies, label='Avg Train Accuracy')
plt.plot(avg_test_accuracies, label='Avg Test Accuracy')
plt.title("Ortalama Accuracy Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'avg_loss_accuracy.png'))
plt.close()

