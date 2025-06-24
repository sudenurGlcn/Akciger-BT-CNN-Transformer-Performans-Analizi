import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Kullanılan cihaz: {device}")

# Veri dönüşümü
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Klasör yolları
root_dir = r'C:\Users\Derin\PycharmProjects\LungClassification\Datasets\IQ-OTHNCCD_K-fold'
output_base = r'D:\Grup-1-LungClassification\Models\SwinModels-IQ-OTHNCCD_K-fold'
os.makedirs(output_base, exist_ok=True)
result_dir = os.path.join(output_base, 'result')
os.makedirs(result_dir, exist_ok=True)

folds = [f"fold_{i}" for i in range(1, 6)]
all_metrics, all_train_losses, all_test_losses = [], [], []
all_train_accuracies, all_test_accuracies = [], []

# Zaman kayıtları için
all_fold_epoch_times = {}  # fold: [epoch süreleri]
all_fold_total_times = {}  # fold: toplam epoch süresi

# Her fold için eğitim
for fold in folds:
    print(f"\n🔁 {fold.upper()} işleniyor...")

    fold_output = os.path.join(output_base, fold)
    os.makedirs(fold_output, exist_ok=True)

    train_dataset = datasets.ImageFolder(os.path.join(root_dir, fold, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(root_dir, fold, "test"), transform=transform)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # SwinV2 modelini oluştur
    model = timm.create_model('swinv2_cr_small_224', pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=1e-4)
    num_epochs = 50
    patience = 5

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    best_loss = float('inf')
    epochs_no_improve = 0

    epoch_times = []  # Bu fold için epoch süreleri

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_dataset)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # Değerlendirme
        model.eval()
        test_loss, correct = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        test_acc = correct / len(test_dataset)
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(test_acc)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(f"📘 Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f} | Test Acc: {test_acc:.4f} | Epoch Süresi: {epoch_duration:.2f} sn")

        if test_losses[-1] < best_loss:
            best_loss = test_losses[-1]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("🛑 Early stopping")
                break

    fold_total_time = sum(epoch_times)
    all_fold_epoch_times[fold] = epoch_times
    all_fold_total_times[fold] = fold_total_time

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
    print(f"Toplam Epoch Süresi: {fold_total_time:.2f} saniye")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {fold}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(fold_output, 'confusion_matrix.png'))
    plt.close()

    with open(os.path.join(fold_output, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    torch.save(model.state_dict(), os.path.join(fold_output, f'model_{fold}.pt'))
    print(f"💾 Model kaydedildi: {os.path.join(fold_output, f'model_{fold}.pt')}")

# Ortalama değerlendirme
print("\n📦 Genel Confusion Matrix hazırlanıyor...")
all_y_true, all_y_pred = [], []

for fold in folds:
    test_dataset = datasets.ImageFolder(os.path.join(root_dir, fold, "test"), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = timm.create_model('swinv2_cr_small_224', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(output_base, fold, f'model_{fold}.pt')))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Genel Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(result_dir, 'avg_confusion_matrix.png'))
plt.close()

# Ortalama metrikler
avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
avg_precision = np.mean([m['precision'] for m in all_metrics])
avg_recall = np.mean([m['recall'] for m in all_metrics])
avg_f1 = np.mean([m['f1'] for m in all_metrics])

with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
    f.write(f"""Ortalama Metrikler (5 Fold):

Accuracy:  {avg_accuracy*100:.2f}%
Precision: {avg_precision:.4f}
Recall:    {avg_recall:.4f}
F1 Score:  {avg_f1:.4f}
""")

# Epoch süreleri dosyasını yaz
with open(os.path.join(result_dir, 'epoch_times.txt'), 'w') as f:
    f.write("Fold bazında epoch süreleri (saniye cinsinden):\n\n")
    for fold in folds:
        f.write(f"{fold}:\n")
        for i, t in enumerate(all_fold_epoch_times[fold]):
            f.write(f"  Epoch {i+1}: {t:.2f} sn\n")
        f.write(f"Toplam epoch süresi: {all_fold_total_times[fold]:.2f} sn\n\n")

# Ortalama grafikler
avg_train_losses = np.nanmean([np.pad(l, (0, 50 - len(l)), constant_values=np.nan) for l in all_train_losses], axis=0)
avg_test_losses = np.nanmean([np.pad(l, (0, 50 - len(l)), constant_values=np.nan) for l in all_test_losses], axis=0)
avg_train_accuracies = np.nanmean([np.pad(a, (0, 50 - len(a)), constant_values=np.nan) for a in all_train_accuracies], axis=0)
avg_test_accuracies = np.nanmean([np.pad(a, (0, 50 - len(a)), constant_values=np.nan) for a in all_test_accuracies], axis=0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(avg_train_losses, label='Train Loss')
plt.plot(avg_test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.title('Ortalama Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(avg_train_accuracies, label='Train Acc')
plt.plot(avg_test_accuracies, label='Test Acc')
plt.xlabel('Epoch')
plt.title('Ortalama Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'avg_loss_accuracy.png'))
plt.close()
