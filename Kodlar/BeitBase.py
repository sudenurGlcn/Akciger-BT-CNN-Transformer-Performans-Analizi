"""BEiT Model K-Fold Cross Validation

BEiT (Bidirectional Encoder representation from Image Transformers) modeli ile
K-fold cross validation eğitimi - CUDA Optimized PyTorch
"""

import os
import numpy as np
import pandas as pd
import cv2
import gc
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.multiprocessing import freeze_support
from transformers import BeitFeatureExtractor, BeitForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import json
import pickle

# ============================================
# CUDA GPU AYARLARI VE DURUM KONTROLÜ
# ============================================

# Zaman damgası
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class IQDataset(Dataset):
    def __init__(self, dataframe, transform=None, feature_extractor=None, class_to_idx=None):
        """
        Args:
            dataframe (pd.DataFrame): Veri seti DataFrame'i
            transform (callable, optional): Görüntü dönüşümleri
            feature_extractor (callable, optional): BEiT feature extractor
            class_to_idx (dict, optional): Sınıf isimlerinden indekslere eşleme
        """
        self.dataframe = dataframe
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        label = self.dataframe.iloc[idx]['labels']

        # Görüntüyü oku
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Görüntü okunamadı: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # BEiT feature extractor ile görüntüyü işle
        if self.feature_extractor:
            # Görüntüyü PIL formatına çevir
            image_pil = transforms.ToPILImage()(image)
            inputs = self.feature_extractor(images=image_pil, return_tensors="pt")
            # pixel_values'ı al ve squeeze et
            image = inputs['pixel_values'].squeeze(0)

        # Label'ı sayısal değere dönüştür
        if self.class_to_idx is not None:
            label_idx = self.class_to_idx[label]
        else:
            label_idx = label

        return image, torch.tensor(label_idx, dtype=torch.long)


def main():
    # CUDA GPU'ları kontrol et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Tespit edilen GPU sayısı: {torch.cuda.device_count()}")

    # Global değişken - GPU durumu
    global GPU_AVAILABLE, DEVICE_NAME, HARDWARE_TYPE, MIXED_PRECISION
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE_NAME = device
    HARDWARE_TYPE = 'CUDA GPU' if GPU_AVAILABLE else 'CPU'

    print(f"🖥️  Kullanılacak Hardware: {HARDWARE_TYPE}")
    print(f"🎯 Device: {DEVICE_NAME}")

    if GPU_AVAILABLE:
        try:
            # GPU bilgilerini göster
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🔧 GPU: {gpu_name}")
            print(f"🔧 CUDA Version: {torch.version.cuda}")

            # Mixed precision training için scaler
            scaler = GradScaler()
            print("✅ Mixed Precision (AMP) aktif")
            MIXED_PRECISION = True

            # GPU memory bilgisi
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"💾 GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")

        except RuntimeError as e:
            print(f"❌ GPU ayarları hatası: {e}")
            device = torch.device('cpu')
            GPU_AVAILABLE = False
            DEVICE_NAME = device
            HARDWARE_TYPE = 'CPU'
            MIXED_PRECISION = False
    else:
        print("⚠️  CUDA GPU bulunamadı, CPU kullanılacak")
        MIXED_PRECISION = False

    # PyTorch CUDA durum kontrolü
    print(f"\n🔍 PyTorch CUDA Durumu:")
    print(f"   - CUDA Available: {torch.cuda.is_available()}")
    print(f"   - CUDA Version: {torch.version.cuda if GPU_AVAILABLE else 'N/A'}")
    print(f"   - Mixed Precision: {'Aktif' if MIXED_PRECISION else 'Pasif'}")

    # Ana dizin (K-fold bölünmüş veri seti)
    main_directory = r'C:\Users\Derin\PycharmProjects\LungClassification\Datasets\IQ-OTHNCCD_K-fold'

    # Sınıf isimleri
    class_labels = ['Benign cases', 'Malignant cases', 'Normal cases']  # IQ-OTHNCCD sınıfları
    # Sınıf indekslerini oluştur
    class_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    k_folds = 5

    # Sonuç klasörlerini oluştur
    model_dir = r'D:\Grup-1-LungClassification\Models\beit_models-IQ-OTHNCCD'
    results_dir =  r'D:\Grup-1-LungClassification\Models\beit_results_dir-IQ-OTHNCCD'

    # Klasörleri oluştur
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Veri dönüşümleri - optimize edilmiş transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def load_fold_data(fold_dir, split):
        """Belirli bir fold ve split için veri yükleme - optimize edilmiş"""
        print(f"📁 Veri yükleniyor: {split} set - {HARDWARE_TYPE}")
        filepaths = []
        labels = []

        split_dir = os.path.join(fold_dir, split)

        # Paralel işleme için glob kullan
        for label in class_labels:
            class_dir = os.path.join(split_dir, label)
            if os.path.exists(class_dir):
                # glob ile daha hızlı dosya listesi oluştur
                jpg_files = glob(os.path.join(class_dir, "*.jpg"))
                filepaths.extend(jpg_files)
                labels.extend([label] * len(jpg_files))

        return pd.DataFrame({
            "filepaths": filepaths,
            "labels": labels
        })

    def train(model, optimizer, train_loader, val_loader, device, scaler=None, model_dir=None, fold=None, epochs=50):
        """Model eğitim fonksiyonu - optimize edilmiş"""
        model.train()
        criterion = nn.CrossEntropyLoss()
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        # Model'i train moduna al ve gradyanları optimize et
        model.train()
        torch.backends.cudnn.benchmark = True  # CUDA optimizasyonu

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch in train_bar:
                inputs, labels = batch
                inputs = inputs.to(device, non_blocking=True)  # non_blocking=True ile asenkron transfer
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  # Daha hızlı sıfırlama

                if GPU_AVAILABLE and MIXED_PRECISION and scaler is not None:
                    with autocast():
                        outputs = model(pixel_values=inputs).logits
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(pixel_values=inputs).logits
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():  # Gereksiz gradyan hesaplamalarını önle
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                train_bar.set_postfix({
                    'loss': f'{train_loss/train_total:.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })

            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            # Validation 
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=GPU_AVAILABLE and MIXED_PRECISION):
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for batch in val_bar:
                    inputs, labels = batch
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(pixel_values=inputs).logits
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

                    val_bar.set_postfix({
                        'loss': f'{val_loss/val_total:.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })

            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

            # History'ye kaydet
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Early stopping kontrolü
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # En iyi modeli kaydet
                if model_dir is not None and fold is not None:
                    torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_fold_{fold}.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break

            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return history

    def evaluate(model, test_loader, device):
        """Model değerlendirme fonksiyonu"""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Evaluating')
            for batch in test_bar:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                if GPU_AVAILABLE and MIXED_PRECISION:
                    with autocast():
                        outputs = model(pixel_values=inputs).logits
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(pixel_values=inputs).logits
                    loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

                test_bar.set_postfix({
                    'loss': f'{test_loss/test_total:.4f}',
                    'acc': f'{100.*test_correct/test_total:.2f}%'
                })

        test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total

        return test_loss, test_acc

    def predict(model, test_loader, device):
        """Model tahmin fonksiyonu"""
        model.eval()
        predictions = []

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Predicting')
            for batch in test_bar:
                inputs, _ = batch
                inputs = inputs.to(device)

                if GPU_AVAILABLE and MIXED_PRECISION:
                    with autocast():
                        outputs = model(pixel_values=inputs).logits
                else:
                    outputs = model(pixel_values=inputs).logits

                predictions.append(outputs.cpu().numpy())

        return np.vstack(predictions)

    # K-fold cross validation
    fold_results = []
    all_histories = []

    print(f"\n{'=' * 60}")
    print(f"🚀 K-FOLD CROSS VALIDATION BAŞLIYOR")
    print(f"📍 Hardware: {HARDWARE_TYPE}")
    print(f"🎯 Device: {DEVICE_NAME}")
    if GPU_AVAILABLE:
        print(f"🔥 Mixed Precision: {'Aktif' if MIXED_PRECISION else 'Pasif'}")
    print(f"{'=' * 60}\n")

    for fold in range(1, k_folds + 1):
        print(f"\n{'=' * 50}")
        print(f"📂 FOLD {fold}/{k_folds}")
        print(f"🖥️  Hardware: {HARDWARE_TYPE}")
        print(f"{'=' * 60}")

        # CUDA memory temizleme
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"💾 Pre-fold GPU Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f}GB allocated, {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f}GB reserved")
        else:
            print(f"🖥️  CPU mode - Fold {fold}")

        # Fold dizinini belirle
        fold_dir = os.path.join(main_directory, f'fold_{fold}')

        # Train ve test verilerini yükle
        train_df = load_fold_data(fold_dir, 'train')
        test_df = load_fold_data(fold_dir, 'test')

        # Eğer veri yoksa atla
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"UYARI: Fold {fold} için veri bulunamadı, bu fold atlanıyor...")
            continue

        # Train setini train ve validation olarak böl (80-20)
        train_set, val_set = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            stratify=train_df['labels']
        )

        print(f"Train örnekleri: {len(train_set)}")
        print(f"Validation örnekleri: {len(val_set)}")
        print(f"Test örnekleri: {len(test_df)}")

        # Sabit batch size
        batch_size = 32
        print(f"📊 Batch size: {batch_size}")

        # BEiT model ve feature extractor'ı yükle
        print("\n🏗️  BEiT model yükleniyor...")
        feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224')
        model = BeitForImageClassification.from_pretrained(
            'microsoft/beit-base-patch16-224',
            num_labels=len(class_labels),
            ignore_mismatched_sizes=True
        )
        model = model.to(device)
        print("✅ BEiT model yüklendi")

        # DataLoader'ları oluştur - optimize edilmiş
        train_loader = DataLoader(
            IQDataset(train_set, transform, feature_extractor, class_to_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count()),  # CPU çekirdek sayısına göre optimize et
            pin_memory=True,
            persistent_workers=True,  # Worker'ları yeniden kullan
            prefetch_factor=2  # Veri ön yükleme
        )

        val_loader = DataLoader(
            IQDataset(val_set, transform, feature_extractor, class_to_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        test_loader = DataLoader(
            IQDataset(test_df, transform, feature_extractor, class_to_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        # Optimizer ve loss function
        optimizer = optim.Adamax(model.parameters(), lr=0.0004, weight_decay=0.01)

        # Model eğitimi
        print(f"\n🚀 Fold {fold} eğitimi başlıyor...")
        print(f"📍 Device: {DEVICE_NAME} ({HARDWARE_TYPE})")
        if GPU_AVAILABLE:
            print(f"🔥 GPU accelerated training aktif")
            print(f"⚡ Mixed Precision: {'Aktif' if MIXED_PRECISION else 'Pasif'}")

        history = train(model, optimizer, train_loader, val_loader, device,
                      scaler if GPU_AVAILABLE else None,
                      model_dir=model_dir,
                      fold=fold,
                      epochs=50)

        # Test değerlendirmesi
        print(f"\n📊 Fold {fold} test değerlendirmesi... ({HARDWARE_TYPE})")
        test_loss, test_acc = evaluate(model, test_loader, device)

        # Tahmin yapma
        print(f"🔮 Tahmin yapılıyor... ({HARDWARE_TYPE})")
        predictions = predict(model, test_loader, device)
        y_pred = np.argmax(predictions, axis=1)

        # True labels'ı al
        y_true = []
        for _, labels in test_loader:
            y_true.extend(labels.cpu().numpy())

        # Sınıf isimlerini al
        y_pred_labels = [class_labels[i] for i in y_pred]
        y_true_labels = [class_labels[i] for i in y_true]

        # Fold sonuçlarını kaydet
        fold_result = {
            'fold': fold,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'y_true': y_true_labels,
            'y_pred': y_pred_labels
        }

        print(f"\n💾 Fold {fold} sonuçları kaydediliyor...")
        save_fold_results(fold, history, fold_result, results_dir, timestamp, class_labels)
        print(f"✅ Fold {fold} sonuçları kaydedildi!")

        # Sonuçları listeye ekle
        fold_results.append(fold_result)
        all_histories.append(history)

        print(f"\n📊 Fold {fold} Sonuçları:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Eğitim tamamlandığı epoch: {len(history['train_loss'])}")

        # Classification report
        print(f"\n📋 Fold {fold} Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels))

        # Model kaydet
        model_path = os.path.join(model_dir, f'beit_model_fold_{fold}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model kaydedildi: {model_path}")

        # Memory temizleme
        del model
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
            print(f"🧹 GPU memory temizlendi")
        gc.collect()

        print(f"✅ Fold {fold} tamamlandı! ({HARDWARE_TYPE})")

    # Final GPU memory durumu
    if GPU_AVAILABLE:
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"\n💾 Final GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        print(f"🔥 GPU accelerated training tamamlandı!")
    else:
        print(f"\n🖥️  CPU training tamamlandı!")

    # Final özet sonuçları kaydet
    print("\n💾 Final özet sonuçlar kaydediliyor...")
    summary = {
        'timestamp': timestamp,
        'total_folds': len(fold_results),
        'average_accuracy': float(np.mean([r['test_accuracy'] for r in fold_results])),
        'std_accuracy': float(np.std([r['test_accuracy'] for r in fold_results])),
        'average_loss': float(np.mean([r['test_loss'] for r in fold_results])),
        'std_loss': float(np.std([r['test_loss'] for r in fold_results]))
    }

    with open(os.path.join(results_dir, f'summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    print("✅ Final özet sonuçlar kaydedildi!")

    print(f"\n{'=' * 60}")
    print(f"🎉 K-fold cross validation tamamlandı!")
    print(f"📍 Kullanılan Hardware: {HARDWARE_TYPE}")
    print(f"🎯 Device: {DEVICE_NAME}")
    if GPU_AVAILABLE:
        print(f"⚡ Mixed Precision: {'Aktif' if MIXED_PRECISION else 'Pasif'}")
    print(f"{'=' * 60}")

    print(f"\nSONUÇLAR KAYDEDİLDİ:")
    print(f"- Modeller: {model_dir}/ klasöründe")
    print(f"- Grafikler ve sonuçlar: {results_dir}/ klasöründe")
    print(f"- Zaman damgası: {timestamp}")
    print(f"- Optimizer: Adamax (Learning Rate: 0.0004)")
    print(f"- Early Stopping: Aktif (patience=5)")
    print(f"- Hardware: {HARDWARE_TYPE}")


def save_fold_results(fold, history, fold_result, results_dir, timestamp, class_labels):
    """Tek bir fold'un sonuçlarını kaydetme fonksiyonu"""
    fold_dir = os.path.join(results_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    # Fold sonuçlarını JSON formatında kaydet
    results_file = os.path.join(fold_dir, f'fold_{fold}_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'fold': fold,
            'test_accuracy': float(fold_result['test_accuracy']),
            'test_loss': float(fold_result['test_loss']),
            'y_true': fold_result['y_true'],
            'y_pred': fold_result['y_pred']
        }, f, indent=4)

    # Eğitim geçmişini pickle formatında kaydet
    history_file = os.path.join(fold_dir, f'fold_{fold}_history_{timestamp}.pkl')
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)

    # Loss ve accuracy grafiği
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold} - Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Fold {fold} - Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, f'fold_{fold}_history_{timestamp}.png'))
    plt.close()

    # Confusion matrix
    y_true = fold_result['y_true']
    y_pred = fold_result['y_pred']
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'Fold {fold} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, f'fold_{fold}_confusion_matrix_{timestamp}.png'))
    plt.close()


if __name__ == '__main__':
    freeze_support()
    main()