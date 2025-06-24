from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import shutil

# Yeni veri seti dizini
base_dir = 'CT-Scan Images Dataset'
sub_dirs = ['NonCancerous','Cancerous']

# Dosya yollarını al
file_paths = []
for sub_dir in sub_dirs:
    full_path = os.path.join(base_dir, sub_dir)
    for root, dirs, files in os.walk(full_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

# Dosya yollarını numpy array'e çevir
data = np.array(file_paths)

# Sınıf etiketlerini oluştur
labels = []
for file_path in file_paths:
    if 'NonCancerous' in file_path:
        labels.append('NonCancerous')
    elif 'Cancerous' in file_path:
        labels.append('Cancerous')
    else:
        labels.append('Unknown')

# Stratified 5 katlı k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Sonuçları kaydetmek için yeni klasör oluştur
output_dir = 'k_fold_results'
os.makedirs(output_dir, exist_ok=True)

fold = 1
for train_index, test_index in skf.split(data, labels):
    train_data, test_data = data[train_index], data[test_index]
    print(f"Fold {fold}:")
    print(f"Train: {train_data}, Test: {test_data}\n")
    
    # Her fold için ayrı klasör oluştur ve verileri kopyala
    fold_dir = os.path.join(output_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Sınıf isimleri
    class_names =  ['NonCancerous','Cancerous']
    
    # Eğitim verilerini kopyala
    train_dir = os.path.join(fold_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    for file_path in train_data:
        class_name = next((name for name in class_names if name in file_path), 'Unknown')
        shutil.copy(file_path, os.path.join(train_dir, class_name))
    
    # Test verilerini kopyala
    test_dir = os.path.join(fold_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    for file_path in test_data:
        class_name = next((name for name in class_names if name in file_path), 'Unknown')
        shutil.copy(file_path, os.path.join(test_dir, class_name))
    
    # Eğitim ve test veri sayısını kaydet
    with open(os.path.join(fold_dir, 'counts.txt'), 'w') as f_counts:
        f_counts.write(f'Train count: {len(train_data)}\n')
        f_counts.write(f'Test count: {len(test_data)}\n')
    
    fold += 1 