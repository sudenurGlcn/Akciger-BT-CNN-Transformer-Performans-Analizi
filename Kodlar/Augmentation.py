import os
import cv2
import math
import random
from collections import defaultdict
from albumentations import (
    Compose, Rotate, ShiftScaleRotate, RandomBrightnessContrast,
    GaussianBlur, GaussNoise, ElasticTransform
)

DATASET_PATH = "K_fold_dataset2"
OUTPUT_PATH = "augmentation_dataset2"

transform = Compose([
    Rotate(limit=15, p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5),
    RandomBrightnessContrast(p=0.3),
    GaussianBlur(blur_limit=(3, 5), p=0.2),
    GaussNoise(std_range=(0.04, 0.05), mean_range=(0.0, 0.0), p=0.2),
    ElasticTransform(alpha=1, sigma=50, p=0.1),
])

def load_images(folder_path):
    images_by_class = defaultdict(list)
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            fpath = os.path.join(label_path, fname)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images_by_class[label].append((img, fname))
    return images_by_class

def save_image(img, path):
    cv2.imwrite(path, img)

def main():
    fold_data = {}
    total_per_class = defaultdict(int)

    for fold in range(1, 6):
        fold_path = os.path.join(DATASET_PATH, f"fold_{fold}", "train")
        images_by_class = load_images(fold_path)
        fold_data[fold] = images_by_class
        for label, items in images_by_class.items():
            total_per_class[label] += len(items)

    class_names = list(total_per_class.keys())
    num_classes = len(class_names)
    total_original = sum(total_per_class.values())
    target_total = total_original * 3
    target_per_class = math.ceil(target_total / num_classes)

    print(f"\n📊 Toplam orijinal örnek: {total_original}")
    print(f"🎯 Hedef toplam: {target_total}")
    print(f"🎯 Sınıf başına hedef: {target_per_class}")

    per_fold_aug_counts = {fold: defaultdict(int) for fold in range(1, 6)}

    for cls in class_names:
        existing_total = total_per_class[cls]
        needed_total = target_per_class
        total_to_add = max(0, needed_total - existing_total)

        fold_contributions = []
        for fold in range(1, 6):
            count = len(fold_data[fold].get(cls, []))
            fold_contributions.append((fold, count))

        total_fold_count = sum([c for _, c in fold_contributions]) or 1

        for fold, count in fold_contributions:
            share = count / total_fold_count
            per_fold_aug_counts[fold][cls] = int(share * total_to_add)

    final_counts = {fold: defaultdict(int) for fold in range(1, 6)}
    overall_counts = defaultdict(int)

    for fold in range(1, 6):
        print(f"\n🔧 Fold {fold} işleniyor...")

        train_output_path = os.path.join(OUTPUT_PATH, f"fold_{fold}", "train")
        test_input_path = os.path.join(DATASET_PATH, f"fold_{fold}", "test")
        test_output_path = os.path.join(OUTPUT_PATH, f"fold_{fold}", "test")

        os.makedirs(train_output_path, exist_ok=True)
        os.makedirs(test_output_path, exist_ok=True)

        for cls, img_list in fold_data[fold].items():
            cls_path = os.path.join(train_output_path, cls)
            os.makedirs(cls_path, exist_ok=True)

            # Orijinal verileri kaydet
            for img, fname in img_list:
                save_image(img, os.path.join(cls_path, f"orig_{fname}"))
                final_counts[fold][cls] += 1
                overall_counts[cls] += 1

            # Augment verileri oluştur
            aug_needed = per_fold_aug_counts[fold][cls]
            for i in range(aug_needed):
                img, fname = random.choice(img_list)
                aug_img = transform(image=img)['image']
                ext = os.path.splitext(fname)[1]  # .jpg, .png, vs.
                aug_name = f"aug_{os.path.splitext(fname)[0]}_{i}{ext}"
                save_image(aug_img, os.path.join(cls_path, aug_name))
                final_counts[fold][cls] += 1
                overall_counts[cls] += 1

        # Test setini kopyala
        for cls in os.listdir(test_input_path):
            src_cls_path = os.path.join(test_input_path, cls)
            dst_cls_path = os.path.join(test_output_path, cls)
            os.makedirs(dst_cls_path, exist_ok=True)

            for fname in os.listdir(src_cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(src_cls_path, fname)
                    dst_file = os.path.join(dst_cls_path, fname)
                    img = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        save_image(img, dst_file)

    print("\n📋 Fold Bazlı Dağılım (Orijinal + Augment):")
    for fold in range(1, 6):
        print(f"\nFold {fold}:")
        for cls in class_names:
            print(f"  {cls}: {final_counts[fold][cls]}")

    print("\n📊 Toplam Dağılım (5 fold toplamı):")
    for cls in class_names:
        print(f"  {cls}: {overall_counts[cls]}")

    print("\n✅ Veri artırımı ve sınıf bazlı dengeleme tamamlandı.")

if __name__ == "__main__":
    main()