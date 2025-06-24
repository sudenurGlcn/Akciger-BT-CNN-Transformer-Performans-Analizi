import os
import shutil
import cv2
import numpy as np
import openpyxl
from skimage.metrics import structural_similarity as ssim

def load_grayscale_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (256, 256)) if img is not None else None

def compute_ssim(img1, img2):
    return ssim(img1, img2)

def compute_histogram_difference(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def detect_nodule_slices(center_path, all_slices, center_idx, max_range=15):
    center_img = load_grayscale_image(center_path)
    if center_img is None:
        return []

    selected_indices = [center_idx]
    ssim_threshold = 0.6
    hist_threshold = 0.3

    # Geriye doğru
    for i in range(center_idx - 1, center_idx - max_range - 1, -1):
        path = all_slices.get(i)
        if not path:
            break
        img = load_grayscale_image(path)
        if img is None:
            break
        ssim_score = compute_ssim(center_img, img)
        hist_diff = compute_histogram_difference(center_img, img)
        if ssim_score < ssim_threshold or hist_diff > hist_threshold:
            break
        selected_indices.append(i)

    # İleriye doğru
    for i in range(center_idx + 1, center_idx + max_range + 1):
        path = all_slices.get(i)
        if not path:
            break
        img = load_grayscale_image(path)
        if img is None:
            break
        ssim_score = compute_ssim(center_img, img)
        hist_diff = compute_histogram_difference(center_img, img)
        if ssim_score < ssim_threshold or hist_diff > hist_threshold:
            break
        selected_indices.append(i)

    return sorted(selected_indices)

def process_scan(ct_folder, center_idx, output_folder):
    input_folder = os.path.join("png_output/malignant", ct_folder)
    if not os.path.exists(input_folder):
        print(f"Klasör bulunamadı: {input_folder}")
        return

    all_pngs = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    indexed_paths = {}
    for f in all_pngs:
        try:
            idx = int(f.split('-')[1].split('.')[0])
            indexed_paths[idx] = os.path.join(input_folder, f)
        except:
            continue

    center_path = indexed_paths.get(center_idx)
    if not center_path:
        print(f"Merkez dilim bulunamadı: {ct_folder} - {center_idx}")
        return

    selected = detect_nodule_slices(center_path, indexed_paths, center_idx)
    os.makedirs(output_folder, exist_ok=True)
    for idx in selected:
        src = indexed_paths[idx]
        dst = os.path.join(output_folder, os.path.basename(src))
        shutil.copy(src, dst)
    print(f"{ct_folder}: {len(selected)} dilim kopyalandı.")

# Ana script
excel_path = "TestSet_NoduleData_PublicRelease_wTruth (1).xlsx"
output_root = "Auto_Selected_Nodule_Slices"
wb = openpyxl.load_workbook(excel_path)
ws = wb.active

for row in ws.iter_rows(min_row=2, values_only=True):
    scan_number = row[0]
    center_img_idx = row[3]
    diagnosis = row[4]

    if not (scan_number and center_img_idx and diagnosis):
        continue
    if "primary lung cancer" not in diagnosis.lower():
        continue
    if "CT" in scan_number.upper():
        ct_num = scan_number.upper().split("CT")[-1].zfill(3)
        ct_folder = f"CT-{ct_num}"
    else:
        continue

    output_folder = os.path.join(output_root, ct_folder)
    process_scan(ct_folder, int(center_img_idx), output_folder)

