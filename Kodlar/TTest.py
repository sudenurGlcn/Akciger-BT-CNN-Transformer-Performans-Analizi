import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1. Model ve veri seti isimleri
# ------------------------------
model_names = [
    'VGG16',
    'Resnet50',
    'EfficientNEtV2S',
    'SwinV2-CR-Small-224',
    'BEiT-base-patch16-224',
    'Convnext v2',
    'Coatnet-1'
]

dataset_names = [
    'IQ-OTH/NCCD Lung Cancer Dataset',
    'SPIE-AAPM-NCI Lung Nodule Classification Challenge Dataset',
    'CT-Scan Images Dataset'
]

# ------------------------------
# 2. Fold doğruluk değerleri
# ------------------------------
model_scores = {
    "IQ-OTH/NCCD Lung Cancer Dataset": {
        "VGG16": [0.98, 0.95, 0.98, 0.97, 0.95],
        "Resnet50": [0.85, 0.88, 0.86, 0.88, 0.87],
        "EfficientNEtV2S": [0.84, 0.83, 0.84, 0.88, 0.87],
        "SwinV2-CR-Small-224": [0.99, 0.99, 0.99, 1.00, 0.99],
        "BEiT-base-patch16-224": [0.99, 0.97, 1.00, 1.00, 1.00],
        "Convnext v2": [0.99, 0.96, 0.97, 0.99, 0.99],
        "Coatnet-1": [0.98, 0.96, 0.97, 0.99, 0.97],
    },
    "SPIE-AAPM-NCI Lung Nodule Classification Challenge Dataset": {
        "VGG16": [1.00,1.00, 1.00, 1.00, 0.99],
        "Resnet50": [0.89, 0.86, 0.87, 0.86, 0.89],
        "EfficientNEtV2S": [0.84, 0.81, 0.81, 0.84, 0.83],
        "SwinV2-CR-Small-224": [1.00,1.00, 1.00,1.00, 0.99],
        "BEiT-base-patch16-224": [1.00, 0.99, 1.00, 1.00, 0.99],
        "Convnext v2": [1.00,1.00, 1.00, 1.00, 1.00],
        "Coatnet-1": [1.00,1.00, 1.00, 1.00, 1.00],
    },
    "CT-Scan Images Dataset": {
        "VGG16": [0.99, 1.00, 1.00, 1.00, 0.99],
        "Resnet50": [0.97, 0.96, 0.99, 0.96, 1.00],
        "EfficientNEtV2S": [0.95, 0.97, 0.92, 0.97, 0.97],
        "SwinV2-CR-Small-224": [0.99,1.00, 1.00, 1.00, 0.99],
        "BEiT-base-patch16-224": [0.99, 1.00, 1.00, 0.97, 0.99],
        "Convnext v2": [0.99,1.00,1.00, 1.00, 1.00],
        "Coatnet-1": [0.99, 0.99, 1.00, 1.00, 1.00],
    }
}

# ------------------------------
# 3. T-testi ve K/B/Kyp karşılaştırmaları
# ------------------------------
comparison_results = {model1: {model2: [0, 0, 0] for model2 in model_names} for model1 in model_names}

for ds in dataset_names:
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:
                continue

            acc1 = model_scores[ds][model1]
            acc2 = model_scores[ds][model2]

            t_stat, p_val = ttest_ind(acc1, acc2, equal_var=False)

            if p_val < 0.1:
                if np.mean(acc1) > np.mean(acc2):
                    comparison_results[model1][model2][0] += 1  # model1 kazandı
                    comparison_results[model2][model1][2] += 1  # model2 kaybetti
                else:
                    comparison_results[model1][model2][2] += 1  # model1 kaybetti
                    comparison_results[model2][model1][0] += 1  # model2 kazandı
            else:
                comparison_results[model1][model2][1] += 1  # beraberlik
                comparison_results[model2][model1][1] += 1

# ------------------------------
# 4. Karşılaştırma tablosu oluşturma (K/B/Kyp)
# ------------------------------
summary_table = pd.DataFrame(index=model_names, columns=model_names)

for model1 in model_names:
    for model2 in model_names:
        if model1 == model2:
            summary_table.loc[model1, model2] = "-"
        else:
            k, b, kyp = comparison_results[model1][model2]
            summary_table.loc[model1, model2] = f"{k}/{b}/{kyp}"

# ------------------------------
# 5. Toplam K/B/Kyp hesaplama
# ------------------------------
total_wins = []
total_draws = []
total_losses = []

for model in model_names:
    win = sum(comparison_results[model][other][0] for other in model_names if other != model)
    draw = sum(comparison_results[model][other][1] for other in model_names if other != model)
    loss = sum(comparison_results[model][other][2] for other in model_names if other != model)

    total_wins.append(win)
    total_draws.append(draw)
    total_losses.append(loss)

summary_table.loc["Toplam Kazanma"] = [total_wins[i] if model != "Toplam Kazanma" else "-" for i, model in enumerate(model_names)]
summary_table.loc["Toplam Beraberlik"] = [total_draws[i] if model != "Toplam Beraberlik" else "-" for i, model in enumerate(model_names)]
summary_table.loc["Toplam Kaybetme"] = [total_losses[i] if model != "Toplam Kaybetme" else "-" for i, model in enumerate(model_names)]

# ------------------------------
# 6. CSV olarak kaydet
# ------------------------------
summary_table.to_csv("model_karsilastirma_t_tablosu6.csv", encoding='utf-8-sig')
print("Tablo 'model_karsilastirma_t_tablosu6.csv' olarak kaydedildi.")

# ------------------------------
# 7. PNG olarak görselleştir ve kaydet
# ------------------------------
fig, ax = plt.subplots(figsize=(len(model_names) * 1.5, 9))
ax.axis('tight')
ax.axis('off')

table_data = summary_table.fillna("").values
table_columns = summary_table.columns.tolist()
table_index = summary_table.index.tolist()

table = ax.table(
    cellText=table_data,
    rowLabels=table_index,
    colLabels=table_columns,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.3, 1.3)

plt.tight_layout()
plt.savefig("model_karsilastirma_tablosu6.png", dpi=300)
plt.close()
print("Tablo 'model_karsilastirma_tablosu6.png' olarak kaydedildi.")
