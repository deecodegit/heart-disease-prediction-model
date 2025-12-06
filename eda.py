import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------
# 1. load dataset (your actual windows path)
# -----------------------------------------------------------

df = pd.read_csv(
    r"C:\Users\Devanshi Sahu\data analytics\projects\heart-disease-ML-project\data\heart.csv"
)

# -----------------------------------------------------------
# 2. create folder to save plots
# -----------------------------------------------------------

output_folder = "static/plots"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------------------------------------
# 3. HISTOGRAMS FOR ALL COLUMNS
# -----------------------------------------------------------

for col in df.columns:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(f'{col} distribution')
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{col}_hist.png")
    plt.close()

print("histograms saved in static/plots/")

# -----------------------------------------------------------
# 4. CORRELATION HEATMAP
# -----------------------------------------------------------

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output_folder}/correlation_heatmap.png")
plt.close()

print("correlation heatmap saved!")
