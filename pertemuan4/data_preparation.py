# 1 Collection of data
import os 
import pandas as pd
import seaborn as sns

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "dataset", "kelulusan_mahasiswa.csv")
output_dir = os.path.join(base_dir, "output")

df = pd.read_csv(dataset_path)
print(df.info())
print(df.head())

# 2 Data Cleaning
print(df.isnull().sum())
df = df.drop_duplicates()

sns.boxplot(x=df['IPK'])

# 3 Data Analysis and Visualization
print(df.describe())
sns.histplot(df['IPK'], bins=10, kde=True)
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

# 4 Feature Engineering
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv(os.path.join(output_dir, "processed_kelulusan.csv"), index=False)

# 5 Data Splitting
from sklearn.model_selection import train_test_split

X = df.drop('Lulus', axis=1)
y = df['Lulus']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42) # takeout stratify for validation/test split

print(X_train.shape, X_val.shape, X_test.shape)