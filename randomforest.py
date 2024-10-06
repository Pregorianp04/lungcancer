import pickle
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = 'C:/Users/ASUS/Downloads/UTS/UTS/cancer_patient_data_sets.csv'
df = pd.read_csv(file_path)

# Asumsi df adalah DataFrame Anda
# Daftar kolom numerik termasuk 'Age'
numeric_cols = ['Air Pollution', 'Alcohol use', 'Dust Allergy', 
                'OccuPational Hazards', 'Genetic Risk', 
                'Chronic Lung Disease', 'Balanced Diet', 
                'Obesity', 'Smoking', 'Passive Smoker', 
                'Chest Pain', 'Coughing of Blood', 'Fatigue', 
                'Weight Loss', 'Shortness of Breath', 'Wheezing', 
                'Swallowing Difficulty', 'Clubbing of Finger Nails', 
                'Frequent Cold', 'Dry Cough', 'Snoring', 'Age']  # Sertakan 'Age' di sini

non_numeric_cols = ['Patient Id', 'Level']  # Kolom non-numerik


# Pisahkan kolom numerik dan non-numerik
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

# 1. Mengatasi missing value
missing_values = df.isnull().sum()
print("Missing value:")
print(missing_values)

# Mengisi missing value hanya di kolom numerik dengan median dari kolom
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Output setelah menangani missing value
print("\nData setelah menangani missing value (5 baris pertama):")
print(df.head())

# 2. Mengatasi outlier
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Mendeteksi outlier di luar range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR)))

# Menampilkan kolom dan nilai outlier
print("\nOutlier per kolom:")
outlier_columns = []  # Menyimpan nama kolom yang memiliki outlier
for col in numeric_cols:
    outlier_values = df.loc[outliers[col], col]
    if not outlier_values.empty:
        print(f"\nKolom {col}:")
        print(outlier_values)
        outlier_columns.append(col)  # Tambahkan kolom yang memiliki outlier

# Mengganti outlier dengan nilai mean
for col in outlier_columns:
    mean_value = df[col].mean()  # Hitung mean
    df[col] = df[col].astype(float)  # Pastikan kolom bertipe float
    df.loc[outliers[col], col] = mean_value  # Ganti outlier dengan mean

# Menampilkan data setelah mengganti outlier
outlier_data_after_replacement = df[outliers.any(axis=1)][outlier_columns]
print("\nData setelah mengganti outlier (hanya kolom yang di-outlier):")
print(outlier_data_after_replacement)

# 3. Normalisasi menggunakan Min-Max Scaling hanya pada kolom numerik kecuali 'Age'
scaler = MinMaxScaler()

# Filter kolom numerik yang tidak termasuk 'Age'
numeric_cols_excluding_age = [col for col in numeric_cols if col != 'Age']
df_numeric_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_cols_excluding_age]), 
                                     columns=numeric_cols_excluding_age)

# Gabungkan kembali kolom non-numerik dan kolom 'Age'
df_preprocessed = pd.concat([df[non_numeric_cols].reset_index(drop=True), 
                             df_numeric_normalized.reset_index(drop=True), 
                             df[['Age']].reset_index(drop=True)], axis=1)

# Output setelah normalisasi Min-Max Scaling
print("\nData setelah normalisasi Min-Max Scaling (5 baris pertama):")
print(df_preprocessed.head())

# 4. Simpan hasil preprocessing ke file CSV baru
output_file_path = 'C:/Users/ASUS/Downloads/UTS/UTS/cancer_patient_data_preprocessing.csv'
df_preprocessed.to_csv(output_file_path, index=False)

# Print lokasi file output
print(f"\nData yang sudah di-preprocessing disimpan di: {output_file_path}")

# RANDOM FOREST
# Load the preprocessed dataset
file_path = 'C:/Users/ASUS/Downloads/UTS/UTS/cancer_patient_data_preprocessing.csv'
df = pd.read_csv(file_path)

# Memisahkan fitur dan label
X = df.drop(['Patient Id', 'Level'], axis=1)  # Kolom 'Patient Id' dan 'Level' tidak termasuk fitur
y = df['Level']  # Kolom 'Level' sebagai label

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Menginisialisasi dan melatih model Random Forest
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train, y_train)

# Melakukan prediksi pada data pengujian
RF_pred = RF.predict(X_test)

# Evaluasi model
print("Confusion Matrix:")
cm = confusion_matrix(y_test, RF_pred)
print(cm)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='summer', xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix of Random Forest Classifier')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, RF_pred, target_names=le.classes_))

# Input Data Pasien
data = {
    'Patient Id': [],
    'Level': [],
    'Age': [],  
    'Gender': [],  
    'Air Pollution': [],
    'Alcohol use': [],
    'Dust Allergy': [],
    'OccuPational Hazards': [],
    'Genetic Risk': [],
    'Chronic Lung Disease': [],
    'Balanced Diet': [],
    'Obesity': [],
    'Smoking': [],
    'Passive Smoker': [],
    'Chest Pain': [],
    'Coughing of Blood': [],
    'Fatigue': [],
    'Weight Loss': [],
    'Shortness of Breath': [],
    'Wheezing': [],
    'Swallowing Difficulty': [],
    'Clubbing of Finger Nails': [],
    'Frequent Cold': [],
    'Dry Cough': [],
    'Snoring': []
}

# Membuat DataFrame kosong
df = pd.DataFrame(data)

# Fungsi untuk menentukan kategori risiko
def determine_risk_level(inputs):
    score = 0
    for key, value in inputs.items():
        if value > 0.5:
            score += 1
            
    if score <= 3:
        return 'Low'
    elif score <= 6:
        return 'Medium'
    else:
        return 'High'

# Menginput data pasien
while True:
    patient_id = input("Masukkan Patient Id (atau ketik 'exit' untuk keluar): ")
    if patient_id.lower() == 'exit':
        break

    age = int(input("Masukkan umur pasien: "))
    gender = input("Masukkan gender pasien (Laki-laki/Perempuan): ")

    inputs = {
        'Air Pollution': float(input("Masukkan nilai untuk Air Pollution (0 - 1): ")),
        'Alcohol use': float(input("Masukkan nilai untuk Alcohol use (0 - 1): ")),
        'Dust Allergy': float(input("Masukkan nilai untuk Dust Allergy (0 - 1): ")),
        'OccuPational Hazards': float(input("Masukkan nilai untuk OccuPational Hazards (0 - 1): ")),
        'Genetic Risk': float(input("Masukkan nilai untuk Genetic Risk (0 - 1): ")),
        'Chronic Lung Disease': float(input("Masukkan nilai untuk Chronic Lung Disease (0 - 1): ")),
        'Balanced Diet': float(input("Masukkan nilai untuk Balanced Diet (0 - 1): ")),
        'Obesity': float(input("Masukkan nilai untuk Obesity (0 - 1): ")),
        'Smoking': float(input("Masukkan nilai untuk Smoking (0 - 1): ")),
        'Passive Smoker': float(input("Masukkan nilai untuk Passive Smoker (0 - 1): ")),
        'Chest Pain': float(input("Masukkan nilai untuk Chest Pain (0 - 1): ")),
        'Coughing of Blood': float(input("Masukkan nilai untuk Coughing of Blood (0 - 1): ")),
        'Fatigue': float(input("Masukkan nilai untuk Fatigue (0 - 1): ")),
        'Weight Loss': float(input("Masukkan nilai untuk Weight Loss (0 - 1): ")),
        'Shortness of Breath': float(input("Masukkan nilai untuk Shortness of Breath (0 - 1): ")),
        'Wheezing': float(input("Masukkan nilai untuk Wheezing (0 - 1): ")),
        'Swallowing Difficulty': float(input("Masukkan nilai untuk Swallowing Difficulty (0 - 1): ")),
        'Clubbing of Finger Nails': float(input("Masukkan nilai untuk Clubbing of Finger Nails (0 - 1): ")),
        'Frequent Cold': float(input("Masukkan nilai untuk Frequent Cold (0 - 1): ")),
        'Dry Cough': float(input("Masukkan nilai untuk Dry Cough (0 - 1): ")),
        'Snoring': float(input("Masukkan nilai untuk Snoring (0 - 1): "))
    }
    
    risk_level = determine_risk_level(inputs)

    new_patient_data = pd.DataFrame({
        'Patient Id': [patient_id],
        'Level': [risk_level],
        'Age': [age],
        'Gender': [gender],
        **inputs
    })

    df = pd.concat([df, new_patient_data], ignore_index=True)
    
    print(f"\nData pasien berhasil ditambahkan. Tingkat risiko: {risk_level}\n")

# Menampilkan Data Pasien
print("\nData Pasien:")
print(df)