import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
output_file_path = 'C:/Users/ASUS/Downloads/UTS/UTS/cancer_patient_data_preprocessed.csv'
df_preprocessed.to_csv(output_file_path, index=False)

# Print lokasi file output
print(f"\nData yang sudah di-preprocessing disimpan di: {output_file_path}")

# 5. Train Naive Bayes
target_column = 'Level'  # Ganti ini dengan nama kolom target yang sesuai

# Pastikan target_column ada dalam kolom DataFrame
if target_column in df_preprocessed.columns:
    # Menghapus kolom non-numerik (termasuk 'Patient Id') dari fitur
    non_feature_cols = non_numeric_cols + [target_column]
    numeric_cols = [col for col in df_preprocessed.columns if col not in non_feature_cols]
    
    # Encode non-numeric columns if necessary
    le = LabelEncoder()
    for col in non_numeric_cols:
        df_preprocessed[col] = le.fit_transform(df_preprocessed[col])

    X = df_preprocessed[numeric_cols]  # Fitur, hanya kolom numerik
    y = df_preprocessed[target_column]  # Target

    # Bagi data menjadi data latih dan data uji (80% latih, 20% uji)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model Gaussian Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Predictions using Naive Bayes
    nb_pred = nb_model.predict(X_test)

    # Matriks kebingungan
    conf_matrix = confusion_matrix(y_test, nb_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix - Naive Bayes')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Print classification report
    report = classification_report(y_test, nb_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, nb_pred))
    
    # If you want to format the output similar to the example you provided
    print("\nFormatted Output:")
    for label in report.keys():
        if label not in ('accuracy', 'macro avg', 'weighted avg'):
            precision = report[label]['precision']
            recall = report[label]['recall']
            f1_score = report[label]['f1-score']
            support = report[label]['support']
            print(f"{label:>10} {precision:>10.2f} {recall:>10.2f} {f1_score:>10.2f} {support:>10}")
    print(f"{'accuracy':>10} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {len(y_test):>10}")



# KLASIFIKASI NAIVE BAIYES
# DataFrame untuk data pasien
data = {
    'Patient Id': [],
    'Level': [],
    'Age': [],  # Tambahkan kolom Age
    'Gender': [],  # Tambahkan kolom Gender
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
    # Kriteria untuk menentukan level risiko
    score = 0
    
    # Menghitung skor berdasarkan input
    for key, value in inputs.items():
        if value > 0.5:
            score += 1
            
    # Menentukan level berdasarkan skor
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

    # Tambahkan input untuk Age dan Gender
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
    
    # Menentukan kategori risiko
    risk_level = determine_risk_level(inputs)

    # Menambahkan data pasien ke DataFrame menggunakan pd.concat
    new_patient_data = pd.DataFrame({
        'Patient Id': [patient_id],
        'Level': [risk_level],
        'Age': [age],  # Menambahkan umur pasien
        'Gender': [gender],  # Menambahkan gender pasien
        **{key: [value] for key, value in inputs.items()}
    })
    
    # Menggabungkan DataFrame
    df = pd.concat([df, new_patient_data], ignore_index=True)

    print(f"\nKategori Risiko untuk Pasien {patient_id}: {risk_level}\n")

# Menampilkan DataFrame akhir
print("Data Pasien:")
print(df)
