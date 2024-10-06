import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/ASUS/Downloads/UTS/UTS/cancer_patient_data_sets.csv'
df = pd.read_csv(file_path)

# 1. Mengatasi missing value
numeric_cols = ['Air Pollution', 'Alcohol use', 'Dust Allergy', 
                'OccuPational Hazards', 'Genetic Risk', 
                'Chronic Lung Disease', 'Balanced Diet', 
                'Obesity', 'Smoking', 'Passive Smoker', 
                'Chest Pain', 'Coughing of Blood', 'Fatigue', 
                'Weight Loss', 'Shortness of Breath', 'Wheezing', 
                'Swallowing Difficulty', 'Clubbing of Finger Nails', 
                'Frequent Cold', 'Dry Cough', 'Snoring', 'Age']  

non_numeric_cols = ['Patient Id', 'Level'] 

# Mengisi missing value hanya di kolom numerik dengan median dari kolom
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 2. Mengatasi outlier
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Mendeteksi outlier di luar range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR)))

# Mengganti outlier dengan nilai mean
for col in numeric_cols:
    mean_value = df[col].mean()
    df.loc[outliers[col], col] = mean_value

# 3. Normalisasi menggunakan Min-Max Scaling hanya pada kolom numerik kecuali 'Age'
scaler = MinMaxScaler()
numeric_cols_excluding_age = [col for col in numeric_cols if col != 'Age']
df_numeric_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_cols_excluding_age]), 
                                     columns=numeric_cols_excluding_age)

# Gabungkan kembali kolom non-numerik dan kolom 'Age'
df_preprocessed = pd.concat([df[non_numeric_cols].reset_index(drop=True), 
                             df_numeric_normalized.reset_index(drop=True), 
                             df[['Age']].reset_index(drop=True)], axis=1)

# 4. Simpan hasil preprocessing ke file CSV baru
output_file_path = 'C:/Users/ASUS/Downloads/UTS/UTS/cancer_patient_data_preprocessing.csv'
df_preprocessed.to_csv(output_file_path, index=False)

# 5. Train Decision Tree
target_column = 'Level'

# Encode non-numeric columns if necessary
le = LabelEncoder()
df_preprocessed[target_column] = le.fit_transform(df_preprocessed[target_column])

X = df_preprocessed[numeric_cols_excluding_age + ['Age']]  # Fitur, termasuk 'Age'
y = df_preprocessed[target_column]  # Target

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions using Decision Tree
dt_pred = dt_model.predict(X_test)

# Matriks kebingungan
conf_matrix = confusion_matrix(y_test, dt_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print classification report
report = classification_report(y_test, dt_pred, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

# Formatted Output
print("\nFormatted Output:")
for label in report.keys():
    if label not in ('accuracy', 'macro avg', 'weighted avg'):
        precision = report[label]['precision']
        recall = report[label]['recall']
        f1_score = report[label]['f1-score']
        support = report[label]['support']
        print(f"{label:>10} {precision:>10.2f} {recall:>10.2f} {f1_score:>10.2f} {support:>10}")
print(f"{'accuracy':>10} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {report['accuracy']:>10.2f} {len(y_test):>10}")

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
