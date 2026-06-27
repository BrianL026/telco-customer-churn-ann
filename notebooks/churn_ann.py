import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("[INFO] Memuat dataset...")
    try:
        # Pastikan file CSV berada di folder data
        df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        print("[ERROR] File dataset tidak ditemukan. Pastikan nama file dan path benar.")
        return

    print("[INFO] Memulai Data Preprocessing...")
    
    # 1. Hapus identifier yang tidak memiliki nilai prediktif
    df.drop('customerID', axis=1, inplace=True)

    # 2. Tangani missing values terselubung
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 3. Label Encoding untuk fitur biner
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col == 'gender':
            df[col] = df[col].map({'Female': 0, 'Male': 1})
        else:
            df[col] = df[col].map({'No': 0, 'Yes': 1})

    # 4. One-Hot Encoding untuk fitur kategorikal multikelas
    multiclass_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                       'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multiclass_cols, drop_first=True)

    # 5. Pisahkan Fitur (X) dan Target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 6. Feature Scaling (Standardisasi)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # 7. Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[INFO] Preprocessing selesai. Dimensi fitur: {X.shape}")
    print(f"[INFO] Distribusi Data -> Latih: {X_train.shape[0]}, Uji: {X_test.shape[0]}")

    print("\n[INFO] Menginisialisasi arsitektur Artificial Neural Network...")
    print("       - Hidden Layers: 2 (64 neuron, 32 neuron)")
    print("       - Activation: ReLU")
    print("       - Solver: Adam\n")
    
    ann_model = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        activation='relu',           
        solver='adam',               
        max_iter=500,                
        random_state=42,
        early_stopping=True          
    )

    print("[INFO] Memulai proses training model (mencari bobot optimal)...")
    ann_model.fit(X_train, y_train)
    print("[INFO] Training selesai!\n")

    print("[INFO] Melakukan pengujian pada data uji...")
    y_pred = ann_model.predict(X_test)

    # Evaluasi Hasil
    accuracy = accuracy_score(y_test, y_pred)
    print("="*40)
    print(f" HASIL EVALUASI MODEL ANN")
    print("="*40)
    print(f"Akurasi Keseluruhan : {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("="*40)

if __name__ == "__main__":
    main()