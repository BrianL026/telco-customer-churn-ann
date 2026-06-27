import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'churn_model.joblib')
SCALER_FILE = os.path.join(BASE_DIR, 'scaler.joblib')
FEATURES_FILE = os.path.join(BASE_DIR, 'model_features.joblib')

class ChurnModelHandler:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.features = None
        self.training_history = None
        
        # Load model if it already exists
        self.load_model()

    def is_model_trained(self):
        return self.model is not None and self.scaler is not None and self.features is not None

    def load_model(self):
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(FEATURES_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
                self.features = joblib.load(FEATURES_FILE)
                print("[INFO] Model and preprocessors loaded successfully.")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to load model files: {e}")
        return False

    def save_model(self):
        if self.is_model_trained():
            joblib.dump(self.model, MODEL_FILE)
            joblib.dump(self.scaler, SCALER_FILE)
            joblib.dump(self.features, FEATURES_FILE)
            print("[INFO] Model and preprocessors saved to disk.")

    def preprocess_raw_data(self, df):
        # 1. Hapus customerID
        df_clean = df.copy()
        if 'customerID' in df_clean.columns:
            df_clean.drop('customerID', axis=1, inplace=True)

        # 2. Tangani missing values di TotalCharges
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)

        # 3. Label Encoding untuk fitur biner
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df_clean.columns:
                if col == 'gender':
                    df_clean[col] = df_clean[col].map({'Female': 0, 'Male': 1})
                else:
                    df_clean[col] = df_clean[col].map({'No': 0, 'Yes': 1})
                    
        # SeniorCitizen is already 0/1 in the dataset, leave it

        # 4. One-Hot Encoding untuk fitur kategorikal multikelas
        multiclass_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                           'Contract', 'PaymentMethod']
        
        # Let's align with drop_first=True as used in the original script
        df_clean = pd.get_dummies(df_clean, columns=multiclass_cols, drop_first=True)
        return df_clean

    def train(self, on_epoch_end_callback=None):
        """
        Trains the ANN MLPClassifier model.
        Returns: evaluation metrics dict, training history dict
        """
        print("[INFO] Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Preprocess
        df_processed = self.preprocess_raw_data(df)
        
        # Split features and target
        X = df_processed.drop('Churn', axis=1, errors='ignore')
        if 'Churn' in df_processed.columns:
            y = df_processed['Churn'].map({'No': 0, 'Yes': 1})
        else:
            # Fallback if Churn was already converted
            y = df['Churn'].map({'No': 0, 'Yes': 1})
            
        self.features = list(X.columns)

        # Feature Scaling (Standardisasi)
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.scaler = StandardScaler()
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        # Train-Test Split (80% Training, 20% Testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize neural network
        # MLPClassifier from sklearn does not support step-by-step progress natively in fit unless we use partial_fit.
        # But we can capture training loss curve via loss_curve_ attribute after fitting.
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            activation='relu',           
            solver='adam',               
            max_iter=500,                
            random_state=42,
            early_stopping=True
        )

        print("[INFO] Fitting model...")
        self.model.fit(X_train, y_train)
        print("[INFO] Model fitted.")

        # Save the model
        self.save_model()

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'loss_curve': self.model.loss_curve_,
            'n_iter': self.model.n_iter_
        }
        
        return metrics

    def predict(self, input_dict):
        """
        Predicts churn for a single customer input dictionary.
        input_dict should have raw categorical values like 'Yes', 'No', 'Female', 'Fiber optic', etc.
        """
        if not self.is_model_trained():
            raise Exception("Model is not trained yet. Please train the model first.")

        # Convert single dictionary to dataframe
        df_single = pd.DataFrame([input_dict])

        # Preprocess the single row to match training format
        # 1. Parse/Fix numerical inputs
        df_single['tenure'] = pd.to_numeric(df_single['tenure'], errors='coerce').fillna(0)
        df_single['MonthlyCharges'] = pd.to_numeric(df_single['MonthlyCharges'], errors='coerce').fillna(0.0)
        df_single['TotalCharges'] = pd.to_numeric(df_single['TotalCharges'], errors='coerce').fillna(0.0)

        # 2. Binary variables mapping
        binary_mappings = {
            'gender': {'Female': 0, 'Male': 1},
            'SeniorCitizen': {'Yes': 1, 'No': 0, 1: 1, 0: 0, '1': 1, '0': 0},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0}
        }
        for col, mapping in binary_mappings.items():
            if col in df_single.columns:
                val = df_single.loc[0, col]
                df_single[col] = mapping.get(val, 0)

        # 3. Handle multiclass and align with model features list
        # We construct a new DataFrame with the exact columns as self.features initialized to 0
        X_single = pd.DataFrame(0, index=[0], columns=self.features, dtype=np.float64)

        # Copy numeric and binary columns that are directly matching
        direct_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                       'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges']
        for col in direct_cols:
            if col in df_single.columns:
                X_single.loc[0, col] = df_single.loc[0, col]

        # Map multiclass values to dummy columns
        multiclass_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                           'Contract', 'PaymentMethod']
        
        for col in multiclass_cols:
            if col in df_single.columns:
                val = df_single.loc[0, col]
                # E.g., 'MultipleLines_Yes'
                dummy_col = f"{col}_{val}"
                if dummy_col in X_single.columns:
                    X_single.loc[0, dummy_col] = 1.0

        # 4. Scale numerical columns
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        X_single[numerical_cols] = self.scaler.transform(X_single[numerical_cols])

        # 5. Run prediction
        pred_class = int(self.model.predict(X_single)[0])
        pred_proba = float(self.model.predict_proba(X_single)[0][1])

        return {
            'churn_prediction': 'Yes' if pred_class == 1 else 'No',
            'churn_probability': pred_proba
        }

if __name__ == '__main__':
    # Test training and predicting
    handler = ChurnModelHandler()
    if not handler.is_model_trained():
        print("Training model...")
        metrics = handler.train()
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    else:
        print("Model already trained.")
        
    sample = {
        'gender': 'Female',
        'SeniorCitizen': 'No',
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }
    
    res = handler.predict(sample)
    print("Sample prediction:", res)
