import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class PhishingDetector:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.model = HistGradientBoostingClassifier(
        max_iter=300,            # Increase iterations (default: 100)
        learning_rate=0.1,      # Lower learning rate for stability
        max_depth=9,             # Deeper trees for better feature learning
        min_samples_leaf=10,     # Prevents overfitting (default: 20)
        max_bins=255,            # More bins for better split decisions
        l2_regularization=0.1,   # Reduces overfitting
        scoring='accuracy',      # Optimize for accuracy
        early_stopping=True,     # Stops training when no improvement
        random_state=42,
        )

        self.numeric_features = ['url_length', 'dots_count', 'digits_count', 'special_chars_count', 
                                 'path_depth', 'avg_token_length']
        self.categorical_features = ['has_http', 'has_www']  

    
    def extract_url_features(self, url):
        url = url.lower()
        dots_count = url.count('.')
        path_depth = url.count('/')
        url_length = len(url)
        digits_count = sum(map(str.isdigit, url))
        special_chars_count = len(re.findall(r'[^a-z0-9.-_/]', url))
        
        tokens = re.split(r'[/.-_]', url)
        tokens = [token for token in tokens if token]  # Remove empty tokens
        avg_token_length = np.mean([len(token) for token in tokens]) if tokens else 0

        
        has_http = url.startswith('http://')
        has_www = 'www.' in url
        
        return {
            'url_length': url_length,
            'dots_count': dots_count,
            'digits_count': digits_count,
            'special_chars_count': special_chars_count,
            'path_depth': path_depth,
            'avg_token_length': avg_token_length,
            'has_http': int(has_http),
            'has_www': int(has_www)
        }

    def load_and_preprocess_data(self, malicious_path, phishing_path, new_data_urls):
    # Load datasets
        df1 = pd.read_csv(malicious_path, usecols=['url', 'type'])
        df2 = pd.read_csv(phishing_path, usecols=['URL', 'Label'])
        df3 = pd.read_csv(new_data_urls, usecols=['url', 'status'])

    # Rename columns for consistency
        df1.rename(columns={'url': 'URL', 'type': 'Label'}, inplace=True)

    # Sort df1: 'benign' → 'good', others → 'bad'
        df1['Label'] = df1['Label'].apply(lambda x: 'good' if x == 'benign' else 'bad')

    # Use all data from df3: map status 0 → 'bad', 1 → 'good'
        df3.rename(columns={'url': 'URL', 'status': 'Label'}, inplace=True)
        df3['Label'] = df3['Label'].map({0: 'bad', 1: 'good'})

    # Merge datasets
        df_combined = pd.concat([df1, df2, df3], ignore_index=True)

    # Convert 'Label' column to numerical values
        df_combined['Label'] = df_combined['Label'].map({'good': 0, 'bad': 1}).astype(np.uint8)

    # Extract features from URLs
        df_features = pd.DataFrame([self.extract_url_features(url) for url in df_combined['URL']])
    
        return pd.concat([df_combined, df_features], axis=1)



    def prepare_features(self, df, training=True):
        if training:
            self.encoder.fit(df[self.categorical_features])
            self.scaler.fit(df[self.numeric_features])

        X_categorical = self.encoder.transform(df[self.categorical_features])
        X_numeric = self.scaler.transform(df[self.numeric_features])

        X_categorical_df = pd.DataFrame(X_categorical, columns=self.encoder.get_feature_names_out(), index=df.index)
        X_numeric_df = pd.DataFrame(X_numeric, columns=self.numeric_features, index=df.index)

        return pd.concat([X_numeric_df, X_categorical_df], axis=1)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, model_name="phishing_detector_v1", directory="models"):
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{model_name}.joblib")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoder': self.encoder
        }, model_path)
        return model_path

    def load_model(self, model_path):
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.encoder = data['encoder']

# === MAIN CODE TO RUN EVERYTHING ===

# Initialize
detector = PhishingDetector()

# Load and preprocess dataset
df = detector.load_and_preprocess_data('malicious_phish.csv', 'phishing_site_urls.csv','new_data_urls.csv')

# Prepare features
X = detector.prepare_features(df)
y = df['Label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
detector.train(X_train, y_train)

# Predict
y_pred = detector.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# Save model
model_path = detector.save_model()
print(f"Model saved at: {model_path}")

# Load model and predict on test data
detector.load_model(model_path)
y_pred_loaded = detector.predict(X_test)

# Verify loaded model
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
print(f"Loaded Model Accuracy: {loaded_accuracy:.2%}")

from sklearn.metrics import classification_report

# Get classification report
print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, target_names=["Good (0)", "Bad (1)"]))

