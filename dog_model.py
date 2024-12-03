import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import pickle
from sklearn.model_selection import cross_val_score

print("Current working directory:", os.getcwd())

# Ensure model directory exists
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# 1. Function to extract features from audio files
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)  # Change to 10 MFCCs
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    return np.concatenate((
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(spectral_contrast.T, axis=0)
    ))

# 2. Data Augmentation function for increasing dataset size
def augment_audio(y, sr):
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    y_stretched = librosa.effects.time_stretch(y, rate=0.8)
    noise = np.random.randn(len(y))
    y_noisy = y + 0.005 * noise

    return [y_shifted, y_stretched, y_noisy]

# 3. Load dataset and apply feature extraction and augmentation
def load_dataset(dataset_path):
                features = []
                labels = []

                for category in os.listdir(dataset_path):
                    category_path = os.path.join(dataset_path, category)

                    for file_name in os.listdir(category_path):
                        file_path = os.path.join(category_path, file_name)
                        y, sr = librosa.load(file_path, sr=22050)
                        feature = extract_features(y, sr)
                        features.append(feature)
                        labels.append(category)

                        for augmented in augment_audio(y, sr):
                            augmented_feature = extract_features(augmented, sr)
                            features.append(augmented_feature)
                            labels.append(category)

                return np.array(features), np.array(labels)


# 4. Load the dataset
dataset_path = 'processed_dataset'
features, labels = load_dataset(dataset_path)

# 5. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 6. Check class distribution
unique, counts = np.unique(labels, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class distribution:", class_distribution)

# 7. Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

clf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid,
                                   n_iter=10, cv=5, verbose=2, n_jobs=2, random_state=42)
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print("Best parameters found: ", best_params)

# 8. Train the model with the best parameters
best_clf = random_search.best_estimator_
best_clf.fit(X_train, y_train)

# 9. Perform feature selection using Recursive Feature Elimination (RFE)
rfe = RFE(estimator=best_clf, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)

# Transform the data based on the selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

print(f"Number of features after RFE: {X_train_rfe.shape[1]}")

# Re-train model with selected features
best_clf.fit(X_train_rfe, y_train)

# 10. Evaluate the model
y_pred = best_clf.predict(X_test_rfe)
print(classification_report(y_test, y_pred))

# 11. Cross-validate the model for better performance estimation
cv_scores = cross_val_score(best_clf, X_train_rfe, y_train, cv=5)
print(f"Cross-validated accuracy: {np.mean(cv_scores):.4f}")

# 12. Save the trained model and RFE selector in the 'model' folder
model_path = os.path.join(model_dir, 'dog_bark_classifier2.pkl')
rfe_path = os.path.join(model_dir, 'rfe_selector2.pkl')

with open(model_path, 'wb') as model_file:
    pickle.dump(best_clf, model_file)

with open(rfe_path, 'wb') as rfe_file:
    pickle.dump(rfe, rfe_file)

# 13. Load the trained model and RFE selector for future use
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(rfe_path, 'rb') as rfe_file:
    rfe = pickle.load(rfe_file)

# Function to handle file prediction
def handle_file_prediction(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    features = extract_features(y, sr)

    features_rfe = rfe.transform(features.reshape(1, -1))  # Reshape for single sample
    prediction = loaded_model.predict(features_rfe)
    return prediction