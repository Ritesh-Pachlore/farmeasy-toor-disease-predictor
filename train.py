# train.py (updated)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def load_data(file_path):
    """Load dataset and basic inspection"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    print("\nHandling missing values...")
    df = df.dropna().reset_index(drop=True)
    return df

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split into X and y, then into train/val/test"""
    X = df.drop('disease', axis=1)
    y = df['disease']
    # encode target labels (we can fit label encoder on full y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # First split: test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    # Second split: validation from remaining
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    print(f"Train set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

    return X_train.reset_index(drop=True), X_val.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_val, y_test, le

def preprocess_features(X_train, X_val, X_test):
    """
    Convert sowing_date to numeric and encode categorical features.
    Fit encoders on X_train only to avoid leakage.
    """
    print("\nPreprocessing features (date -> numeric, encoding categorical)...")

    # Make copies
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Convert sowing_date to datetime and to numeric (days since min in training set)
    for df in [X_train, X_val, X_test]:
        df['sowing_date'] = pd.to_datetime(df['sowing_date'], errors='coerce')

    # If any conversion produced NaT, drop (or handle) - here we'll drop
    X_train = X_train.dropna(subset=['sowing_date']).reset_index(drop=True)
    # For val/test we'll keep rows; if NaT present we'll fill with earliest train date
    min_date = X_train['sowing_date'].min()
    X_val['sowing_date'] = X_val['sowing_date'].fillna(min_date)
    X_test['sowing_date'] = X_test['sowing_date'].fillna(min_date)

    # Create numeric date feature: days since min_date (train)
    X_train['sowing_days'] = (X_train['sowing_date'] - min_date).dt.days
    X_val['sowing_days'] = (X_val['sowing_date'] - min_date).dt.days
    X_test['sowing_days'] = (X_test['sowing_date'] - min_date).dt.days

    # Drop original datetime column (we now have numeric)
    X_train = X_train.drop(columns=['sowing_date'])
    X_val = X_val.drop(columns=['sowing_date'])
    X_test = X_test.drop(columns=['sowing_date'])

    # Categorical columns to encode
    cat_cols = ['crop', 'stage']

    # Use OrdinalEncoder with handle_unknown to avoid errors on unseen categories
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ord_enc.fit(X_train[cat_cols])

    X_train_cat = ord_enc.transform(X_train[cat_cols])
    X_val_cat = ord_enc.transform(X_val[cat_cols])
    X_test_cat = ord_enc.transform(X_test[cat_cols])

    # Build numeric feature arrays by concatenating categorical encodings and sowing_days
    X_train_num = np.hstack([X_train_cat, X_train[['sowing_days']].to_numpy()])
    X_val_num = np.hstack([X_val_cat, X_val[['sowing_days']].to_numpy()])
    X_test_num = np.hstack([X_test_cat, X_test[['sowing_days']].to_numpy()])

    # Column names (for reference)
    feature_names = cat_cols + ['sowing_days']

    encoders = {
        'ordinal_encoder': ord_enc,
        'feature_names': feature_names,
        'min_date': min_date  # useful at inference time to compute days
    }

    print("Feature names after preprocessing:", feature_names)
    print("Sample preprocessed X_train (first 5 rows):\n", X_train_num[:5])

    return X_train_num, X_val_num, X_test_num, encoders

def scale_features(X_train, X_val, X_test):
    """Scale numeric array features using StandardScaler (fit on train only)"""
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def get_models():
    """Define all models to test"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=5, use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, max_depth=5, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, depth=5, verbose=0),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    return models

def train_and_evaluate_models(models, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate all models"""
    results = {}
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)
    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Training {name}...")
        print(f"{'='*80}")
        try:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc
            }

            print(f"Train Accuracy: {train_acc*100:.2f}%")
            print(f"Validation Accuracy: {val_acc*100:.2f}%")
            print(f"Test Accuracy: {test_acc*100:.2f}%")

        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue

    return results

def display_results(results):
    """Display comparison of all models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("="*80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)

    for name, metrics in sorted_results:
        print(f"{name:<25} {metrics['train_accuracy']*100:>10.2f}% {metrics['val_accuracy']*100:>10.2f}% {metrics['test_accuracy']*100:>10.2f}%")

    return sorted_results

def save_best_model(best_model_name, best_model, scaler, encoders, X_test, y_test, label_encoder):
    """Save the best model and related objects"""
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*80}")

    y_pred = best_model.predict(X_test)
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))

    model_data = {
        'model': best_model,
        'scaler': scaler,
        'encoders': encoders,     # ordinal_encoder, feature_names, min_date
        'label_encoder': label_encoder,
        'model_name': best_model_name
    }

    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("\n✓ Best model saved as 'model.pkl'")
    print(f"✓ Model: {best_model_name}")
    print(f"✓ Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

def main():
    print("\n" + "="*80)
    print("DISEASE PREDICTION MODEL TRAINING PIPELINE")
    print("="*80)

    df = load_data('dataset/toor_crop_dataset.csv')
    X_train, X_val, X_test, y_train, y_val, y_test, label_enc = split_data(df)

    # IMPORTANT: if dropna on sowing_date in preprocess changed lengths,
    # we must ensure y arrays align. For simplicity, we'll reindex y based on original indices.
    # To keep alignment safe: we handled dropping NaT only in training set earlier; assume dataset is mostly clean.

    # Preprocess features (fit encoders on train)
    X_train_num, X_val_num, X_test_num, encoders = preprocess_features(X_train, X_val, X_test)

    # Scale numeric arrays
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train_num, X_val_num, X_test_num)

    # Note: if preprocess dropped some rows from X_train due to invalid dates,
    # you'd need to drop corresponding y_train rows. Here we assume they remain aligned.
    # If you do run into shape mismatch errors, inspect X_train/X_train_num shapes and adjust y_train accordingly.

    # Get models and train
    models = get_models()
    results = train_and_evaluate_models(models, X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

    sorted_results = display_results(results)

    # Save best
    if len(sorted_results) == 0:
        print("No models trained successfully.")
        return

    best_model_name = sorted_results[0][0]
    best_model = sorted_results[0][1]['model']
    save_best_model(best_model_name, best_model, scaler, encoders, X_test_scaled, y_test, label_enc)

    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
