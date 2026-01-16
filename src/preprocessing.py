"""
Data preprocessing module for IDS ML project.
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    """Preprocess network traffic data for ML models."""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Load data from CSV file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or PKL.")
            
            print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean the dataset by handling missing values and outliers."""
        print("Cleaning data...")
        
        df = df.copy()
        
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        print(f"Removed {initial_shape - df.shape[0]} duplicate rows")
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Handling missing values...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if len(df[col].mode()) > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna('unknown')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def extract_features(self, df):
        """Extract and engineer features from network traffic data."""
        print("Extracting features...")
        
        df_encoded = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Liste des colonnes cibles potentielles
        target_cols = [
            'label', 'Label', 'LABEL',
            'attack', 'Attack', 'ATTACK',
            'class', 'Class', 'CLASS',
            'target', 'Target', 'TARGET',
            'category', 'Category', 'CATEGORY',
            'type', 'Type', 'TYPE',
            'attack_cat', 'Attack_Cat', 'ATTACK_CAT'
        ]
        
        for col in target_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
            if col in categorical_cols:
                categorical_cols.remove(col)
        
        for col in categorical_cols:
            if col not in target_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                numeric_cols.append(col)
        
        self.feature_columns = numeric_cols
        print(f"Feature columns: {len(self.feature_columns)}")
        
        return df_encoded
    
    def find_target_column(self, df):
        """
        Trouve automatiquement la colonne cible dans le DataFrame.
        PRIORITÉ: Colonnes avec catégories d'attaque détaillées (attack_cat).
        """
        # PRIORITÉ 1: attack_cat avec catégories détaillées
        priority_category_cols = ['attack_cat', 'Attack_Cat', 'ATTACK_CAT']
        for name in priority_category_cols:
            if name in df.columns:
                n_unique = df[name].nunique()
                # Si plus de 2 catégories, c'est la bonne colonne
                if n_unique > 2:
                    print(f"✓ Utilisation de '{name}' comme cible ({n_unique} catégories)")
                    return name
        
        # PRIORITÉ 2: Colonnes label avec catégories (pas binaire)
        label_cols = ['Label', 'label', 'LABEL', 'category', 'Category']
        for name in label_cols:
            if name in df.columns:
                n_unique = df[name].nunique()
                if n_unique > 2:
                    print(f"✓ Utilisation de '{name}' comme cible ({n_unique} catégories)")
                    return name
        
        # PRIORITÉ 3: Première colonne cible trouvée
        possible_names = [
            'Label', 'label', 'LABEL',
            'Attack', 'attack', 'ATTACK',
            'Class', 'class', 'CLASS',
            'Target', 'target', 'TARGET',
            'attack_cat', 'Attack_Cat', 'ATTACK_CAT'
        ]
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        # Chercher des colonnes contenant ces mots-clés
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(keyword in col_lower for keyword in ['label', 'attack', 'class', 'target', 'category']):
                return col
        
        # Dernière colonne catégorielle
        last_col = df.columns[-1]
        if df[last_col].dtype == 'object' and df[last_col].nunique() < 50:
            print(f"⚠️ Utilisation de la dernière colonne comme cible: {last_col}")
            return last_col
        
        return None
    
    def prepare_data(self, df, target_column=None, test_size=0.2, random_state=42):
        """Prepare data for training by splitting and scaling."""
        print("Preparing data for training...")
        
        if target_column is None:
            target_col = self.find_target_column(df)
        else:
            target_col = target_column
        
        if target_col is None:
            print("\n❌ Colonnes disponibles dans le dataset:")
            for i, col in enumerate(df.columns, 1):
                print(f"  {i:3d}. {col}")
            raise ValueError("Target column not found.")
        
        print(f"✓ Colonne cible: {target_col}")
        
        X = df[self.feature_columns].copy()
        y = df[target_col].copy()
        
        print(f"\nDistribution des classes:")
        class_dist = y.value_counts()
        for label, count in class_dist.items():
            print(f"  {label}: {count} ({count/len(y)*100:.2f}%)")
        
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            print(f"\n✓ Classes encodées: {len(self.label_encoder.classes_)} classes")
            for idx, cls in enumerate(self.label_encoder.classes_):
                print(f"  {idx}: {cls}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=self.feature_columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, file_path='models/preprocessor.pkl'):
        """Save the preprocessor for later use."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }, file_path)
        print(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path='models/preprocessor.pkl'):
        """Load a saved preprocessor."""
        preprocessor = joblib.load(file_path)
        self.scaler = preprocessor['scaler']
        self.label_encoder = preprocessor['label_encoder']
        self.feature_columns = preprocessor['feature_columns']
        print(f"Preprocessor loaded from {file_path}")

    def ensure_feature_compatibility(self, df):
        """Assure la compatibilité des features."""
        if self.feature_columns is None:
            return df
        
        df_compatible = df.copy()
        
        for feature in self.feature_columns:
            if feature not in df_compatible.columns:
                print(f"⚠️ Ajout de la feature manquante: {feature}")
                df_compatible[feature] = 0
        
        extra_features = [col for col in df_compatible.columns 
                        if col not in self.feature_columns 
                        and col not in ['label', 'Label', 'attack_cat', 'Attack_Cat']]
        if extra_features:
            print(f"⚠️ Suppression des features non utilisées: {extra_features}")
            df_compatible = df_compatible.drop(columns=extra_features, errors='ignore')
        
        return df_compatible[self.feature_columns]


def main():
    """Example usage of the preprocessor."""
    print("Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'duration': np.random.exponential(10, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh'], n_samples),
        'src_bytes': np.random.poisson(1000, n_samples),
        'dst_bytes': np.random.poisson(2000, n_samples),
        'count': np.random.poisson(5, n_samples),
        'srv_count': np.random.poisson(3, n_samples),
        'dst_host_count': np.random.poisson(10, n_samples),
        'attack_cat': np.random.choice(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'], n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/sample_data.csv', index=False)
    print("Sample data saved to data/raw/sample_data.csv")
    
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.clean_data(df)
    df_features = preprocessor.extract_features(df_cleaned)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_features)
    
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)
    
    preprocessor.save_preprocessor()
    
    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()