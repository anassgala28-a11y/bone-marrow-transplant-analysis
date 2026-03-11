import numpy as np
import pandas as pd

def optimize_memory(df): 
    """
    Optimise l'usage mémoire du DataFrame en convertissant les types de données.
    Exigence du projet Coding Week 2026.
    """
    # Calcul de la mémoire initiale
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # On ne traite que les colonnes numériques
        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Cas des Entiers
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # Cas des Flottants (Ex: float64 vers float32 comme demandé)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        
        # Optionnel : Convertir les colonnes objets avec peu de valeurs uniques en 'category'
        elif col_type == object:
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    return df, start_mem, end_mem