import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(input_filepath, output_filepath=None, save_output=True):
    
    # 1. Load data
    print("="*60)
    print("STEP 1: Loading data...")
    print(f"Input file: {input_filepath}")
    df = pd.read_csv(input_filepath)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Cek missing values
    print("\n" + "="*60)
    print("STEP 2: Checking missing values...")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")
    
    # 3. Menghapus duplikat
    print("\n" + "="*60)
    print("STEP 3: Removing duplicates...")
    duplicates_before = df.duplicated().sum()
    print(f"Duplicates found: {duplicates_before}")
    df.drop_duplicates(inplace=True)
    print(f"Duplicates after removal: {df.duplicated().sum()}")
    print(f"Data shape after removing duplicates: {df.shape}")
    
    # 4. Standardisasi data numerik
    print("\n" + "="*60)
    print("STEP 4: Standardizing numerical features...")
    standard_scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    print(f"Numerical columns: {numerical_cols}")
    
    df_scaled = df.copy()
    df_scaled[numerical_cols] = standard_scaler.fit_transform(df[numerical_cols])
    print("Standardization completed.")
    
    # 5. Menghapus kolom yang tidak relevan
    print("\n" + "="*60)
    print("STEP 5: Dropping irrelevant columns...")
    columns_to_drop = ['name']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_scaled.columns]
    if existing_columns_to_drop:
        df_scaled = df_scaled.drop(columns=existing_columns_to_drop)
        print(f"Dropped columns: {existing_columns_to_drop}")
    else:
        print("No columns to drop.")
    
    # 6. Handling outliers menggunakan IQR
    print("\n" + "="*60)
    print("STEP 6: Handling outliers using IQR method...")
    num_cols = df_scaled.select_dtypes(include=["float64", "int64"]).columns.tolist()
    print(f"Processing outliers for columns: {num_cols}")
    
    rows_before = df_scaled.shape[0]
    df_scaled = handle_outliers_iqr(df_scaled, num_cols, drop=True)
    rows_after = df_scaled.shape[0]
    print(f"Total rows removed: {rows_before - rows_after}")
    print(f"Data shape after outlier removal: {df_scaled.shape}")
    
    # 7. Encoding kolom kategorikal
    print("\n" + "="*60)
    print("STEP 7: Encoding categorical features...")
    cat_cols = df_scaled.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {cat_cols}")
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_scaled[col] = le.fit_transform(df_scaled[col])
        label_encoders[col] = le
        print(f"  - {col}: {len(le.classes_)} unique values encoded")
    print("Label encoding completed.")
    
    # 8. Membuat kategori harga (target)
    print("\n" + "="*60)
    print("STEP 8: Creating price categories...")
    bins = [
        df_scaled['selling_price'].quantile(0.0),
        df_scaled['selling_price'].quantile(0.33),
        df_scaled['selling_price'].quantile(0.66),
        df_scaled['selling_price'].quantile(1.0)
    ]
    labels = ['Low', 'Medium', 'High']
    
    df_scaled['price_category'] = pd.cut(
        df_scaled['selling_price'], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )
    
    print("Price category distribution:")
    print(df_scaled['price_category'].value_counts())
    
    # 9. Save hasil preprocessing
    if save_output:
        print("\n" + "="*60)
        print("STEP 9: Saving preprocessed data...")
        
        # Jika output_filepath tidak diberikan, buat otomatis
        if output_filepath is None:
            # Ambil nama file dari input
            input_filename = os.path.basename(input_filepath)
            input_name = os.path.splitext(input_filename)[0]
            
            # Dapatkan direktori saat ini (harusnya di folder preprocessing)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Nama file output di folder yang sama dengan script ini
            output_filepath = os.path.join(current_dir, f"{input_name}_preprocessed.csv")
        
        # Buat folder output jika belum ada
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        
        # Simpan ke CSV (dengan handling price_category yang bertipe category)
        df_to_save = df_scaled.copy()
        # Konversi category menjadi string agar bisa disimpan
        if 'price_category' in df_to_save.columns:
            df_to_save['price_category'] = df_to_save['price_category'].astype(str)
        
        df_to_save.to_csv(output_filepath, index=False)
        print(f"✓ Preprocessed data saved to: {output_filepath}")
        print(f"✓ File size: {os.path.getsize(output_filepath) / 1024:.2f} KB")
    
    # 10. Final result
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED!")
    print(f"Final data shape: {df_scaled.shape}")
    print(f"Final columns: {list(df_scaled.columns)}")
    print("="*60)
    
    return df_scaled, label_encoders, standard_scaler


def handle_outliers_iqr(data, columns, drop=True):

    df_out = data.copy()
    
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers_count = ((df_out[col] < lower) | (df_out[col] > upper)).sum()
        
        if drop:
            df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
            if outliers_count > 0:
                print(f"  - {col}: Removed {outliers_count} outliers")
        else:
            df_out[col] = df_out[col].clip(lower, upper)
            if outliers_count > 0:
                print(f"  - {col}: Clipped {outliers_count} outliers")
    
    return df_out


def prepare_train_data(df_processed, target_column='price_category', drop_original_target=True):

    print("\n" + "="*60)
    print("PREPARING DATA FOR TRAINING...")
    
    df_train = df_processed.copy()
    
    # Encode target jika masih kategorikal
    if df_train[target_column].dtype == 'object' or df_train[target_column].dtype.name == 'category':
        le_target = LabelEncoder()
        y = le_target.fit_transform(df_train[target_column])
        print(f"Target encoded. Classes: {le_target.classes_}")
    else:
        y = df_train[target_column].values
        le_target = None
    
    # Drop kolom target
    columns_to_drop = [target_column]
    if drop_original_target and 'selling_price' in df_train.columns:
        columns_to_drop.append('selling_price')
    
    X = df_train.drop(columns=columns_to_drop)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    print("="*60)
    
    return X, y, le_target


# Contoh penggunaan sesuai struktur folder
if __name__ == "__main__":

    # Path ke file dataset raw
    input_filepath = "C:\\Users\\Bima Indra\\Documents\\Eksperimen_Bima Indra Sakti\\CAR DETAILS FROM CAR DEKHO_raw.csv"
    
    # Jalankan preprocessing
    # File hasil akan otomatis disimpan di folder preprocessing/
    df_processed, label_encoders, scaler = preprocess_data(
        input_filepath=input_filepath,
        save_output=True
    )
    
    # Prepare data untuk training
    X, y, le_target = prepare_train_data(df_processed, target_column='price_category')
    
    # Tampilkan hasil
    print("\n" + "="*60)
    print("SAMPLE OF PROCESSED DATA:")
    print("="*60)
    print(df_processed.head(10))
    
    print("\n" + "="*60)
    print("SAMPLE OF TRAINING DATA:")
    print("="*60)
    print("X (Features):")
    print(X.head())
    print("\ny (Target):")
    print(y[:10])
    
    print("\n" + "="*60)
    print("INFO:")
    print("="*60)
    print(f"Preprocessing completed successfully!")
    print(f"Preprocessed data saved automatically")
    print(f"Data ready for model training")