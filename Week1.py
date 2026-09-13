import numpy as np
import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    dataset_path = input("Enter the CSV file path: ").strip()
    if not dataset_path:
        dataset_path = r"C:\Users\mrudu\Microsoft VS Code\AI-Pioneers-Internship\sample_dataset.csv"

    df = load_dataset(dataset_path)
    print_section("1. Initial dataset from CSV")
    print(f"Loading data from: {dataset_path}")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nMissing values before cleaning:\n", df.isnull().sum())

    print_section("2. Missing value treatment")
    numeric_cols = ["Age", "Salary", "Experience"]
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in ["Gender", "City"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(df.isnull().sum())

    print_section("3. Duplicate removal")
    print("Duplicates before removal:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Duplicates after removal:", df.duplicated().sum())

    print_section("4. Feature selection")
    selected_features = ["Age", "Salary", "Experience", "Gender", "Department", "Target"]
    df = df[selected_features]
    print(df.head())

    print_section("5. Encoding categorical variables")
    encoded_df = pd.get_dummies(df, columns=["Gender", "Department"], drop_first=True)
    print(encoded_df.head())

    print_section("6. Normalization")
    X = encoded_df.drop(columns=["Target"])
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X_scaled_df = X.copy()

    for col in numeric_features:
        mean = X_scaled_df[col].mean()
        std = X_scaled_df[col].std(ddof=0)
        X_scaled_df[col] = (X_scaled_df[col] - mean) / std if std != 0 else 0

    print(X_scaled_df.head())
    print("\nScaled feature means approximately 0:")
    print(X_scaled_df[numeric_features].mean().round(4))

    print_section("7. Exploratory Data Analysis")
    print("\nSummary statistics:\n", encoded_df.describe(include="all").T)

    corr = encoded_df[["Age", "Salary", "Experience", "Target"]].corr()
    print("\nCorrelation matrix:\n", corr)

    department_columns = [col for col in encoded_df.columns if col.startswith("Department_")]
    if department_columns:
        department_salary = pd.Series({
            dept: encoded_df.loc[encoded_df[dept] == 1, "Salary"].mean()
            for dept in department_columns
        })
        print("\nAverage salary by department:\n", department_salary)
    else:
        print("\nNo department columns available after encoding.")

    output_path = "cleaned_dataset.csv"
    encoded_df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")

    print_section("8. Final preprocessing outcome")
    print("The dataset is now ready for ML model training:")
    print("- Missing values handled")
    print("- Duplicates removed")
    print("- Categorical variables encoded")
    print("- Numerical features normalized")
    print("- CSV-based dataset workflow used")


if __name__ == "__main__":
    main()
