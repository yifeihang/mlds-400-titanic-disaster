import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paths
TRAIN_CSV = 'src/data/train.csv'
TEST_CSV = 'src/data/test.csv'
OUT_PRED_CSV = 'src/data/results/test_predictions_py.csv'

# --- CONFIG / what we will do (kept small so CHANGES can reference them) ---
DROP_COLS = ['PassengerId', 'Name', 'Ticket', 'Cabin']
DUMMY_COLS = ['Sex', 'Embarked', 'Pclass']
STANDARDIZE_COLS = ['Age', 'SibSp', 'Parch', 'Fare']

def load_train(path):
    return pd.read_csv(path)

def preprocess_train(df):
    # Drop columns
    df = df.drop(columns=DROP_COLS, errors='ignore')
    # Dummies
    df = pd.get_dummies(df, prefix_sep='_', columns=DUMMY_COLS)
    # Impute train medians
    age_median = df['Age'].median()
    fare_median = df['Fare'].median()
    df['Age'] = df['Age'].fillna(age_median)
    df['Fare'] = df['Fare'].fillna(fare_median)
    # Fit scaler on TRAIN numeric columns
    scaler = StandardScaler()
    df_scaled_nums = pd.DataFrame(scaler.fit_transform(df[STANDARDIZE_COLS]),
                                  columns=STANDARDIZE_COLS,
                                  index=df.index)
    # Replace numeric cols with scaled versions
    df_final = pd.concat([df_scaled_nums, df.drop(columns=STANDARDIZE_COLS)], axis=1)
    return df_final, scaler, age_median, fare_median

def preprocess_test(df_test, train_columns, scaler, age_median, fare_median):
    # Keep PassengerId for output if present
    passenger_ids = None
    if 'PassengerId' in df_test.columns:
        passenger_ids = df_test['PassengerId'].copy()
    # Drop same cols except keep PassengerId for output (we already saved it)
    df_test = df_test.drop(columns=DROP_COLS, errors='ignore')
    # Dummies
    df_test = pd.get_dummies(df_test, prefix_sep='_', columns=DUMMY_COLS)
    # Impute using TRAIN medians
    df_test['Age'] = df_test['Age'].fillna(age_median)
    df_test['Fare'] = df_test['Fare'].fillna(fare_median)
    # Align columns to train features (train_columns includes 'Survived' for train)
    train_feature_cols = [c for c in train_columns if c != 'Survived']
    df_test_aligned = df_test.reindex(columns=train_feature_cols, fill_value=0)
    # Standardize numeric cols using TRAIN-fitted scaler (transform only)
    df_test_scaled_nums = pd.DataFrame(scaler.transform(df_test_aligned[STANDARDIZE_COLS]),
                                       columns=STANDARDIZE_COLS,
                                       index=df_test_aligned.index)
    df_test_final = pd.concat([df_test_scaled_nums, df_test_aligned.drop(columns=STANDARDIZE_COLS)], axis=1)
    return df_test_final, passenger_ids

def main():
    # Load & preprocess train
    train_raw = load_train(TRAIN_CSV)
    train_proc, scaler, age_med, fare_med = preprocess_train(train_raw)

    # Prepare X/y
    X_train = train_proc.drop(columns='Survived')
    y_train = train_proc['Survived']

    # Fit model
    model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)

    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Load & preprocess test
    test_raw = pd.read_csv(TEST_CSV)
    X_test, passenger_ids = preprocess_test(test_raw, train_proc.columns, scaler, age_med, fare_med)

    # Predict on X_test
    preds = model.predict(X_test)

    # Build output DataFrame (include PassengerId if available)
    if passenger_ids is not None:
        out_df = pd.DataFrame({'PassengerId': passenger_ids, 'Predicted': preds})
    else:
        # If no PassengerId, give a simple index
        out_df = pd.DataFrame({'Index': X_test.index, 'Predicted': preds})

    # Save predictions
    out_df.to_csv(OUT_PRED_CSV, index=False)

    # --- Minimal printing as requested ---
    print("CHANGES")
    print(f"- Dropped columns: {DROP_COLS}")
    print(f"- One-hot encoded columns: {DUMMY_COLS}")
    print(f"- Imputed Age and Fare using TRAIN medians (no test leakage)")
    print(f"- Standardized columns (scaler fit on TRAIN): {STANDARDIZE_COLS}")
    print(f"- Aligned test features to train features before predicting")
    print()
    print("RESULTS")
    print(f"- Training accuracy: {train_acc:.4f}")
    print(f"- Predictions saved to: {OUT_PRED_CSV}")
    print(f"- Total predictions: {len(preds)} (Survived=1 count: {int((preds==1).sum())})")

if __name__ == '__main__':
    main()
