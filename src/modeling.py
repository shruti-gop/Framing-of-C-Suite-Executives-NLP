import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import pickle
import os

def build_master_dataset():
    """
    Merge sentiment + topic + embedding features with stock labels.
    Aligned to paper: perception features only, no financial variables.
    """
    print("Loading feature files...")
    sentiment = pd.read_csv("data/processed/daily_sentiment.csv", parse_dates=["date"])
    topics = pd.read_csv("data/processed/daily_topics.csv", parse_dates=["date"])
    embeddings = pd.read_csv("data/processed/daily_embeddings.csv", parse_dates=["date"])

    # Merge all NLP features
    features = sentiment.merge(topics, on=["ticker", "date"], how="inner")
    features = features.merge(embeddings, on=["ticker", "date"], how="inner")

    # Load stock labels
    stocks = []
    stock_dir = "data/stocks"
    for fname in os.listdir(stock_dir):
        if fname.endswith(".csv"):
            ticker = fname.replace(".csv", "")
            s = pd.read_csv(f"{stock_dir}/{fname}")
            s.columns = [c[0] if isinstance(c, tuple) else c for c in s.columns]
            s.columns = [c.lower() for c in s.columns]
            s["date"] = pd.to_datetime(s["date"])
            s["ticker"] = ticker
            # Flatten MultiIndex columns if present
            s.columns = [c[0] if isinstance(c, tuple) else c for c in s.columns]
            stocks.append(s[["date", "ticker", "label", "future_return"]])

    stock_df = pd.concat(stocks, ignore_index=True)

    # Merge NLP features with stock labels
    master = features.merge(stock_df, on=["ticker", "date"], how="inner")
    master = master.dropna(subset=["label"])
    master = master.sort_values(["ticker", "date"]).reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)
    master.to_csv("data/processed/master_dataset.csv", index=False)
    print(f"Master dataset: {len(master)} rows, {master.shape[1]} columns")
    return master

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get all NLP feature columns, excluding metadata and label.
    Aligned to paper: sentiment scores, topic probabilities,
    semantic embeddings, media attention volume.
    """
    exclude = {
        "ticker", "date", "label", "future_return",
        "executive", "company", "sentiment_label"
    }
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [np.float64, np.int64, float, int]]

def train_and_evaluate(df: pd.DataFrame):
    """
    Train three models using time-series cross-validation.
    Aligned to paper: logistic regression baseline,
    gradient boosting, and XGBoost ensemble.
    TimeSeriesSplit prevents lookahead bias.
    """
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} features")

    X = df[feature_cols].fillna(0).values
    y = df["label"].values

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4,
            eval_metric="logloss", random_state=42,
            verbosity=0
        ),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    all_results = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")
        fold_accs, fold_aucs = [], []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            fold_accs.append(acc)
            fold_aucs.append(auc)
            print(f"  Fold {fold+1}: Acc={acc:.3f}, AUC={auc:.3f}")

        mean_acc = np.mean(fold_accs)
        mean_auc = np.mean(fold_aucs)
        print(f"  → Mean Acc: {mean_acc:.3f}, Mean AUC: {mean_auc:.3f}")
        all_results[model_name] = {
            "accuracy": mean_acc,
            "auc": mean_auc
        }

        os.makedirs("models", exist_ok=True)
        with open(f"models/{model_name.lower().replace(' ', '_')}.pkl", "wb") as f:
            pickle.dump(model, f)

    return all_results

if __name__ == "__main__":
    df = build_master_dataset()
    results = train_and_evaluate(df)

    print("\n=== Final Results ===")
    for model, metrics in results.items():
        print(f"{model}: Accuracy={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")