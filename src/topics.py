import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import pickle
import os

def build_topic_model(df: pd.DataFrame, text_col="clean_text", n_topics=5):
    """
    Fit BERTopic on all articles.
    Aligned to paper: 5 topic clusters -
    financial performance, product/innovation, leadership decisions,
    regulatory/legal, market strategy.
    """
    print("Building sentence embeddings for BERTopic...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_topics,
        verbose=True
    )

    docs = df[text_col].tolist()
    topics, probs = topic_model.fit_transform(docs)

    os.makedirs("models", exist_ok=True)
    with open("models/bertopic_model.pkl", "wb") as f:
        pickle.dump(topic_model, f)

    topic_info = topic_model.get_topic_info()
    print("\nTop topics found:")
    print(topic_info[["Topic", "Count", "Name"]].to_string())

    return topic_model, topics, probs

def extract_topic_features(df: pd.DataFrame, topics, probs):
    """
    Attach topic assignments and build daily topic distributions.
    Aligned to paper: topic proportions per time window per executive.
    """
    df = df.copy()
    df["topic_id"] = topics

    if hasattr(probs, "shape") and len(probs.shape) == 2:
        df["topic_confidence"] = probs.max(axis=1)
    else:
        df["topic_confidence"] = probs

    df["date"] = pd.to_datetime(df["date"])

    topic_dummies = pd.get_dummies(df["topic_id"], prefix="topic")
    df_with_topics = pd.concat([df, topic_dummies], axis=1)

    topic_cols = [c for c in df_with_topics.columns if c.startswith("topic_")]
    daily_topics = (
        df_with_topics.groupby(["ticker", "date"])[topic_cols]
        .mean()
        .reset_index()
    )

    return df_with_topics, daily_topics

if __name__ == "__main__":
    df = pd.read_csv("data/processed/articles_sentiment.csv")
    model, topics, probs = build_topic_model(df)
    df_topics, daily_topics = extract_topic_features(df, topics, probs)

    df_topics.to_csv("data/processed/articles_topics.csv", index=False)
    daily_topics.to_csv("data/processed/daily_topics.csv", index=False)
    print(f"\nTopic modeling done.")
    print(f"Articles with topics: {len(df_topics)} rows")
    print(f"Daily topics: {len(daily_topics)} rows")