import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import os

def generate_embeddings(df: pd.DataFrame, text_col="clean_text", model_name="all-MiniLM-L6-v2"):
    """
    Generate 384-dim sentence embeddings for each article.
    Aligned to paper: transformer-based sentence-level embeddings.
    """
    print(f"Generating embeddings with {model_name}...")
    model = SentenceTransformer(model_name)

    texts = df[text_col].tolist()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings

def reduce_embeddings(embeddings: np.ndarray, n_components=20):
    """
    Apply PCA to reduce from 384 dims to 20 dims.
    Aligned to paper: first 20 components capture ~76% of variance.
    """
    print(f"Running PCA: {embeddings.shape[1]}d → {n_components}d")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    explained = pca.explained_variance_ratio_.cumsum()
    print(f"  Variance explained by {n_components} components: {explained[-1]:.1%}")
    return reduced, pca

def aggregate_daily_embeddings(df: pd.DataFrame, embeddings_reduced: np.ndarray) -> pd.DataFrame:
    """
    Average PCA-reduced embeddings per ticker per day.
    Aligned to paper: embeddings averaged across time windows.
    """
    embed_df = pd.DataFrame(
        embeddings_reduced,
        columns=[f"embed_{i}" for i in range(embeddings_reduced.shape[1])]
    )
    df_embed = pd.concat([
        df[["ticker", "date"]].reset_index(drop=True),
        embed_df
    ], axis=1)

    df_embed["date"] = pd.to_datetime(df_embed["date"])
    daily_embed = df_embed.groupby(["ticker", "date"]).mean().reset_index()
    return daily_embed

if __name__ == "__main__":
    df = pd.read_csv("data/processed/articles_topics.csv")

    embeddings = generate_embeddings(df)

    os.makedirs("models", exist_ok=True)
    np.save("models/raw_embeddings.npy", embeddings)

    # Trial 1 — 20 components (aligned to paper)
    reduced_20, pca_20 = reduce_embeddings(embeddings, n_components=20)
    daily_embed_20 = aggregate_daily_embeddings(df, reduced_20)
    daily_embed_20.to_csv("data/processed/daily_embeddings_20.csv", index=False)
    print(f"20 components: {len(daily_embed_20)} rows saved.")

    # Trial 2 — 40 components (extended trial)
    reduced_40, pca_40 = reduce_embeddings(embeddings, n_components=40)
    daily_embed_40 = aggregate_daily_embeddings(df, reduced_40)
    daily_embed_40.to_csv("data/processed/daily_embeddings_40.csv", index=False)
    print(f"40 components: {len(daily_embed_40)} rows saved.")

    # Keep default as 20 for main pipeline
    daily_embed_20.to_csv("data/processed/daily_embeddings.csv", index=False)
    print(f"Default embeddings set to 20 components.")