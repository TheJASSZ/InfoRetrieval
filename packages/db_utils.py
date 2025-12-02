import os
import sqlite3
import sqlite_vec
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from settings import logger, DB_PATH, INSERT_QUERY, FETCH_QUERY, BASE_DIR


SQLITE_VEC_FOLDER = os.path.dirname(os.path.abspath(__file__))


def load_sqlite_vec_extension(db: sqlite3.Connection):
    """
    Loads the sqlite-vec 'vec0' extension (Windows/Mac/Linux).
    """

    suffixes = ["dll", "dylib", "so"]  # Windows / macOS / Linux

    for suf in suffixes:
        fname = f"vec0.{suf}"
        full = os.path.join(SQLITE_VEC_FOLDER, fname)
        if os.path.exists(full):
            db.enable_load_extension(True)
            db.load_extension(full)
            logger.info(f"Loaded sqlite-vec extension: {full}")
            return

    raise FileNotFoundError(
        "vec0 extension not found. Put vec0.dll / vec0.dylib / vec0.so in project folder."
    )


model = SentenceTransformer("BAAI/bge-base-en-v1.5")  # embedding model

def embed(text: str):
    """Return normalized float32 numpy vector."""
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32)


def create_database(db_path=DB_PATH):
    """
    Creates a SQLite database with a vec0 VIRTUAL TABLE for embeddings.
    """

    db_exists = os.path.exists(db_path)

    db = sqlite3.connect(db_path)

    # Load vec0 first
    load_sqlite_vec_extension(db)

    embed_dim = model.get_sentence_embedding_dimension()

    if not db_exists:
        # Create vec0 table
        db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS summaries USING vec0(
            summary TEXT,
            metadata TEXT,
            embedding FLOAT[{embed_dim}]
        );
        """)
        db.commit()
        logger.info(f"Database created with vec0 table (dim={embed_dim})")

    logger.info(f"Database exists with vec0 table (dim={embed_dim})")

    return db

db = create_database()

def add_entry(db: sqlite3.Connection, summary: str, metadata: dict):
    """
    Adds a new entry with:
        - summary (string)
        - metadata (dict -> JSON)
        - embedding vector (JSON string)
    """
    vec = embed(summary)
    vec_json = json.dumps(vec.tolist())

    db.execute(INSERT_QUERY,(summary, json.dumps(metadata), vec_json))
    db.commit()

    logger.info(f"Inserted: {summary[:70]}...")


def _search_internal(db: sqlite3.Connection, query: str, k: int = 5):
    """Internal: returns raw rows as tuples."""
    qvec = embed(query)
    q_json = json.dumps(qvec.tolist())

    rows = db.execute(FETCH_QUERY, (q_json, k)
    ).fetchall()

    # Convert metadata JSON string → dict
    parsed = []
    for rid, summary, metadata, distance in rows:
        parsed.append({
            "id": rid,
            "summary": summary,
            "metadata": json.loads(metadata),
            "distance": distance
        })

    return parsed


def search_as_dataframe(db: sqlite3.Connection, query: str, k: int = 5):
    """Returns search results as a pandas DataFrame."""
    rows = _search_internal(db, query, k)
    return pd.DataFrame(rows)


def search_as_dict(db: sqlite3.Connection, query: str, k: int = 5):
    """Returns results as a list of Python dicts."""
    return _search_internal(db, query, k)


# DB test
# add_entry(db,
#     "A dog riding a skateboard at the beach.",
#     {"type": "image", "path": "images/dog_skate.png"}
# )
#
# add_entry(db,
#     "Guide on how to bake sourdough bread in simple steps.",
#     {"type": "url", "source": "https://example.com/bread"}
# )
#
# # Search
# print("\n--- DataFrame Results ---")
# df = search_as_dataframe(db, "dog on skateboard")
# print(df)
#
# print("\n--- Dict Results ---")
# d = search_as_dict(db, "dog on skateboard")
# print(d)
