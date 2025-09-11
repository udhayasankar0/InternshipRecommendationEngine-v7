# model_utils.py

from sentence_transformers import SentenceTransformer, util
import numpy as np
import joblib
import os
import torch
from typing import List, Dict, Set
import pandas as pd

import config
from rules import parse_skills, normalize_skill

def get_model() -> SentenceTransformer:
    """Loads the SBERT model."""
    return SentenceTransformer(config.MODEL_NAME)

def create_job_embeddings(df: pd.DataFrame, model: SentenceTransformer, cache_path: str, recompute: bool = False) -> torch.Tensor:
    """Creates and caches embeddings for job listings."""
    if os.path.exists(cache_path) and not recompute:
        print(f"Loading cached job embeddings from {cache_path}...")
        return joblib.load(cache_path)
    
    print("Creating and caching new job embeddings...")
    # CORRECTED: Use 'role' and 'Skills' columns from the CSV
    df['combined_text'] = df['role'].astype(str) + " " + df['Skills'].fillna('').astype(str)
    
    # Generate embeddings
    job_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
    
    # Cache embeddings
    joblib.dump(job_embeddings, cache_path)
    print("Job embeddings cached successfully.")
    
    return job_embeddings

def create_skill_embeddings(df: pd.DataFrame, model: SentenceTransformer, cache_path: str, recompute: bool = False) -> Dict[str, torch.Tensor]:
    """Creates and caches embeddings for unique skill tokens."""
    if os.path.exists(cache_path) and not recompute:
        print(f"Loading cached skill embeddings from {cache_path}...")
        return joblib.load(cache_path)

    print("Creating and caching new skill embeddings...")
    all_skills: Set[str] = set()
    df['parsed_skills'].apply(lambda skills: all_skills.update(skills))
    
    unique_skills = sorted(list(all_skills))
    print(f"Found {len(unique_skills)} unique skills to embed.")
    
    skill_embeddings_tensor = model.encode(unique_skills, convert_to_tensor=True)
    
    skill_embedding_dict = {skill: emb for skill, emb in zip(unique_skills, skill_embeddings_tensor)}
    
    joblib.dump(skill_embedding_dict, cache_path)
    print("Skill embeddings cached successfully.")
    
    return skill_embedding_dict

def create_user_embedding(user_profile: dict, model: SentenceTransformer) -> torch.Tensor:
    """Creates an embedding for the user profile."""
    user_text = " ".join(user_profile['skills']) + " " + user_profile['qualification']
    return model.encode(user_text, convert_to_tensor=True)

def get_semantic_scores(user_embedding: torch.Tensor, job_embeddings: torch.Tensor) -> np.ndarray:
    """Calculates cosine similarity between user and job embeddings."""
    if user_embedding.device != job_embeddings.device:
        job_embeddings = job_embeddings.to(user_embedding.device)
        
    cosine_scores = util.dot_score(user_embedding, job_embeddings)
    
    return cosine_scores[0].cpu().numpy()