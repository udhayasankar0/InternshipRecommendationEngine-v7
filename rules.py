import pandas as pd
import re
from dateutil.parser import parse
import ast
from typing import List, Optional, Set, Dict, Tuple, Any
from rapidfuzz import process, fuzz
from sentence_transformers import util
import torch

import config

# ... (SYNONYMS and other functions remain the same) ...
SYNONYMS = {
    "natural language processing": "nlp",
    "js": "javascript",
    "reactjs": "react",
    "react.js": "react",
    "cpp": "c++",
    "c plus plus": "c++",
    "bleander": "blender",
    "unity": "unity3d"
}

def normalize_skill(skill: str) -> str:
    """Normalizes a single skill string."""
    if not isinstance(skill, str):
        return ""
    s = skill.lower().strip()
    s = re.sub(r'[\s\-_]+', ' ', s) # Replace hyphens/underscores with spaces
    s = re.sub(r'[^\w\s+#.]', '', s) # Remove punctuation except for essentials
    return SYNONYMS.get(s, s)

def parse_skills(skills_str: str) -> List[str]:
    """Parses the 'Skills and Perks' string into a list of normalized skills."""
    if not skills_str or pd.isna(skills_str):
        return []
    try:
        skills_list = ast.literal_eval(skills_str)
        if isinstance(skills_list, list):
            return list(set(normalize_skill(s) for s in skills_list if s))
    except (ValueError, SyntaxError):
        return list(set(normalize_skill(s) for s in skills_str.split(',') if s))
    return []

def match_skills(user_skills: List[str], job_skills: List[str], skill_embeddings: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """Matches user skills against job skills using a multi-stage pipeline."""
    if not user_skills or not job_skills:
        return []
    norm_user_skills = [normalize_skill(s) for s in user_skills]
    norm_job_skills = [normalize_skill(s) for s in job_skills]
    matched_skills = {}
    for js in norm_job_skills:
        if js in norm_user_skills:
            if js not in matched_skills:
                matched_skills[js] = {'mode': 'exact', 'score': 1.0}
    remaining_job_skills = [js for js in norm_job_skills if js not in matched_skills]
    for js in remaining_job_skills:
        best_match = process.extractOne(js, norm_user_skills, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= config.FUZZY_CUTOFF:
             if js not in matched_skills:
                matched_skills[js] = {'mode': f'fuzzy({int(best_match[1])})', 'score': best_match[1] / 100.0}
    remaining_job_skills = [js for js in norm_job_skills if js not in matched_skills]
    for js in remaining_job_skills:
        js_emb = skill_embeddings.get(js)
        if js_emb is None: continue
        best_sbert_score = 0
        for us in norm_user_skills:
            us_emb = skill_embeddings.get(us)
            if us_emb is not None:
                sim = util.pytorch_cos_sim(js_emb, us_emb).item()
                if sim > best_sbert_score:
                    best_sbert_score = sim
        if best_sbert_score >= config.SBERT_CUTOFF:
            if js not in matched_skills:
                matched_skills[js] = {'mode': f'sbert({best_sbert_score:.2f})', 'score': best_sbert_score}
    return [{'skill': k, **v} for k, v in matched_skills.items()]

def hard_skill_filter(matched_skills: List[Dict[str, Any]]) -> bool:
    """Filters internships based on the number of matched skills."""
    return len(matched_skills) >= config.MIN_SKILL_MATCHES

def parse_stipend(stipend_str: str) -> Optional[float]:
    if not isinstance(stipend_str, str): return None
    stipend_str = stipend_str.lower()
    if 'performance based' in stipend_str or 'unpaid' in stipend_str: return 0.0
    numbers = [int(s) for s in re.findall(r'\d+', stipend_str)]
    if not numbers: return None
    stipend = float(numbers[0])
    if '/week' in stipend_str: stipend *= 4.33
    return stipend

def parse_date(date_str: str) -> Optional[str]:
    try:
        return parse(date_str, dayfirst=False).strftime('%Y-%m-%d')
    except (ValueError, TypeError, AttributeError):
        return None

def score_stipend(job_stipend: Optional[float], min_stipend: Optional[float]) -> float:
    if job_stipend is None or min_stipend is None or min_stipend == 0:
        return 0.5
    if job_stipend >= min_stipend:
        return 1.0
    ratio = job_stipend / min_stipend
    return max(0.0, ratio)

def score_deadline(apply_by_date: Optional[str], available_from: Optional[str]) -> float:
    """
    Scores the internship based on its application deadline.
    A missing deadline is treated as neutral (0.5).
    """
    # CORRECTED: Explicitly handle missing deadlines as neutral
    if not apply_by_date or pd.isna(apply_by_date) or not available_from:
        return 0.5
    try:
        deadline = pd.to_datetime(apply_by_date)
        availability = pd.to_datetime(available_from)
        return 1.0 if deadline >= availability else 0.0
    except (ValueError, TypeError):
        return 0.5 # Fallback for parsing errors```