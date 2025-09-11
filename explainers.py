import pandas as pd
from typing import List, Dict, Any
import config

def generate_why_tags(internship: Dict[str, Any], user_profile: Dict[str, Any]) -> List[str]:
    """Generates detailed 'why this fits' tags for a recommended internship."""
    tags = []

    # Semantic match score
    semantic_score = internship.get('semantic_score', 0.0)
    if semantic_score > config.SEMANTIC_HIGH_THRESHOLD:
        tags.append(f"Semantic match: high (score: {semantic_score:.2f})")
    elif semantic_score > config.SEMANTIC_GOOD_THRESHOLD:
        tags.append(f"Semantic match: good (score: {semantic_score:.2f})")

    # Skill match details
    matched_skills = internship.get('skill_matches', [])
    if matched_skills:
        skill_details = [f"{m['skill']} ({m['mode']})" for m in matched_skills]
        tags.append(f"Matches {len(matched_skills)}/{len(user_profile.get('skills',[]))} skills: {', '.join(skill_details)}")

    # Location score and distance
    location_score = internship.get('location_score', 0.0)
    distance_km = internship.get('distance_km')
    job_city = internship.get('city', internship.get('location', 'N/A'))
    if pd.notna(distance_km):
        tags.append(f"Location: {job_city} - {distance_km:.1f} km away (score: {location_score:.2f})")
    elif location_score > 0.0:
        tags.append(f"Location: {job_city} - good fit (score: {location_score:.2f})")

    # Stipend score
    job_stipend = internship.get('stipend_numeric')
    min_stipend = user_profile.get('min_stipend')
    stipend_score = internship.get('stipend_score', 0.0)
    if pd.notna(job_stipend) and min_stipend is not None:
        if job_stipend >= min_stipend:
            tags.append(f"Stipend ₹{int(job_stipend)} - meets your ₹{int(min_stipend)} min (score: {stipend_score:.2f})")
        else:
            tags.append(f"Stipend ₹{int(job_stipend)} - below your ₹{int(min_stipend)} min (score: {stipend_score:.2f})")

    # Deadline
    if 'deadline' in internship and pd.notna(internship['deadline']):
        tags.append(f"Apply by: {internship['deadline']}")
    else:
        tags.append("Apply by: No deadline specified")


    return tags