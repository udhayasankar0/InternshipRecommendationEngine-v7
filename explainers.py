import pandas as pd
from typing import List, Dict, Any, Optional
import config

def generate_why_tags(internship: Dict[str, Any], user_profile: Dict[str, Any]) -> List[str]:
    """
    Legacy function to maintain compatibility.
    Generates simple 'why this fits' tags for a recommended internship.
    """
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

    # Location and distance
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

def generate_detailed_explanation(internship: Dict[str, Any], user_profile: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Generates a detailed explanation of why an internship matches the user's profile.
    All scores are on a 0-100 scale.
    """
    # Convert decimal scores to 0-100 scale
    semantic_score = internship.get('semantic_score', 0.0) * 100
    skill_score = len(internship.get('skill_matches', [])) / len(user_profile.get('skills', [])) * 100
    location_score = internship.get('location_score', 0.0) * 100
    stipend_score = internship.get('stipend_score', 0.5) * 100
    date_score = internship.get('date_score', 0.5) * 100

    explanation = {
        "why_this_fits": {
            "semantic_match": {
                "score": round(semantic_score),
                "weight": round(weights['semantic_score'] * 100),
                "weighted_score": round(semantic_score * weights['semantic_score']),
                "explanation": get_semantic_explanation(semantic_score)
            },
            "skill_match": {
                "score": round(skill_score),
                "weight": round(weights['skill_overlap_ratio'] * 100),
                "weighted_score": round(skill_score * weights['skill_overlap_ratio']),
                "matched_skills": [m['skill'] for m in internship.get('skill_matches', [])],
                "explanation": get_skill_match_explanation(internship.get('skill_matches', []), user_profile.get('skills', []))
            },
            "location_match": {
                "score": round(location_score),
                "weight": round(weights['location_score'] * 100),
                "weighted_score": round(location_score * weights['location_score']),
                "explanation": get_location_explanation(internship, user_profile, location_score)
            },
            "stipend_match": {
                "score": round(stipend_score),
                "weight": round(weights['stipend_score'] * 100),
                "weighted_score": round(stipend_score * weights['stipend_score']),
                "explanation": get_stipend_explanation(internship.get('stipend_numeric'), user_profile.get('min_stipend'))
            },
            "deadline_match": {
                "score": round(date_score),
                "weight": round(weights['date_score'] * 100),
                "weighted_score": round(date_score * weights['date_score']),
                "explanation": get_deadline_explanation(internship.get('deadline'), user_profile.get('available_from'))
            }
        }
    }
    
    total_score = sum(factor["weighted_score"] for factor in explanation["why_this_fits"].values())
    explanation["why_this_fits"]["total_score"] = round(total_score)
    
    return explanation

def get_semantic_explanation(score: float) -> str:
    if score >= 80:
        return "Excellent semantic match with your profile and qualifications"
    elif score >= 60:
        return "Good semantic alignment with your background"
    elif score >= 40:
        return "Moderate relevance to your profile"
    else:
        return "Limited semantic match with your background"

def get_skill_match_explanation(matched_skills: List[Dict[str, str]], user_skills: List[str]) -> str:
    match_ratio = len(matched_skills) / len(user_skills)
    skills_str = ", ".join(f"{m['skill']} ({m['mode']})" for m in matched_skills)
    
    if match_ratio >= 0.75:
        return f"Strong skill match: {skills_str}"
    elif match_ratio >= 0.5:
        return f"Good skill overlap: {skills_str}"
    else:
        return f"Some relevant skills: {skills_str}"

def get_location_explanation(job_data: Dict[str, Any], user_profile: Dict[str, Any], score: float) -> str:
    job_city = job_data.get('city', job_data.get('location', 'N/A'))
    distance_km = job_data.get('distance_km')
    
    if pd.notna(distance_km):
        if score >= 90:
            return f"Perfect location match in {job_city} ({distance_km:.1f} km)"
        elif score >= 70:
            return f"Good location in {job_city} ({distance_km:.1f} km)"
        else:
            return f"Location may require consideration: {job_city} ({distance_km:.1f} km)"
    else:
        return f"Location information limited: {job_city}"

def get_stipend_explanation(job_stipend: Optional[float], min_stipend: Optional[float]) -> str:
    if not job_stipend or not min_stipend:
        return "Stipend information incomplete"
    
    if job_stipend >= min_stipend:
        ratio = job_stipend / min_stipend
        if ratio >= 1.5:
            return f"Excellent compensation: ₹{int(job_stipend):,} (significantly above your minimum)"
        else:
            return f"Meets your minimum stipend requirement: ₹{int(job_stipend):,}"
    else:
        percentage = (job_stipend / min_stipend) * 100
        return f"Below your minimum stipend requirement: ₹{int(job_stipend):,} ({percentage:.0f}% of desired ₹{int(min_stipend):,})"

def get_deadline_explanation(deadline: Optional[str], available_from: Optional[str]) -> str:
    if not deadline or pd.isna(deadline):
        return "No application deadline specified - position is open"
    
    try:
        deadline_date = pd.to_datetime(deadline)
        available_date = pd.to_datetime(available_from) if available_from else pd.Timestamp.now()
        days_until = (deadline_date - available_date).days
        
        if days_until < 0:
            return "Application deadline has passed"
        elif days_until < 7:
            return f"Urgent: Only {days_until} days left to apply"
        elif days_until < 14:
            return f"Application window closing soon: {days_until} days left"
        else:
            return f"Plenty of time to apply: {days_until} days until deadline"
    except (ValueError, TypeError):
        return "Deadline information unclear"