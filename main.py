# main.py

import pandas as pd
import json
import argparse
import ast
# CORRECTED: Added 'Optional' to the import list
from typing import List, Dict, Any, Optional

from rules import (
    parse_skills, parse_stipend, parse_date, hard_skill_filter,
    score_stipend, score_deadline, match_skills
)
from geolocation import get_location_score
from model_utils import (
    get_model, create_job_embeddings, create_skill_embeddings,
    create_user_embedding, get_semantic_scores
)
from explainers import generate_why_tags, generate_detailed_explanation
import config

def parse_location_field(loc_field):
    """Safely parses the stringified list in the location column."""
    try:
        arr = ast.literal_eval(loc_field)
        if isinstance(arr, (list, tuple)) and len(arr) >= 1:
            city = str(arr[0]).strip()
            pincode = str(arr[1]).strip() if len(arr) > 1 else None
            return city, pincode
    except (ValueError, SyntaxError):
        return str(loc_field).strip(), None
    return str(loc_field).strip(), None

def load_data(dataset_path: str, pincode_path: str) -> pd.DataFrame:
    """Loads and preprocesses the internship and pincode data."""
    try:
        df = pd.read_csv(dataset_path)
        pincodes_df = pd.read_csv(pincode_path)
    except FileNotFoundError as e:
        print(f"Error: A required data file was not found. {e}")
        return pd.DataFrame()

    df[['city', 'pincode']] = df['location'].apply(lambda s: pd.Series(parse_location_field(s)))
    
    pincodes_df['pincode'] = pincodes_df['pincode'].astype(str)
    df['pincode'] = df['pincode'].astype(str).str.replace('.0', '', regex=False)
    
    df = df.merge(pincodes_df[['pincode', 'lat', 'lon']], on='pincode', how='left')

    df['parsed_skills'] = df['Skills'].apply(parse_skills)
    df['stipend_numeric'] = df['Stipend'].apply(parse_stipend)
    if 'Apply by Date' in df.columns:
        df['deadline'] = df['Apply by Date'].apply(parse_date)
    else:
        df['deadline'] = None
        
    return df

def recommend_internships(df: pd.DataFrame, user_profile: Dict[str, Any], model, job_embeddings, skill_embeddings, k: int, max_distance: Optional[float]) -> List[Dict[str, Any]]:
    """Recommends internships to a user."""
    
    df['skill_matches'] = df['parsed_skills'].apply(
        lambda x: match_skills(user_profile['skills'], x, skill_embeddings)
    )
    df['passes_skill_filter'] = df['skill_matches'].apply(hard_skill_filter)
    filtered_df = df[df['passes_skill_filter']].copy()
    
    if filtered_df.empty:
        print("No internships passed the hard skill filter.")
        return []

    user_embedding = create_user_embedding(user_profile, model)
    filtered_job_embeddings = job_embeddings[filtered_df.index]
    filtered_df['semantic_score'] = get_semantic_scores(user_embedding, filtered_job_embeddings)
    
    location_results = filtered_df.apply(lambda row: get_location_score(row, user_profile), axis=1, result_type='expand')
    filtered_df[['location_score', 'distance_km']] = location_results

    if max_distance is not None:
        initial_count = len(filtered_df)
        filtered_df = filtered_df[(filtered_df['distance_km'].isna()) | (filtered_df['distance_km'] <= max_distance)]
        print(f"Filtered out {initial_count - len(filtered_df)} jobs based on max distance of {max_distance} km.")

    if filtered_df.empty:
        print("No internships remaining after distance filter.")
        return []

    filtered_df['stipend_score'] = filtered_df['stipend_numeric'].apply(lambda x: score_stipend(x, user_profile.get('min_stipend')))
    filtered_df['date_score'] = filtered_df['deadline'].apply(lambda x: score_deadline(x, user_profile.get('available_from')))
    filtered_df['skill_overlap_ratio'] = filtered_df['skill_matches'].apply(lambda x: len(x) / len(user_profile['skills']))

    weights = config.SCORING_WEIGHTS
    filtered_df['final_score'] = (
        weights['semantic_score'] * filtered_df['semantic_score'] +
        weights['skill_overlap_ratio'] * filtered_df['skill_overlap_ratio'] +
        weights['location_score'] * filtered_df['location_score'] +
        weights['stipend_score'] * filtered_df['stipend_score'] +
        weights['date_score'] * filtered_df['date_score']
    )

    top_k = filtered_df.sort_values(by='final_score', ascending=False).head(k)
    
    recommendations = []
    for _, row in top_k.iterrows():
        row_dict = row.to_dict()
        row_dict['why_tags'] = generate_why_tags(row_dict, user_profile)
        recommendations.append(row_dict)
        
    return recommendations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Internship Recommender System')
    parser.add_argument('--dataset', type=str, default=config.DATASET_PATH)
    parser.add_argument('--user', type=str, default=config.USER_PROFILE_PATH)
    parser.add_argument('--pincode-file', type=str, default=config.PINCODE_DATA_PATH)
    parser.add_argument('--k', type=int, default=config.TOP_K)
    parser.add_argument('--recompute-embeddings', action='store_true')
    parser.add_argument('--recompute-skill-embs', action='store_true')
    parser.add_argument('--max-distance-km', type=float, default=None, help='Maximum distance in KM to consider for recommendations.')

    args = parser.parse_args()

    try:
        with open(args.user, 'r') as f:
            user_profile = json.load(f)
    except FileNotFoundError:
        print(f"Error: The user profile file was not found at {args.user}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file at {args.user}")
        exit()
    
    try:
        pincodes_df = pd.read_csv(args.pincode_file)
        user_pincode = user_profile.get('pincode')
        if user_pincode:
            user_loc_data = pincodes_df[pincodes_df['pincode'].astype(str) == str(user_pincode)]
            if not user_loc_data.empty:
                user_profile['lat'] = user_loc_data.iloc[0]['lat']
                user_profile['lon'] = user_loc_data.iloc[0]['lon']
    except FileNotFoundError:
        print(f"Warning: Pincode file not found at {args.pincode_file}. Location scoring will be limited.")


    df = load_data(args.dataset, args.pincode_file)
    if df.empty: exit()
    
    model = get_model()
    job_embeddings = create_job_embeddings(df, model, config.EMBEDDINGS_CACHE_PATH, args.recompute_embeddings)
    skill_embeddings = create_skill_embeddings(df, model, config.SKILL_EMBEDDINGS_CACHE_PATH, args.recompute_skill_embs)
    
    recommendations = recommend_internships(df, user_profile, model, job_embeddings, skill_embeddings, k=args.k, max_distance=args.max_distance_km)
    
    output_results = []
    print("\n--- Top Recommendations ---")
    for rec in recommendations:
        output_results.append({
            "id": rec['id'],
            "title": rec['role'],
            "company": rec['company'],
            "location": rec['location'],
            "stipend": rec['stipend_numeric'],
            "duration": rec['Duration'],
            "deadline": rec['deadline'],
            "score": rec['final_score'],
            "matched_skills": [m['skill'] for m in rec['skill_matches']],
            "distance_km": rec.get('distance_km'),
            "why": rec['why_tags'],
        })
    print(json.dumps(output_results, indent=2, default=str))

    print("\n--- DEBUG: Score Contribution Breakdown ---")
    weights = config.SCORING_WEIGHTS
    for i, rec in enumerate(recommendations):
        # Check for distance_km before formatting
        distance_str = f"{rec.get('distance_km'):.1f} km" if pd.notna(rec.get('distance_km')) else "N/A"
        detailed_explanation = generate_detailed_explanation(rec, user_profile, weights)
        why_this_fits = detailed_explanation["why_this_fits"]
        
        print(f"\n#{i+1}: {rec['role']} at {rec['company']} (Total Score: {why_this_fits['total_score']}/100)")
        print("\nWhy This Fits:")
        print(f"  Semantic Match ({why_this_fits['semantic_match']['weight']}%):")
        print(f"    Score: {why_this_fits['semantic_match']['score']}/100")
        print(f"    {why_this_fits['semantic_match']['explanation']}")
        
        print(f"\n  Skills ({why_this_fits['skill_match']['weight']}%):")
        print(f"    Score: {why_this_fits['skill_match']['score']}/100")
        print(f"    {why_this_fits['skill_match']['explanation']}")
        
        print(f"\n  Location ({why_this_fits['location_match']['weight']}%):")
        print(f"    Score: {why_this_fits['location_match']['score']}/100")
        print(f"    {why_this_fits['location_match']['explanation']}")
        
        print(f"\n  Stipend ({why_this_fits['stipend_match']['weight']}%):")
        print(f"    Score: {why_this_fits['stipend_match']['score']}/100")
        print(f"    {why_this_fits['stipend_match']['explanation']}")
        
        print(f"\n  Deadline ({why_this_fits['deadline_match']['weight']}%):")
        print(f"    Score: {why_this_fits['deadline_match']['score']}/100")
        print(f"    {why_this_fits['deadline_match']['explanation']}")