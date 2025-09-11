# geolocation.py

import pandas as pd
from haversine import haversine, Unit
from typing import Optional, Tuple

import config

def get_location_score(job_row: pd.Series, user_profile: dict) -> Tuple[float, Optional[float]]:
    """
    Calculates a location score based on Haversine distance or city matching.
    Returns the score and the distance in kilometers.
    """
    user_lat = user_profile.get('lat')
    user_lon = user_profile.get('lon')
    job_lat = job_row.get('lat')
    job_lon = job_row.get('lon')
    
    distance_km = None
    score = 0.0

    # 1. Highest precision: Haversine distance if coordinates are available
    if pd.notna(user_lat) and pd.notna(user_lon) and pd.notna(job_lat) and pd.notna(job_lon):
        user_coords = (user_lat, user_lon)
        job_coords = (job_lat, job_lon)
        distance_km = haversine(user_coords, job_coords, unit=Unit.KILOMETERS)
        
        # Linear decay score
        radius_km = config.LOCATION_RADIUS_KM
        score = max(0.0, 1.0 - (distance_km / radius_km))

    # 2. Fallback: Exact city match if coordinates are missing
    else:
        job_city = str(job_row.get('city', '')).lower().strip()
        user_city = str(user_profile.get('preferred_location', '')).lower().strip()
        if user_city and job_city and user_city == job_city:
            score = 1.0  # Perfect score for same city
            distance_km = 0.0

    # 3. Consider remote work preference as a potential boost
    job_loc_lower = str(job_row.get('location', '')).lower()
    if user_profile.get('remote_ok', False) and ('remote' in job_loc_lower or 'work from home' in job_loc_lower):
        score = max(score, 0.8)

    return score, distance_km