import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler

# --- Constants ---
# Rep segmentation thresholds
REP_THRESHOLDS = {
    'Squats': {'peak_height': -100, 'distance': 15}, # Find peaks on inverted signal
    'Push-ups': {'peak_height': -90, 'distance': 15},
    'Bicep Curls': {'peak_height': -60, 'distance': 10} 
}
# --- *** NEW: Define a window size for scoring *** ---
# This is how many frames to look at before and after the peak of the rep
REP_WINDOW_SIZE = 15 

# --- This template loading logic is correct ---
TEMPLATES = {}

try:
    TEMPLATE_SQUAT = pd.read_csv('templates/perfect_squat.csv')['main_angle'].values
    TEMPLATES['Squats'] = TEMPLATE_SQUAT
except FileNotFoundError:
    print("WARNING: 'templates/perfect_squat.csv' not found. Squat analysis disabled.")
except Exception as e:
    print(f"Error loading squat template: {e}")

try:
    TEMPLATE_PUSHUP = pd.read_csv('templates/perfect_pushup.csv')['main_angle'].values
    TEMPLATES['Push-ups'] = TEMPLATE_PUSHUP
except FileNotFoundError:
    print("WARNING: 'templates/perfect_pushup.csv' not found. Push-up analysis disabled.")
except Exception as e:
    print(f"Error loading pushup template: {e}")

try:
    TEMPLATE_CURL = pd.read_csv('templates/perfect_curl.csv')['main_angle'].values
    TEMPLATES['Bicep Curls'] = TEMPLATE_CURL
except FileNotFoundError:
    print("WARNING: 'templates/perfect_curl.csv' not found. Bicep Curl analysis disabled.")
except Exception as e:
    print(f"Error loading curl template: {e}")
# --- End of template loading ---


# --- Helper Functions (Unchanged) ---

def segment_reps(main_angle_series, exercise_type):
    """
    Finds the index of the "bottom" (lowest angle) of each rep.
    """
    if exercise_type not in REP_THRESHOLDS:
        print(f"Warning: No rep threshold defined for {exercise_type}")
        return np.array([])
        
    params = REP_THRESHOLDS[exercise_type]
    
    inverted_series = -np.array(main_angle_series)
    
    peaks, _ = find_peaks(
        inverted_series, 
        height=params['peak_height'], 
        distance=params['distance']
    )
    return peaks

def normalize_series(series):
    """Normalizes a time series to a 0-1 scale."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(series.reshape(-1, 1)).flatten()

# --- *** START OF CORRECTED SCORING FUNCTION *** ---
def calculate_form_score_dtw(rep_segment, template_series):
    """
    Calculates a form score from 0-100 using Dynamic Time Warping (DTW).
    """
    if rep_segment.size < 5 or template_series.size == 0:
        return 0

    try:
        # Normalize both series to 0-1 range
        rep_normalized = normalize_series(rep_segment)
        template_normalized = normalize_series(template_series)
    except ValueError:
        return 0 # Error during normalization (e.g., flat segment)

    # Use standard DTW distance (sum of distances)
    distance = dtw.distance(rep_normalized, template_normalized)
    
    # --- New Heuristic ---
    # The 'distance' is the sum of distances along the warping path.
    # We normalize this by the length of the rep segment.
    
    if len(rep_normalized) == 0:
        return 0

    normalized_distance = distance / len(rep_normalized)
    
    # Now, convert this normalized distance to a score.
    # A 'perfect' match has a normalized_distance of 0 (score 100).
    # A 'terrible' match might have an avg error of 0.5 per step.
    # We'll set this 'terrible' match to be a score of 0.
    
    max_norm_dist_heuristic = 0.5 
    
    score = 100 * (1 - (normalized_distance / max_norm_dist_heuristic))
    
    return max(0, min(100, score)) # Clamp score between 0 and 100
# --- *** END OF CORRECTED SCORING FUNCTION *** ---


# --- Main Analysis Function (CORRECTED) ---

def analyze_workout(main_angle_series, form_angle_series, exercise_type):
    """
    Performs a full analysis on a completed workout's time-series data.
    """
    # Clean data: remove None values and filter out 0s
    main_angles = np.array([a for a in main_angle_series if a is not None and a > 0])
    form_angles = np.array([f for f in form_angle_series if f is not None and f > 0])
    
    if main_angles.size == 0 or form_angles.size == 0:
         return {"error": "No valid angle data found."}

    # 1. Find all rep indices
    rep_peaks = segment_reps(main_angles, exercise_type)
    
    if rep_peaks.size == 0:
        return {"error": "No complete reps were detected."}

    # 2. Analyze Rep Depth and Consistency
    rep_depths = main_angles[rep_peaks]
    avg_depth = np.mean(rep_depths)
    consistency_score = np.std(rep_depths)

    # 3. Analyze Form Degradation
    form_at_peaks = form_angles[rep_peaks]
    
    num_reps_to_compare = max(1, len(form_at_peaks) // 5)
    
    avg_form_start = np.mean(form_at_peaks[:num_reps_to_compare])
    avg_form_end = np.mean(form_at_peaks[-num_reps_to_compare:])
    
    form_degradation = avg_form_end - avg_form_start

    # --- *** START OF CORRECTED SCORING LOGIC *** ---
    # 4. Calculate DTW Form Score for each rep
    form_scores = []
    template = TEMPLATES.get(exercise_type)
    
    if template is not None:
        # Loop through each peak (deepest part of the rep)
        for peak_idx in rep_peaks:
            # Define the start and end of the rep "window"
            start_idx = np.max([0, peak_idx - REP_WINDOW_SIZE])
            end_idx = np.min([len(main_angles) - 1, peak_idx + REP_WINDOW_SIZE])
            
            # Get the rep segment
            rep_segment = main_angles[start_idx:end_idx]
            
            # Ensure segment is long enough and score it
            if len(rep_segment) > 5: 
                score = calculate_form_score_dtw(rep_segment, template)
                form_scores.append(score)
    # --- *** END OF CORRECTED SCORING LOGIC *** ---
    
    avg_form_score = np.mean(form_scores) if form_scores else 0
    
    return {
        "total_reps": len(rep_peaks),
        "avg_depth": avg_depth,
        "consistency_score": consistency_score,
        "form_degradation": form_degradation,
        "avg_form_score": avg_form_score,
    }