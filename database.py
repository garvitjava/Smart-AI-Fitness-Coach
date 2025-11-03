import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = 'fitness_data.db'

def setup_database():
    """Creates the database and tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # --- *** MODIFIED TABLE *** ---
    # Added new columns to store analysis results
    c.execute('''
    CREATE TABLE IF NOT EXISTS workouts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        exercise TEXT NOT NULL,
        reps INTEGER,
        avg_depth REAL,
        avg_form_score REAL,
        consistency_score REAL
    )
    ''')
    
    # Table for detailed time-series data (unchanged)
    c.execute('''
    CREATE TABLE IF NOT EXISTS workout_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout_id INTEGER NOT NULL,
        frame_time INTEGER NOT NULL,
        main_angle REAL,
        form_angle REAL,
        FOREIGN KEY (workout_id) REFERENCES workouts(id)
    )
    ''')
    
    conn.commit()
    conn.close()

# --- *** MODIFIED FUNCTION *** ---
def save_workout(username, summary, time_series_data, analysis_results):
    """Saves a completed workout session, its time-series data, and its analysis."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Get analysis results, using .get() with a default of 0 in case analysis failed
    avg_depth = analysis_results.get('avg_depth', 0)
    avg_form_score = analysis_results.get('avg_form_score', 0)
    consistency_score = analysis_results.get('consistency_score', 0)
    
    try:
        # 1. Insert the main workout summary with analysis
        c.execute("""
            INSERT INTO workouts 
            (username, timestamp, exercise, reps, avg_depth, avg_form_score, consistency_score) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username, summary['Timestamp'], summary['Exercise'], summary['Reps'],
                avg_depth, avg_form_score, consistency_score
            )
        )
        
        workout_id = c.lastrowid
        
        # 2. Prepare and insert the time-series data
        data_to_insert = []
        main_angles = time_series_data['main_angle']
        form_angles = time_series_data['form_angle']
        
        for i in range(len(main_angles)):
            data_to_insert.append((
                workout_id,
                i,
                main_angles[i],
                form_angles[i]
            ))
        
        c.executemany("INSERT INTO workout_data (workout_id, frame_time, main_angle, form_angle) VALUES (?, ?, ?, ?)",
                      data_to_insert)
        
        conn.commit()
        
    except Exception as e:
        print(f"Error saving workout: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_history(username):
    """Fetches the workout history summary for a user as a DataFrame."""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query(
            "SELECT id, timestamp, exercise, reps FROM workouts WHERE username = ? ORDER BY timestamp DESC",
            conn,
            params=(username,)
        )
        return df
    except Exception as e:
        print(f"Error fetching history: {e}")
        return pd.DataFrame(columns=["id", "timestamp", "exercise", "reps"])
    finally:
        conn.close()

def get_workout_details(workout_id):
    """Fetches the full time-series data for a specific workout ID."""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query(
            "SELECT frame_time, main_angle, form_angle FROM workout_data WHERE workout_id = ? ORDER BY frame_time ASC",
            conn,
            params=(int(workout_id),)
        )
        return df['main_angle'], df['form_angle'], df
        
    except Exception as e:
        print(f"Error fetching workout details: {e}")
        return None, None, None
    finally:
        conn.close()

# --- *** NEW FUNCTION *** ---
def get_progress_data(username, exercise_type):
    """Fetches all saved analysis data for a specific exercise to show progress."""
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, reps, avg_depth, avg_form_score, consistency_score 
            FROM workouts 
            WHERE username = ? AND exercise = ?
            ORDER BY timestamp ASC
            """,
            conn,
            params=(username, exercise_type)
        )
        # Convert timestamp to datetime objects for better plotting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error fetching progress data: {e}")
        return pd.DataFrame(columns=["timestamp", "reps", "avg_depth", "avg_form_score", "consistency_score"])
    finally:
        conn.close()