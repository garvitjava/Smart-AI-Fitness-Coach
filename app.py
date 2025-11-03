import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
import numpy as np # Import numpy for cleaning data

# Import your modules
import database as db
import llm_integration as llm
from exercise_processor import ExerciseProcessor
import analysis  # Import your new analysis module

# --- Database Setup ---
# This will create the new database file if it's missing
db.setup_database()

# --- Page Configuration ---
st.set_page_config(page_title="Smart AI Fitness Coach", layout="wide")
st.title("Smart AI Fitness Coach 🏋️‍♂️")
st.caption("Using MediaPipe, Time-Series Analysis, and Groq LLM")

# --- Session State Initialization ---
if 'run' not in st.session_state:
    st.session_state.run = False
if 'exercise_processor' not in st.session_state:
    st.session_state.exercise_processor = None
if 'llm_advice' not in st.session_state:
    st.session_state.llm_advice = ""
if 'username' not in st.session_state:
    st.session_state.username = "DefaultUser"

# --- *** MODIFIED: Added Tab 3 *** ---
tab1, tab2, tab3 = st.tabs(["🏋️‍♂️ Live Workout", "📊 Workout Analysis", "📈 Progress"])

# ==============================================================================
# --- TAB 1: LIVE WORKOUT ---
# ==============================================================================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Your Live Workout")
        video_placeholder = st.empty()
        if not st.session_state.run:
            video_placeholder.info("Click 'Start Workout' in the sidebar to begin!")

    with col2:
        st.header("Workout Analysis")
        
        stats_cols = st.columns(3)
        rep_placeholder = stats_cols[0].empty()
        state_placeholder = stats_cols[1].empty()
        feedback_placeholder = stats_cols[2].empty()
        
        st.subheader("Joint Angle (Time Series)")
        chart_placeholder = st.empty()

        if not st.session_state.run:
            rep_placeholder.metric("Reps", 0)
            state_placeholder.metric("State", "STOPPED")
            feedback_placeholder.info("Your stats will appear here.")
            chart_placeholder.info("Your joint angle graph will appear here.")
    
    advice_placeholder = st.empty()
    if st.session_state.llm_advice:
        advice_placeholder.header("AI Coach Feedback")
        advice_placeholder.markdown(st.session_state.llm_advice)

# ==============================================================================
# --- TAB 2: WORKOUT ANALYSIS (Formerly Dashboard) ---
# ==============================================================================
with tab2:
    st.header(f"Single Workout Analysis for {st.session_state.username}")
    
    if st.button("Refresh Data"):
        st.rerun() 
    
    history_df = db.get_history(st.session_state.username)
    
    if history_df.empty:
        st.info("No workouts found. Complete a workout in the 'Live Workout' tab!")
    else:
        history_df['display'] = history_df.apply(
            lambda row: f"{row['timestamp']} - {row['exercise']} ({row['reps']} reps)", axis=1
        )
        selected_display = st.selectbox(
            "Select a workout to analyze:", 
            history_df['display']
        )
        
        selected_row = history_df[history_df['display'] == selected_display].iloc[0]
        selected_id = selected_row['id']
        selected_exercise = selected_row['exercise']
        
        main_angles, form_angles, full_df = db.get_workout_details(selected_id)
        
        if main_angles is None or main_angles.empty:
            st.error("Could not load data for this workout.")
        else:
            st.subheader(f"Analysis for {selected_display}")
            
            with st.spinner("Analyzing your workout..."):
                analysis_results = analysis.analyze_workout(main_angles, form_angles, selected_exercise)
            
            if "error" in analysis_results:
                st.error(f"Analysis Error: {analysis_results['error']}")
            else:
                st.success("Analysis complete!")
                
                # (Help text and labels are unchanged)
                score_help = "A score from 0-100 comparing your rep's main joint angle to a 'perfect' rep using Dynamic Time Warping (DTW). Higher is better."
                consistency_help = "The standard deviation of your rep depth (lowest angle). A lower number means you were more consistent. (Lower is better)"
                
                if selected_exercise == "Squats":
                    angle_name, form_name = "Knee Angle", "Hip Angle"
                    degradation_help = "Change in your average Hip Angle. A negative value means your chest may be caving in more as you get tired."
                elif selected_exercise == "Push-ups":
                    angle_name, form_name = "Elbow Angle", "Back Angle"
                    degradation_help = "Change in your average Back Angle. A negative value means your back may be sagging more as you get tired."
                elif selected_exercise == "Bicep Curls":
                    angle_name, form_name = "Elbow Angle", "Shoulder Angle"
                    degradation_help = "Change in your average Shoulder Angle. A positive value means you may be swinging your arms more as you get tired."
                else:
                    angle_name, form_name = "Main Angle", "Form Angle"
                    degradation_help = "Change in your average form angle from the start to the end of your set."

                metric_cols = st.columns(4)
                metric_cols[0].metric(
                    "Avg. Form Score", 
                    f"{analysis_results['avg_form_score']:.1f} / 100",
                    help=score_help
                )
                metric_cols[1].metric(
                    "Avg. Rep Depth", 
                    f"{analysis_results['avg_depth']:.1f}°"
                )
                metric_cols[2].metric(
                    "Depth Consistency", 
                    f"{analysis_results['consistency_score']:.2f}°", 
                    help=consistency_help
                )
                metric_cols[3].metric(
                    "Form Degradation", 
                    f"{analysis_results['form_degradation']:.2f}°", 
                    help=degradation_help
                )
                
                st.subheader("Full Workout Data")
                plot_df = full_df.rename(columns={'main_angle': angle_name, 'form_angle': form_name})
                st.line_chart(plot_df, x='frame_time', y=[angle_name, form_name])

# ==============================================================================
# --- *** NEW: TAB 3: PROGRESS *** ---
# ==============================================================================
with tab3:
    st.header(f"Long-Term Progress for {st.session_state.username}")
    
    # Let user pick which exercise to view
    exercise_to_plot = st.selectbox(
        "Select an exercise to track:",
        ("Squats", "Push-ups", "Bicep Curls")
    )
    
    if exercise_to_plot:
        # Get all progress data for that exercise
        progress_df = db.get_progress_data(st.session_state.username, exercise_to_plot)
        
        if progress_df.empty:
            st.info(f"No workout history found for {exercise_to_plot}. Complete some workouts!")
        else:
            # Let user pick which metric to plot
            metric_to_plot = st.selectbox(
                "Select a metric to track:",
                ("Reps per Workout", "Avg. Form Score", "Avg. Rep Depth", "Depth Consistency")
            )
            
            # Map the user-friendly name to the database column name
            metric_map = {
                "Reps per Workout": "reps",
                "Avg. Form Score": "avg_form_score",
                "Avg. Rep Depth": "avg_depth",
                "Depth Consistency": "consistency_score"
            }
            db_column_name = metric_map[metric_to_plot]
            
            # Rename columns for the chart legend
            chart_df = progress_df[['timestamp', db_column_name]].copy()
            chart_df = chart_df.rename(columns={db_column_name: metric_to_plot})
            
            # Plot the data
            st.subheader(f"{metric_to_plot} for {exercise_to_plot} Over Time")
            st.line_chart(chart_df, x='timestamp', y=metric_to_plot)
            
            st.subheader(f"Raw Data for {exercise_to_plot}")
            st.dataframe(progress_df)

# ==============================================================================
# --- SIDEBAR (Controls all tabs) ---
# ==============================================================================
st.sidebar.title("Controls")
st.sidebar.text_input("Enter your name:", value=st.session_state.username, key="username")

st.sidebar.markdown("---")

exercise_type = st.sidebar.selectbox(
    "Choose your exercise:",
    ("Squats", "Push-ups", "Bicep Curls"),
    key="exercise_choice"
)

if st.sidebar.button("Start Workout", type="primary"):
    st.session_state.llm_advice = ""
    advice_placeholder.empty() 
    
    st.session_state.run = True
    st.session_state.exercise_processor = ExerciseProcessor(st.session_state.exercise_choice)
    
    with tab1.container():
        video_placeholder.empty()
        chart_placeholder.empty()
    st.rerun() 

# --- *** MODIFIED: Stop Workout Logic *** ---
if st.sidebar.button("Stop Workout"):
    st.session_state.run = False
    if st.session_state.exercise_processor:
        summary = st.session_state.exercise_processor.get_summary()
        ts_data = st.session_state.exercise_processor.joint_time_series
        
        # --- Run analysis BEFORE saving ---
        main_angles_raw = ts_data['main_angle']
        form_angles_raw = ts_data['form_angle']
        
        # Clean data (must be identical to analysis.py)
        main_angles_clean = np.array([a for a in main_angles_raw if a is not None and a > 0])
        form_angles_clean = np.array([f for f in form_angles_raw if f is not None and f > 0])
        
        analysis_results = {} # Default empty dict
        if len(main_angles_clean) > 5: # Only analyze if there's enough data
             analysis_results = analysis.analyze_workout(
                 main_angles_clean, 
                 form_angles_clean, 
                 summary['Exercise']
             )
        
        # --- Save workout AND analysis results ---
        db.save_workout(st.session_state.username, summary, ts_data, analysis_results)
        
        if summary['Reps'] > 0: 
            groq_client = llm.get_groq_client()
            if groq_client:
                with st.spinner("Getting post-workout advice..."):
                    st.session_state.llm_advice = llm.get_workout_advice(groq_client, summary)
            else:
                st.session_state.llm_advice = "Could not connect to AI Coach. Is your API key set?"
        else:
            st.session_state.llm_advice = "Workout saved, but no reps were counted. Try again!"
        
        st.session_state.exercise_processor = None
    
    st.rerun()


# ==============================================================================
# --- Main Webcam Loop (Unchanged) ---
# ==============================================================================
if st.session_state.run and st.session_state.exercise_processor:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to open webcam. Please check permissions.")
        st.session_state.run = False
    else:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from webcam.")
                st.session_state.run = False
                break
            
            frame = cv2.flip(frame, 1)
            ep = st.session_state.exercise_processor
            frame = ep.process_frame(frame)
            
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            rep_placeholder.metric("Reps", ep.counter)
            state_placeholder.metric("State", ep.state.upper())
            feedback_placeholder.markdown(f"**Feedback:**<br>{ep.feedback}", unsafe_allow_html=True)

            ts_data = ep.joint_time_series
            plot_data = {
                'Main Angle': [v for v in ts_data['main_angle'] if v is not None],
                'Form Angle': [v for v in ts_data['form_angle'] if v is not None]
            }
            df = pd.DataFrame(plot_data)
            
            if ep.exercise_type == "Squats":
                angle_name, form_name = "Knee Angle", "Hip Angle"
            elif ep.exercise_type == "Push-ups":
                angle_name, form_name = "Elbow Angle", "Back Angle"
            elif ep.exercise_type == "Bicep Curls":
                angle_name, form_name = "Elbow Angle", "Shoulder Angle"
            else:
                angle_name, form_name = "Main", "Form"
            
            df.rename(columns={'Main Angle': angle_name, 'Form Angle': form_name}, inplace=True)
            
            if not df.empty:
                chart_placeholder.line_chart(df)

        cap.release()
else:
    if 'cap' in locals() and 'cap'in vars() and cap.isOpened():
        cap.release()