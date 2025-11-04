import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import av
import threading # Already imported, but needed for the fix
import time

# Import streamlit-webrtc components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration

# Import your modules
import database as db
import llm_integration as llm
from exercise_processor import ExerciseProcessor
import analysis

# --- Database Setup ---
db.setup_database()

# --- Page Configuration ---
st.set_page_config(page_title="Smart AI Fitness Coach", layout="wide")
st.title("Smart AI Fitness Coach 🏋️‍♂️")
st.caption("Using MediaPipe, Time-Series Analysis, and Groq LLM")

# --- Session State Initialization ---
if 'exercise_processor' not in st.session_state:
    st.session_state.exercise_processor = None
if 'llm_advice' not in st.session_state:
    st.session_state.llm_advice = ""
if 'username' not in st.session_state:
    st.session_state.username = "DefaultUser"
if 'webrtc_key' not in st.session_state:
    st.session_state.webrtc_key = "default"

# --- Placeholders ---
tab1, tab2, tab3 = st.tabs(["🏋️‍♂️ Live Workout", "📊 Workout Analysis", "📈 Progress"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Your Live Workout")
        video_placeholder = st.empty() 

    with col2:
        st.header("Workout Analysis")
        stats_cols = st.columns(3)
        rep_placeholder = stats_cols[0].empty()
        state_placeholder = stats_cols[1].empty()
        feedback_placeholder = stats_cols[2].empty()
        
        st.subheader("Joint Angle (Time Series)")
        chart_placeholder = st.empty()

        rep_placeholder.metric("Reps", 0)
        state_placeholder.metric("State", "STOPPED")
        feedback_placeholder.info("Your stats will appear here.")
        chart_placeholder.info("Your joint angle graph will appear here.")
    
    advice_placeholder = st.empty()
    if st.session_state.llm_advice:
        advice_placeholder.header("AI Coach Feedback")
        advice_placeholder.markdown(st.session_state.llm_advice)

# ==============================================================================
# --- *** MODIFIED: WEBRTC Video Processor *** ---
# ==============================================================================
class FitnessVideoProcessor(VideoProcessorBase):
    def __init__(self, exercise_type: str):
        self.lock = threading.Lock() 
        self.exercise_type = exercise_type
        self.exercise_processor = ExerciseProcessor(exercise_type)
        
        self.rep_count = 0
        self.state = "STOPPED"
        self.feedback = "Click Start to begin."
        self.chart_data = pd.DataFrame(columns=["Frame", "Main Angle", "Form Angle"])
        self.frame_count = 0

        # --- *** NEW: Thread-safe event for on_ended *** ---
        # This flag will be set by the 'on_ended' callback
        self.workout_ended = threading.Event()
        self.final_summary = None
        self.final_ts_data = None

    def get_summary_and_data(self):
        """Helper to safely get all data at the end."""
        with self.lock:
            return (
                self.exercise_processor.get_summary(),
                self.exercise_processor.joint_time_series
            )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Process the frame using your existing logic
        img = self.exercise_processor.process_frame(img)
        
        # Update live stats safely
        with self.lock:
            self.rep_count = self.exercise_processor.counter
            self.state = self.exercise_processor.state.upper()
            self.feedback = self.exercise_processor.feedback
            
            self.frame_count += 1
            main_angle = self.exercise_processor.joint_time_series['main_angle'][-1] if self.exercise_processor.joint_time_series['main_angle'] else None
            form_angle = self.exercise_processor.joint_time_series['form_angle'][-1] if self.exercise_processor.joint_time_series['form_angle'] else None
            
            new_data = {
                "Frame": self.frame_count,
                "Main Angle": main_angle if main_angle is not None else np.nan,
                "Form Angle": form_angle if form_angle is not None else np.nan
            }
            
            self.chart_data = pd.concat(
                [self.chart_data, pd.DataFrame([new_data])], 
                ignore_index=True
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # --- *** NEW: on_ended Callback *** ---
    def on_ended(self):
        """
        This method is called automatically by streamlit-webrtc
        when the stream is stopped (e.g., user clicks Stop).
        """
        with self.lock:
            # Get the final data one last time
            self.final_summary, self.final_ts_data = self.exercise_processor.get_summary(), self.exercise_processor.joint_time_series
            
        # Set the event flag to signal the main thread
        self.workout_ended.set()

# ==============================================================================
# --- TAB 2 & 3 (Unchanged) ---
# ==============================================================================
with tab2:
    # ... (Your code for Tab 2 is correct, no changes) ...
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
                metric_cols[0].metric("Avg. Form Score", f"{analysis_results['avg_form_score']:.1f} / 100", help=score_help)
                metric_cols[1].metric("Avg. Rep Depth", f"{analysis_results['avg_depth']:.1f}°")
                metric_cols[2].metric("Depth Consistency", f"{analysis_results['consistency_score']:.2f}°", help=consistency_help)
                metric_cols[3].metric("Form Degradation", f"{analysis_results['form_degradation']:.2f}°", help=degradation_help)
                
                st.subheader("Full Workout Data")
                plot_df = full_df.rename(columns={'main_angle': angle_name, 'form_angle': form_name})
                st.line_chart(plot_df, x='frame_time', y=[angle_name, form_name])


with tab3:
    # ... (Your code for Tab 3 is correct, no changes) ...
    st.header(f"Long-Term Progress for {st.session_state.username}")
    
    exercise_to_plot = st.selectbox("Select an exercise to track:", ("Squats", "Push-ups", "Bicep Curls"))
    
    if exercise_to_plot:
        progress_df = db.get_progress_data(st.session_state.username, exercise_to_plot)
        
        if progress_df.empty:
            st.info(f"No workout history found for {exercise_to_plot}. Complete some workouts!")
        else:
            metric_to_plot = st.selectbox("Select a metric to track:", ("Reps per Workout", "Avg. Form Score", "Avg. Rep Depth", "Depth Consistency"))
            
            metric_map = {
                "Reps per Workout": "reps", "Avg. Form Score": "avg_form_score",
                "Avg. Rep Depth": "avg_depth", "Depth Consistency": "consistency_score"
            }
            db_column_name = metric_map[metric_to_plot]
            
            chart_df = progress_df[['timestamp', db_column_name]].copy()
            chart_df = chart_df.rename(columns={db_column_name: metric_to_plot})
            
            st.subheader(f"{metric_to_plot} for {exercise_to_plot} Over Time")
            st.line_chart(chart_df, x='timestamp', y=metric_to_plot)
            
            st.subheader(f"Raw Data for {exercise_to_plot}")
            st.dataframe(progress_df)

# ==============================================================================
# --- SIDEBAR (Controls) ---
# ==============================================================================
st.sidebar.title("Controls")
st.sidebar.text_input("Enter your name:", value=st.session_state.username, key="username")
st.sidebar.markdown("---")
exercise_type = st.sidebar.selectbox(
    "Choose your exercise:", ("Squats", "Push-ups", "Bicep Curls"), key="exercise_choice"
)

# This button *initializes* the workout
if st.sidebar.button("Start Workout", type="primary"):
    st.session_state.llm_advice = ""
    advice_placeholder.empty() 
    
    st.session_state.exercise_processor = FitnessVideoProcessor(st.session_state.exercise_choice)
    st.session_state.webrtc_key = str(datetime.now().timestamp())
    st.rerun()

# ==============================================================================
# --- *** MODIFIED: Main Webcam Logic *** ---
# ==============================================================================

# --- NEW: Add RTC Configuration for STUN server ---
# This helps with connection issues on cloud platforms
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

if st.session_state.exercise_processor:
    processor = st.session_state.exercise_processor
    
    with video_placeholder.container():
        ctx = webrtc_streamer(
            key=st.session_state.webrtc_key,
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration=RTC_CONFIGURATION # --- NEW: Added STUN server
        )
    
    # This loop updates the stats on the screen *while* the component is playing
    while ctx.state.playing:
        with processor.lock:
            rep_placeholder.metric("Reps", processor.rep_count)
            state_placeholder.metric("State", processor.state)
            feedback_placeholder.markdown(f"**Feedback:**<br>{processor.feedback}", unsafe_allow_html=True)
            
            chart_df = processor.chart_data.copy()
            if not chart_df.empty:
                if processor.exercise_type == "Squats":
                    angle_name, form_name = "Knee Angle", "Hip Angle"
                elif processor.exercise_type == "Push-ups":
                    angle_name, form_name = "Elbow Angle", "Back Angle"
                elif processor.exercise_type == "Bicep Curls":
                    angle_name, form_name = "Elbow Angle", "Shoulder Angle"
                else:
                    angle_name, form_name = "Main", "Form"
                
                chart_df = chart_df.rename(columns={'Main Angle': angle_name, 'Form Angle': form_name})
                chart_placeholder.line_chart(chart_df.set_index("Frame"), y=[angle_name, form_name])
        
        time.sleep(0.1) 
        
    # --- *** NEW: 'on_ended' CHECK *** ---
    # This block replaces your old 'if not ctx.state.playing...'
    # It checks the event flag set by the 'on_ended' callback.
    
    if processor.workout_ended.is_set():
        st.info("Workout stopped. Saving data...")
        
        # Get the final data from the processor
        summary = processor.final_summary
        ts_data = processor.final_ts_data
        
        if summary is None or ts_data is None:
             st.error("Error retrieving workout data. Data may not be saved.")
        else:
            # Clean data (must be identical to analysis.py)
            main_angles_raw = ts_data['main_angle']
            form_angles_raw = ts_data['form_angle']
            main_angles_clean = np.array([a for a in main_angles_raw if a is not None and a > 0])
            form_angles_clean = np.array([f for f in form_angles_raw if f is not None and f > 0])
            
            analysis_results = {} 
            if len(main_angles_clean) > 5:
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
        
        # Important: Clear the processor to reset the state
        st.session_state.exercise_processor = None
        st.rerun()

else:
    video_placeholder.info("Click 'Start Workout' in the sidebar to begin!")
