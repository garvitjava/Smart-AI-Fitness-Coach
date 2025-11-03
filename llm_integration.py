import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_groq_client():
    """Initializes and returns the Groq client."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found in .env file. Please add it.")
            return None
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Groq client. Error: {e}")
        return None

def get_workout_advice(client, summary):
    """Gets post-workout advice from Groq LLM."""
    if not client:
        return "LLM client not available."

    prompt = f"""
    Act as an expert AI fitness coach. My user just completed a workout.
    Workout Details:
    - Exercise: {summary['Exercise']}
    - Repetitions: {summary['Reps']}

    Based on this {summary['Exercise']} workout, please provide:
    1.  A brief, encouraging comment on their workout (1 sentence).
    2.  A specific nutrition tip that aids recovery from this exercise.
    3.  A recommendation for a complementary exercise or stretch.

    Keep the response concise, friendly, and formatted in Markdown.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error getting advice from LLM: {e}"