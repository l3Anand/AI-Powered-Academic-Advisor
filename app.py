import streamlit as st
import pandas as pd
import os
from groq import Groq
from recommendation_system import CourseRecommender

# Load data
@st.cache_data
def load_data():
    students_df = pd.read_csv('data/students.csv')
    courses_df = pd.read_csv('data/courses.csv')
    course_history_df = pd.read_csv('data/course_history.csv')
    return students_df, courses_df, course_history_df

# Initialize the recommender
@st.cache_resource
def init_recommender():
    recommender = CourseRecommender()
    recommender.load_data()
    recommender.train_model()
    return recommender

# Initialize Groq client with API key
client = Groq(api_key="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")

# Main app
def main():
    st.title("Course Recommendation System")
    st.write("Select a student ID to view recommendations and study plan.")

    # Load data
    students_df, courses_df, course_history_df = load_data()
    recommender = init_recommender()

    # Dropdown for student selection
    student_ids = students_df['student_id'].tolist()
    selected_student_id = st.selectbox("Select Student ID:", student_ids)

    if st.button("Get Recommendations"):
        # Get recommendations
        recommendations = recommender.get_recommendations(selected_student_id, n_recommendations=5)
        st.subheader("Recommended Courses")
        st.dataframe(recommendations[['course_id', 'name', 'category', 'difficulty']])

        # Get study plan
        student = students_df[students_df['student_id'] == selected_student_id].iloc[0]
        target_career = student['career_goal']
        st.subheader(f"Career Goal: {target_career}")
        study_plan = recommender.get_personalized_study_plan(selected_student_id, target_career)
        for semester in study_plan:
            st.write(f"Semester {semester['semester']}:")
            for course in semester['courses']:
                st.write(f"  - {course['course_id']}: {course['name']} ({course['credits']} credits)")

        # Generate detailed study plan explanation using Groq
        st.subheader("Personalized Study Plan")
        prompt = f"Explain the importance of the following study plan for a student aiming for a career in {target_career}: {study_plan}"
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        st.write(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main() 