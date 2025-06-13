import os
from data_generator import StudentDataGenerator
from recommendation_system import CourseRecommender
import pandas as pd

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate sample data
    print("Generating sample data...")
    generator = StudentDataGenerator(n_students=1000, n_courses=50)
    students_df, course_history_df = generator.generate_student_data()
    generator.save_data(students_df, course_history_df)
    
    # Initialize and train the recommendation system
    print("\nInitializing recommendation system...")
    recommender = CourseRecommender()
    recommender.load_data()
    
    print("Training recommendation model...")
    recommender.train_model()
    
    # Example: Get recommendations for a sample student
    sample_student_id = students_df['student_id'].iloc[0]
    print(f"\nGetting recommendations for student {sample_student_id}...")
    
    # Get course recommendations
    recommendations = recommender.get_recommendations(sample_student_id, n_recommendations=5)
    print("\nRecommended courses:")
    print(recommendations[['course_id', 'name', 'category', 'difficulty']].to_string())
    
    # Get personalized study plan
    student = students_df[students_df['student_id'] == sample_student_id].iloc[0]
    target_career = student['career_goal']
    print(f"\nGenerating study plan for career goal: {target_career}")
    
    study_plan = recommender.get_personalized_study_plan(sample_student_id, target_career)
    print("\nPersonalized Study Plan:")
    for semester in study_plan:
        print(f"\nSemester {semester['semester']}:")
        for course in semester['courses']:
            print(f"  - {course['course_id']}: {course['name']} ({course['credits']} credits)")

if __name__ == "__main__":
    main() 