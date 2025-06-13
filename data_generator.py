import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class StudentDataGenerator:
    def __init__(self, n_students=1000, n_courses=50):
        self.n_students = n_students
        self.n_courses = n_courses
        self.course_difficulties = np.random.normal(0.5, 0.15, n_courses)
        self.course_difficulties = np.clip(self.course_difficulties, 0.1, 0.9)
        
        # Define course categories
        self.categories = ['Computer Science', 'Mathematics', 'Physics', 
                          'Engineering', 'Business', 'Arts', 'Social Sciences']
        
        # Generate course data
        self.courses = self._generate_courses()
        
    def _generate_courses(self):
        courses = []
        for i in range(self.n_courses):
            course = {
                'course_id': f'CS{i:03d}',
                'name': f'Course {i+1}',
                'category': np.random.choice(self.categories),
                'difficulty': self.course_difficulties[i],
                'credits': np.random.choice([2, 3, 4]),
                'prerequisites': [] if i < 5 else [f'CS{(i-5):03d}']
            }
            courses.append(course)
        return pd.DataFrame(courses)
    
    def generate_student_data(self):
        # Generate student profiles
        students = []
        for i in range(self.n_students):
            student = {
                'student_id': f'ST{i:04d}',
                'gpa': np.random.normal(3.0, 0.5),
                'interests': np.random.choice(self.categories, size=2, replace=False).tolist(),
                'career_goal': np.random.choice(self.categories),
                'learning_style': np.random.choice(['visual', 'auditory', 'reading', 'kinesthetic']),
                'enrollment_date': (datetime.now() - timedelta(days=np.random.randint(0, 1000))).strftime('%Y-%m-%d')
            }
            students.append(student)
        
        # Generate course history
        course_history = []
        for student in students:
            n_courses_taken = np.random.randint(5, 20)
            courses_taken = np.random.choice(self.courses['course_id'], size=n_courses_taken, replace=False)
            
            for course_id in courses_taken:
                course_difficulty = self.courses[self.courses['course_id'] == course_id]['difficulty'].values[0]
                student_ability = (student['gpa'] - 2.0) / 2.0  # Normalize GPA to 0-1 scale
                
                # Calculate grade based on student ability and course difficulty
                grade_prob = 1 / (1 + np.exp(-5 * (student_ability - course_difficulty)))
                grade = np.random.binomial(1, grade_prob)
                grade = 'A' if grade == 1 else 'B'
                
                course_history.append({
                    'student_id': student['student_id'],
                    'course_id': course_id,
                    'grade': grade,
                    'semester': np.random.randint(1, 9)
                })
        
        return pd.DataFrame(students), pd.DataFrame(course_history)
    
    def save_data(self, students_df, course_history_df):
        students_df.to_csv('data/students.csv', index=False)
        course_history_df.to_csv('data/course_history.csv', index=False)
        self.courses.to_csv('data/courses.csv', index=False)

if __name__ == "__main__":
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate and save data
    generator = StudentDataGenerator()
    students_df, course_history_df = generator.generate_student_data()
    generator.save_data(students_df, course_history_df)
    
    print("Data generation completed. Files saved in 'data' directory:")
    print("- students.csv")
    print("- course_history.csv")
    print("- courses.csv") 