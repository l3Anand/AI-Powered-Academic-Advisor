import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model

class CourseRecommender:
    def __init__(self):
        self.students_df = None
        self.courses_df = None
        self.course_history_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.student_id_to_idx = None
        self.idx_to_student_id = None
        self.course_id_to_idx = None
        self.idx_to_course_id = None
        
    def load_data(self):
        """Load the generated data"""
        self.students_df = pd.read_csv('data/students.csv')
        self.courses_df = pd.read_csv('data/courses.csv')
        self.course_history_df = pd.read_csv('data/course_history.csv')
        
    def _create_student_features(self, student_id):
        """Create feature vector for a student"""
        student = self.students_df[self.students_df['student_id'] == student_id].iloc[0]
        history = self.course_history_df[self.course_history_df['student_id'] == student_id]
        
        # Basic features
        features = {
            'gpa': student['gpa'],
            'semester': history['semester'].max() if not history.empty else 1
        }
        
        # Add category preferences based on course history
        for category in self.courses_df['category'].unique():
            category_courses = self.courses_df[self.courses_df['category'] == category]['course_id']
            category_history = history[history['course_id'].isin(category_courses)]
            features[f'category_{category}'] = len(category_history) / len(self.courses_df)
            
        return pd.Series(features)
    
    def _create_course_features(self, course_id):
        """Create feature vector for a course"""
        course = self.courses_df[self.courses_df['course_id'] == course_id].iloc[0]
        
        features = {
            'difficulty': course['difficulty'],
            'credits': course['credits']
        }
        
        # Add category features
        for category in self.courses_df['category'].unique():
            features[f'category_{category}'] = 1 if course['category'] == category else 0
            
        return pd.Series(features)
    
    def train_model(self):
        """Train the recommendation model using collaborative filtering"""
        # Map student_id and course_id to integer indices
        unique_student_ids = self.students_df['student_id'].unique()
        unique_course_ids = self.courses_df['course_id'].unique()
        self.student_id_to_idx = {sid: i for i, sid in enumerate(unique_student_ids)}
        self.idx_to_student_id = {i: sid for i, sid in enumerate(unique_student_ids)}
        self.course_id_to_idx = {cid: i for i, cid in enumerate(unique_course_ids)}
        self.idx_to_course_id = {i: cid for i, cid in enumerate(unique_course_ids)}

        # First convert grades to numerical values
        grade_map = {'A': 1.0, 'B': 0.5}
        self.course_history_df['grade_value'] = self.course_history_df['grade'].map(grade_map)
        
        user_item_matrix = pd.pivot_table(
            self.course_history_df,
            values='grade_value',
            index='student_id',
            columns='course_id',
            fill_value=0
        )
        
        n_users = len(user_item_matrix)
        n_items = len(user_item_matrix.columns)
        embedding_dim = 32
        
        # User input
        user_input = layers.Input(shape=(1,), name='user_input')
        user_embedding = layers.Embedding(n_users, embedding_dim, name='user_embedding')(user_input)
        user_vec = layers.Flatten(name='user_flatten')(user_embedding)
        
        # Item input
        item_input = layers.Input(shape=(1,), name='item_input')
        item_embedding = layers.Embedding(n_items, embedding_dim, name='item_embedding')(item_input)
        item_vec = layers.Flatten(name='item_flatten')(item_embedding)
        
        # Merge layers
        concat = layers.Concatenate()([user_vec, item_vec])
        dense1 = layers.Dense(64, activation='relu')(concat)
        dense2 = layers.Dense(32, activation='relu')(dense1)
        output = layers.Dense(1, activation='sigmoid')(dense2)
        
        # Create model
        self.model = Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Prepare training data
        user_ids = []
        item_ids = []
        ratings = []
        
        for idx, row in user_item_matrix.iterrows():
            for col in user_item_matrix.columns:
                if row[col] > 0:
                    user_ids.append(self.student_id_to_idx[idx])
                    item_ids.append(self.course_id_to_idx[col])
                    ratings.append(row[col])
        
        # Train the model
        self.model.fit(
            [np.array(user_ids), np.array(item_ids)],
            np.array(ratings),
            epochs=10,
            batch_size=64,
            validation_split=0.2
        )
    
    def get_recommendations(self, student_id, n_recommendations=5):
        """Get course recommendations for a student"""
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        if self.student_id_to_idx is None or self.course_id_to_idx is None:
            raise ValueError("ID mappings not initialized. Train the model first.")
        
        # Get student's course history
        student_history = self.course_history_df[
            self.course_history_df['student_id'] == student_id
        ]['course_id'].tolist()
        
        # Get all courses not taken by the student
        available_courses = self.courses_df[
            ~self.courses_df['course_id'].isin(student_history)
        ]['course_id'].tolist()
        
        # Predict ratings for available courses
        predictions = []
        student_idx = self.student_id_to_idx[student_id]
        for course_id in available_courses:
            course_idx = self.course_id_to_idx[course_id]
            pred = self.model.predict([
                np.array([student_idx]),
                np.array([course_idx])
            ], verbose=0)
            predictions.append((course_id, pred[0][0]))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_courses = [course_id for course_id, _ in predictions[:n_recommendations]]
        
        # Get course details
        recommendations = self.courses_df[
            self.courses_df['course_id'].isin(recommended_courses)
        ]
        
        return recommendations
    
    def get_personalized_study_plan(self, student_id, target_career):
        """Generate a personalized study plan based on career goals"""
        student = self.students_df[self.students_df['student_id'] == student_id].iloc[0]
        current_semester = self.course_history_df[
            self.course_history_df['student_id'] == student_id
        ]['semester'].max()
        
        # Get courses relevant to target career
        career_courses = self.courses_df[
            self.courses_df['category'] == target_career
        ]
        
        # Sort courses by difficulty
        career_courses = career_courses.sort_values('difficulty')
        
        # Create study plan
        study_plan = []
        remaining_semesters = 8 - current_semester
        
        for semester in range(current_semester + 1, current_semester + remaining_semesters + 1):
            semester_courses = career_courses.head(2)  # Take 2 courses per semester
            study_plan.append({
                'semester': semester,
                'courses': semester_courses[['course_id', 'name', 'credits']].to_dict('records')
            })
            career_courses = career_courses.iloc[2:]  # Remove selected courses
            
        return study_plan 