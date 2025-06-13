# Course Recommendation System

This project implements an intelligent course recommendation system that suggests personalized study plans based on student performance, interests, and career goals.

## Features

- Student performance analysis
- Interest-based course recommendations
- Career goal alignment
- Personalized study plans
- Performance prediction

## Project Structure

- `data_generator.py`: Generates simulated student data
- `recommendation_system.py`: Core recommendation algorithms
- `main.py`: Main application entry point
- `requirements.txt`: Project dependencies

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Data Structure

The system uses the following student attributes:
- Academic performance (GPA, course grades)
- Interest areas
- Career goals
- Previous course history
- Learning style preferences

## Algorithms

The recommendation system uses:
- Collaborative filtering
- Content-based filtering
- Reinforcement learning for optimization
- Performance prediction models 