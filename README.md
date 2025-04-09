# Personalized-Mentor-Recommendation-System
A machine learning-based recommendation system that matches CLAT aspirants with suitable mentors based on their preferences, learning styles, and academic goals.
Overview
This system uses content-based filtering with cosine similarity to recommend mentors to CLAT aspirants. It processes multiple attributes including preferred subjects, target colleges, preparation levels, and learning styles to find the most compatible mentors.
Features

Personalized mentor recommendations based on multiple criteria
Feedback collection and analysis to improve recommendations over time
Profile-based matching using subject expertise, teaching style, and academic background
Automatic adjustment of recommendations based on user feedback


Approach:

Data Processing
The system transforms categorical and multi-value data into a format suitable for similarity calculations:

One-hot encoding for categorical features (subjects, colleges, learning styles)
Multi-value field handling via exploding and aggregation
Standardization of numerical features
Feature alignment to ensure compatibility between mentor and aspirant profiles

Recommendation Algorithm
The core recommendation engine uses cosine similarity to calculate the match between aspirants and mentors:

Transform aspirant preferences into a feature vector
Compare with all mentor feature vectors using cosine similarity
Rank mentors by similarity score
Return top 3 recommended mentors

Feedback System
The system incorporates user feedback to improve over time:

Collects ratings and comments from aspirants about mentors
Adjusts similarity scores based on previous feedback
Analyzes patterns to identify successful mentor-aspirant matches
Provides insights on matching effectiveness
