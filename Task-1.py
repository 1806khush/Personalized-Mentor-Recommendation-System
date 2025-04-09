#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def generate_mock_data():
  
    preferred_subjects = ['Legal Reasoning', 'Logical Reasoning', 'English', 'GK', 'Quantitative Techniques']
    target_colleges = ['NLSIU Bangalore', 'NALSAR Hyderabad', 'NUJS Kolkata', 'NLU Delhi', 'NLIU Bhopal']
    preparation_levels = ['Beginner', 'Intermediate', 'Advanced']
    learning_styles = ['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic']

    mentor_data = []
    for i in range(20):  # Creating 20 mentors for the task.
        mentor = {
            'mentor_id': f'M{i+1}',
            'name': f'Mentor {i+1}',
            'strong_subjects': np.random.choice(preferred_subjects, size=np.random.randint(1, 4), replace=False).tolist(),
            'alma_mater': np.random.choice(target_colleges),
            'teaching_experience': np.random.randint(1, 6),  # Years of teaching for this task.
            'teaching_style': np.random.choice(learning_styles),
            'clat_percentile': np.random.randint(90, 100),# random module function.
            'specialization': np.random.choice(preferred_subjects)
        }
        mentor_data.append(mentor)
    
    # Generate aspirant/student data
    aspirant_data = []
    for i in range(10):  # Creating 10 students for demonstration
        aspirant = {
            'aspirant_id': f'A{i+1}',
            'name': f'Aspirant {i+1}',
            'preferred_subjects': np.random.choice(preferred_subjects, size=np.random.randint(1, 4), replace=False).tolist(),
            'target_colleges': np.random.choice(target_colleges, size=np.random.randint(1, 3), replace=False).tolist(),
            'preparation_level': np.random.choice(preparation_levels),
            'learning_style': np.random.choice(learning_styles)
        }
        aspirant_data.append(aspirant)
    
    return pd.DataFrame(mentor_data), pd.DataFrame(aspirant_data)

# 2. Process and transform the data for compatibility with ML algorithms

def prepare_features(mentors_df, aspirants_df):
    
    # Explode strong_subjects to have one row per subject
    mentor_subjects = mentors_df.explode('strong_subjects')[['mentor_id', 'strong_subjects']]
    
    # One-hot encode subjects
    mentor_subjects_dummies = pd.get_dummies(mentor_subjects, columns=['strong_subjects'], prefix='', prefix_sep='')
    
    # Group by mentor_id to get subject vectors
    mentor_subject_matrix = mentor_subjects_dummies.groupby('mentor_id').sum().iloc[:, 1:]
    
    # One-hot encode alma_mater
    mentor_college_dummies = pd.get_dummies(mentors_df[['mentor_id', 'alma_mater']], columns=['alma_mater'], prefix='', prefix_sep='')
    mentor_college_matrix = mentor_college_dummies.groupby('mentor_id').sum().iloc[:, 1:]
    
    # One-hot encode teaching_style
    mentor_style_dummies = pd.get_dummies(mentors_df[['mentor_id', 'teaching_style']], columns=['teaching_style'], prefix='', prefix_sep='')
    mentor_style_matrix = mentor_style_dummies.groupby('mentor_id').sum().iloc[:, 1:]
    
    # Scale numerical features
    scaler = StandardScaler()
    mentor_numerical = pd.DataFrame(
        scaler.fit_transform(mentors_df[['teaching_experience', 'clat_percentile']]),
        columns=['teaching_experience', 'clat_percentile'],
        index=mentors_df['mentor_id']
    )
    
    # Combine all mentor features
    mentor_features = pd.concat([
        mentor_subject_matrix,
        mentor_college_matrix,
        mentor_style_matrix,
        mentor_numerical
    ], axis=1).fillna(0)
    
    # Explode preferred_subjects to have one row per subject
    aspirant_subjects = aspirants_df.explode('preferred_subjects')[['aspirant_id', 'preferred_subjects']]
    
    # One-hot encode subjects
    aspirant_subjects_dummies = pd.get_dummies(aspirant_subjects, columns=['preferred_subjects'], prefix='', prefix_sep='')
    
    # Group by aspirant_id to get subject vectors
    aspirant_subject_matrix = aspirant_subjects_dummies.groupby('aspirant_id').sum().iloc[:, 1:]
    
    # Explode and one-hot encode target_colleges
    aspirant_colleges = aspirants_df.explode('target_colleges')[['aspirant_id', 'target_colleges']]
    aspirant_college_dummies = pd.get_dummies(aspirant_colleges, columns=['target_colleges'], prefix='', prefix_sep='')
    aspirant_college_matrix = aspirant_college_dummies.groupby('aspirant_id').sum().iloc[:, 1:]
    
    # One-hot encode preparation_level
    aspirant_level_dummies = pd.get_dummies(aspirants_df[['aspirant_id', 'preparation_level']], columns=['preparation_level'], prefix='', prefix_sep='')
    aspirant_level_matrix = aspirant_level_dummies.groupby('aspirant_id').sum().iloc[:, 1:]
    
    # One-hot encode learning_style
    aspirant_style_dummies = pd.get_dummies(aspirants_df[['aspirant_id', 'learning_style']], columns=['learning_style'], prefix='', prefix_sep='')
    aspirant_style_matrix = aspirant_style_dummies.groupby('aspirant_id').sum().iloc[:, 1:]
    
    # Combine all aspirant features
    aspirant_features = pd.concat([
        aspirant_subject_matrix,
        aspirant_college_matrix,
        aspirant_level_matrix,
        aspirant_style_matrix
    ], axis=1).fillna(0)
    
    # Ensure both feature matrices have the same columns
    all_columns = sorted(list(set(mentor_features.columns) | set(aspirant_features.columns)))
    
    for col in all_columns:
        if col not in mentor_features.columns:
            mentor_features[col] = 0
        if col not in aspirant_features.columns:
            aspirant_features[col] = 0
    
    mentor_features = mentor_features[all_columns]
    aspirant_features = aspirant_features[all_columns]
    
    return mentor_features, aspirant_features

# 3. Implementation of the recommendation system

def recommend_mentors(mentors_df, aspirants_df, aspirant_id, top_n=3):
    # Prepare feature matrices
    mentor_features, aspirant_features = prepare_features(mentors_df, aspirants_df)
    
    # Get the feature vector for the target aspirant
    if aspirant_id not in aspirant_features.index:
        raise ValueError(f"Aspirant ID {aspirant_id} not found in the data")
    
    aspirant_vector = aspirant_features.loc[aspirant_id].values.reshape(1, -1)
    
    # Calculate cosine similarity between the aspirant and all mentors
    similarities = cosine_similarity(aspirant_vector, mentor_features.values)
    
    # Create a dataframe with mentor IDs and their similarity scores
    similarity_df = pd.DataFrame({
        'mentor_id': mentor_features.index,
        'similarity_score': similarities[0]
    })
    
    # Sort by similarity score in descending order
    top_mentors = similarity_df.sort_values('similarity_score', ascending=False).head(top_n)
    
    # Get the full details of the recommended mentors
    recommended_mentors = mentors_df[mentors_df['mentor_id'].isin(top_mentors['mentor_id'])].copy()
    
    # Add similarity scores to the results
    recommended_mentors = recommended_mentors.merge(top_mentors, on='mentor_id')
    
    return recommended_mentors.sort_values('similarity_score', ascending=False)

# 4. Main function to run the recommendation system

def main():
    print("Generating mock data...")
    mentors_df, aspirants_df = generate_mock_data()
    
    print("\nMentor Data Sample:")
    print(mentors_df.head(2))
    
    print("\nAspirant Data Sample:")
    print(aspirants_df.head(2))
    
    # Demo: Recommend mentors for a specific aspirant
    aspirant_id = 'A1'
    print(f"\nRecommending mentors for Aspirant {aspirant_id}:")
    
    aspirant_info = aspirants_df[aspirants_df['aspirant_id'] == aspirant_id].iloc[0]
    print(f"\nAspirant Details:")
    print(f"Preferred Subjects: {', '.join(aspirant_info['preferred_subjects'])}")
    print(f"Target Colleges: {', '.join(aspirant_info['target_colleges'])}")
    print(f"Preparation Level: {aspirant_info['preparation_level']}")
    print(f"Learning Style: {aspirant_info['learning_style']}")
    
    top_mentors = recommend_mentors(mentors_df, aspirants_df, aspirant_id)
    
    print("\nTop 3 Recommended Mentors:")
    for i, (_, mentor) in enumerate(top_mentors.iterrows(), 1):
        print(f"\nRecommendation #{i}: {mentor['name']} (Score: {mentor['similarity_score']:.4f})")
        print(f"Strong Subjects: {', '.join(mentor['strong_subjects'])}")
        print(f"Alma Mater: {mentor['alma_mater']}")
        print(f"Teaching Experience: {mentor['teaching_experience']} years")
        print(f"Teaching Style: {mentor['teaching_style']}")
        print(f"CLAT Percentile: {mentor['clat_percentile']}%")
        print(f"Specialization: {mentor['specialization']}")


# Fixed code for the feedback system class to address the pandas concatenation warning

class FeedbackSystem:
    def __init__(self, mentors_df, aspirants_df):
        self.mentors_df = mentors_df
        self.aspirants_df = aspirants_df
        
        # Initialize with proper data types for all columns to avoid the warning
        self.feedback_data = pd.DataFrame({
            'aspirant_id': pd.Series(dtype='object'),
            'mentor_id': pd.Series(dtype='object'),
            'rating': pd.Series(dtype='float'),
            'comments': pd.Series(dtype='object'),
            'timestamp': pd.Series(dtype='datetime64[ns]')
        })
    
    def submit_feedback(self, aspirant_id, mentor_id, rating, comments=''):
        """Record feedback from an aspirant about a mentor"""
        import datetime
        
        new_feedback = pd.DataFrame({
            'aspirant_id': [aspirant_id],
            'mentor_id': [mentor_id],
            'rating': [float(rating)],  # Ensure consistent dtype
            'comments': [comments],
            'timestamp': [datetime.datetime.now()]
        })
        
        # Use pd.concat with explicitly defined dtypes to avoid the warning
        self.feedback_data = pd.concat([self.feedback_data, new_feedback], ignore_index=True)
        print(f"Feedback recorded: Aspirant {aspirant_id} rated Mentor {mentor_id} with {rating}/5")
    
    # The rest of the class remains the same
    def adjust_recommendations_with_feedback(self, aspirant_id, top_n=3):
        """Use historical feedback to adjust recommendations"""
        # Get base recommendations
        base_recommendations = recommend_mentors(self.mentors_df, self.aspirants_df, aspirant_id, top_n=top_n+2)
        
        # Check if we have feedback for this aspirant
        aspirant_feedback = self.feedback_data[self.feedback_data['aspirant_id'] == aspirant_id]
        
        if len(aspirant_feedback) == 0:
            # No feedback yet, return base recommendations
            return base_recommendations.head(top_n)
        
        # Calculate average ratings for mentors
        mentor_ratings = aspirant_feedback.groupby('mentor_id')['rating'].mean().reset_index()
        
        for idx, row in base_recommendations.iterrows():
            mentor_id = row['mentor_id']
            mentor_feedback = mentor_ratings[mentor_ratings['mentor_id'] == mentor_id]
            
            if len(mentor_feedback) > 0:
                # Adjust score based on rating (positive if rating > 3, negative otherwise)
                rating = mentor_feedback.iloc[0]['rating']
                adjustment = (rating - 3) / 10  # Small adjustment based on rating
                base_recommendations.at[idx, 'similarity_score'] += adjustment
        

        return base_recommendations.sort_values('similarity_score', ascending=False).head(top_n)
    
    def analyze_feedback_patterns(self):
        """Analyze feedback to identify patterns and areas for improvement"""
        if len(self.feedback_data) < 5:
            return "Not enough feedback data for meaningful analysis."
        
        # Average rating per mentor
        mentor_avg_ratings = self.feedback_data.groupby('mentor_id')['rating'].agg(['mean', 'count']).reset_index()
        top_mentors = mentor_avg_ratings.sort_values('mean', ascending=False).head(3)
        bottom_mentors = mentor_avg_ratings[mentor_avg_ratings['count'] >= 3].sort_values('mean').head(3)
        
        # Identify which mentors are doing well with which types of aspirants
        enhanced_feedback = self.feedback_data.merge(
            self.aspirants_df, on='aspirant_id'
        ).merge(
            self.mentors_df, on='mentor_id', suffixes=('_aspirant', '_mentor')
        )
        
        # Analyze which learning styles match well
        style_match = enhanced_feedback[enhanced_feedback['learning_style'] == enhanced_feedback['teaching_style']]
        style_mismatch = enhanced_feedback[enhanced_feedback['learning_style'] != enhanced_feedback['teaching_style']]
        
        style_match_avg = style_match['rating'].mean() if len(style_match) > 0 else 0
        style_mismatch_avg = style_mismatch['rating'].mean() if len(style_mismatch) > 0 else 0
        
        # Output analysis
        analysis = f"Feedback Analysis:\n"
        analysis += f"Total feedback entries: {len(self.feedback_data)}\n\n"
        
        analysis += "Top-rated mentors:\n"
        for _, mentor in top_mentors.iterrows():
            analysis += f"- Mentor {mentor['mentor_id']}: {mentor['mean']:.2f}/5 ({mentor['count']} ratings)\n"
        
        if len(bottom_mentors) > 0:
            analysis += "\nMentors needing improvement:\n"
            for _, mentor in bottom_mentors.iterrows():
                analysis += f"- Mentor {mentor['mentor_id']}: {mentor['mean']:.2f}/5 ({mentor['count']} ratings)\n"
        
        analysis += f"\nLearning style matching impact:\n"
        analysis += f"- When styles match: {style_match_avg:.2f}/5 avg rating\n"
        analysis += f"- When styles don't match: {style_mismatch_avg:.2f}/5 avg rating\n"
        
        return analysis
# Run the demo if executed as a script
if __name__ == "__main__":
    main()
   
    print("\n--- BONUS: Feedback System Demonstration ---")
    mentors_df, aspirants_df = generate_mock_data()
    feedback_system = FeedbackSystem(mentors_df, aspirants_df)
    
    # Simulate some feedback
    feedback_system.submit_feedback('A1', 'M3', 5, "Great mentor, very helpful!")
    feedback_system.submit_feedback('A1', 'M7', 2, "Not very knowledgeable about my preferred subjects")
    feedback_system.submit_feedback('A2', 'M3', 4, "Excellent teaching style")
    feedback_system.submit_feedback('A3', 'M12', 5, "Perfect match for my learning style")
    feedback_system.submit_feedback('A2', 'M8', 3, "Average experience")
    
    # Get recommendations with feedback adjustments
    print("\nAdjusted recommendations for A1 based on feedback:")
    adjusted_recommendations = feedback_system.adjust_recommendations_with_feedback('A1')
    
    for i, (_, mentor) in enumerate(adjusted_recommendations.iterrows(), 1):
        print(f"\nAdjusted Recommendation #{i}: {mentor['name']} (Score: {mentor['similarity_score']:.4f})")
        print(f"Strong Subjects: {', '.join(mentor['strong_subjects'])}")
        print(f"Teaching Style: {mentor['teaching_style']}")
    
    # Show analysis of the feedback patterns
    print("\nFeedback Analysis:")
    print(feedback_system.analyze_feedback_patterns())
    
    print("\n--- Future Improvement Suggestions ---")
    print("""
1. Dynamic Feature Weighting: Automatically adjust feature weights based on aspirant performance and feedback.
2. Time-decay Factor: Give more weight to recent feedback over older ones.

    """)


# In[ ]:





# In[ ]:




