"""
Enhanced Streamlit Web Application for Music Wellbeing Recommender
Integrates the trained RandomForest model for real recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os
from typing import Dict, Union, List

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# Page configuration
st.set_page_config(
    page_title="🎵 Music Wellbeing Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    """Load the trained model and related artifacts."""
    try:
        # Get the absolute path to models directory
        # Works both locally and on Streamlit Cloud
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up from app/ to music-wellbeing-recommender/
        models_dir = os.path.join(project_root, 'models')
        
        model_path = os.path.join(models_dir, 'music_effect_model.pkl')
        encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        features_path = os.path.join(models_dir, 'feature_columns.pkl')
        
        # Debug: show paths being used
        st.sidebar.info(f"Loading models from: {models_dir}")
        
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        feature_columns = joblib.load(features_path)
        
        return {
            'model': model,
            'label_encoder': label_encoder,
            'feature_columns': feature_columns,
            'loaded': True
        }
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error(f"Current directory: {os.path.dirname(os.path.abspath(__file__))}")
        st.error(f"Looking for models in: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')}")
        return {'loaded': False}

def recommend_music_genres(
    user_profile: Dict[str, Union[int, float, bool]], 
    target_to_improve: str,
    model_artifacts: dict,
    top_n: int = 5,
    use_target_optimization: bool = True
) -> pd.DataFrame:
    """
    Recommend music genres using target-optimized algorithm for better personalization.
    """
    if not model_artifacts['loaded']:
        return pd.DataFrame()
    
    model = model_artifacts['model']
    label_encoder = model_artifacts['label_encoder']
    feature_columns = model_artifacts['feature_columns']
    
    if use_target_optimization:
        return target_optimized_recommend(
            user_profile=user_profile,
            target_to_improve=target_to_improve,
            model=model,
            feature_columns=feature_columns,
            label_encoder=label_encoder,
            top_n=top_n
        )
    else:
        # Fallback to standard recommendation if needed
        return standard_recommend(
            user_profile=user_profile,
            target_to_improve=target_to_improve,
            model=model,
            feature_columns=feature_columns,
            label_encoder=label_encoder,  
            top_n=top_n
        )

def target_optimized_recommend(
    user_profile: Dict[str, Union[int, float, bool]], 
    target_to_improve: str,
    model, 
    feature_columns: List[str],
    label_encoder,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Enhanced recommendation system that optimizes specifically for the target condition.
    
    This version:
    1. Simulates different improvement scenarios for the target condition
    2. Tests how each genre affects the likelihood of improvement when the target is reduced
    3. Measures the 'therapeutic potential' of each genre for the specific condition
    4. Provides target-specific optimization rather than general music effects
    """
    
    # Validate target
    if target_to_improve not in user_profile:
        return pd.DataFrame()
    
    current_target_level = user_profile[target_to_improve]
    
    # Extract available genres
    genre_features = [col for col in feature_columns if col.startswith('Fav genre_')]
    available_genres = [genre.replace('Fav genre_', '') for genre in genre_features]
    
    if not available_genres:
        return pd.DataFrame()
    
    recommendations = []
    
    # Baseline: Current state prediction
    baseline_profile = pd.Series(user_profile).reindex(feature_columns, fill_value=False)
    for col in baseline_profile.index:
        if isinstance(baseline_profile[col], (bool, np.bool_)):
            baseline_profile[col] = int(baseline_profile[col])
        elif pd.isna(baseline_profile[col]):
            baseline_profile[col] = 0
    
    baseline_probs = model.predict_proba(baseline_profile.values.reshape(1, -1))[0]
    baseline_improve_prob = baseline_probs[np.where(label_encoder.classes_ == 'Improve')[0][0]]
    
    # Test each genre with target condition improvement simulation
    for genre in available_genres:
        genre_scores = []
        
        # Test multiple improvement scenarios
        improvement_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # Reduce target by 10%-50%
        
        for improvement_factor in improvement_levels:
            # Create improved profile (reduce target condition)
            improved_profile = pd.Series(user_profile).copy()
            
            # Simulate improvement in target condition
            original_level = improved_profile[target_to_improve]
            improved_level = max(0, original_level * (1 - improvement_factor))
            improved_profile[target_to_improve] = improved_level
            
            # Set genre preference
            for genre_feature in genre_features:
                improved_profile[genre_feature] = False
            improved_profile[f'Fav genre_{genre}'] = True
            
            # Prepare for prediction
            test_profile = improved_profile.reindex(feature_columns, fill_value=False)
            for col in test_profile.index:
                if isinstance(test_profile[col], (bool, np.bool_)):
                    test_profile[col] = int(test_profile[col])
                elif pd.isna(test_profile[col]):
                    test_profile[col] = 0
            
            # Get prediction
            probs = model.predict_proba(test_profile.values.reshape(1, -1))[0]
            improve_prob = probs[np.where(label_encoder.classes_ == 'Improve')[0][0]]
            
            # Calculate improvement potential (difference from baseline)
            improvement_potential = improve_prob - baseline_improve_prob
            genre_scores.append({
                'improvement_factor': improvement_factor,
                'target_level': improved_level,
                'improve_prob': improve_prob,
                'improvement_potential': improvement_potential
            })
        
        # Calculate average improvement potential across scenarios
        avg_improvement_potential = np.mean([score['improvement_potential'] for score in genre_scores])
        max_improvement_potential = max([score['improvement_potential'] for score in genre_scores])
        
        # Calculate target-specific therapeutic score
        therapeutic_score = (avg_improvement_potential + max_improvement_potential) / 2
        
        recommendations.append({
            'Genre': genre,
            'Improvement_Probability': baseline_improve_prob + therapeutic_score,
            'Improvement_Percentage': (baseline_improve_prob + therapeutic_score) * 100,
            'Therapeutic_Score': therapeutic_score,
            'Predicted_Effect': 'Improve' if therapeutic_score > 0 else 'No effect',
            'Target_Optimized': target_to_improve
        })
    
    # Convert to DataFrame and sort by therapeutic score
    recommendations_df = pd.DataFrame(recommendations)
    if recommendations_df.empty:
        return pd.DataFrame()
    
    recommendations_df = recommendations_df.sort_values('Therapeutic_Score', ascending=False)
    return recommendations_df.head(top_n)

def standard_recommend(
    user_profile: Dict[str, Union[int, float, bool]], 
    target_to_improve: str,
    model, 
    feature_columns: List[str],
    label_encoder,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Standard recommendation system (fallback method).
    """
    # Extract available genres
    genre_features = [col for col in feature_columns if col.startswith('Fav genre_')]
    available_genres = [genre.replace('Fav genre_', '') for genre in genre_features]
    
    if not available_genres:
        return pd.DataFrame()
    
    # Create base profile
    base_profile = pd.Series(user_profile)
    recommendations = []
    
    # Test each genre
    for genre in available_genres:
        test_profile = base_profile.copy()
        
        # Reset all genre flags
        for genre_feature in genre_features:
            test_profile[genre_feature] = False
        
        # Set current genre
        test_profile[f'Fav genre_{genre}'] = True
        
        # Reindex and prepare for prediction
        test_profile = test_profile.reindex(feature_columns, fill_value=False)
        
        # Fixed type conversion - handle boolean and NaN values properly
        for col in test_profile.index:
            if isinstance(test_profile[col], (bool, np.bool_)):
                test_profile[col] = int(test_profile[col])
            elif pd.isna(test_profile[col]):
                test_profile[col] = 0
        
        # Get prediction
        test_array = test_profile.values.reshape(1, -1)
        probabilities = model.predict_proba(test_array)[0]
        
        # Find "Improve" probability
        improve_idx = np.where(label_encoder.classes_ == 'Improve')[0]
        if len(improve_idx) > 0:
            improve_prob = probabilities[improve_idx[0]]
            recommendations.append({
                'Genre': genre,
                'Improvement_Probability': improve_prob,
                'Improvement_Percentage': improve_prob * 100,
                'Predicted_Effect': label_encoder.classes_[np.argmax(probabilities)]
            })
    
    # Convert to DataFrame and sort
    recommendations_df = pd.DataFrame(recommendations)
    if recommendations_df.empty:
        return pd.DataFrame()
    
    recommendations_df = recommendations_df.sort_values('Improvement_Probability', ascending=False)
    return recommendations_df.head(top_n)

def display_header():
    """Display the main header and introduction."""
    st.markdown('<h1 class="main-header">🎵 Music Wellbeing Recommender</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🎯 AI-Powered Music Recommendations for Mental Wellbeing</strong><br>
        Get personalized music genre recommendations based on your mental health profile. 
        Our machine learning model analyzes your anxiety, depression, insomnia, and OCD levels 
        to suggest music genres that are most likely to improve your wellbeing.
    </div>
    """, unsafe_allow_html=True)

def get_user_input():
    """Get user input from the sidebar."""
    st.sidebar.header("🧠 Your Profile")
    
    with st.sidebar:
        st.markdown("### 👤 Personal Information")
        age = st.slider("Age", 18, 80, 25, help="Your current age")
        hours_per_day = st.slider("Daily Music Listening (hours)", 0.0, 12.0, 3.0, 0.5,
                                help="How many hours do you listen to music per day?")
        
        st.markdown("### 🧠 Mental Health Indicators")
        st.caption("Rate each on a scale of 0-10 (0=None, 10=Severe)")
        
        anxiety = st.slider("😰 Anxiety Level", 0, 10, 5,
                           help="How would you rate your current anxiety level?")
        depression = st.slider("😔 Depression Level", 0, 10, 5,
                              help="How would you rate your current depression level?")
        insomnia = st.slider("😴 Insomnia Level", 0, 10, 5,
                            help="How severe are your sleep difficulties?")
        ocd = st.slider("🔄 OCD Level", 0, 10, 5,
                       help="How would you rate your OCD symptoms?")
        
        st.markdown("### 🎵 Music Preferences")
        bpm = st.slider("Preferred BPM (Beats Per Minute)", 40, 200, 120,
                       help="What tempo do you prefer? (60-90: Slow, 90-120: Moderate, 120+: Fast)")
        
        primary_streaming = st.selectbox("Primary Streaming Service", [
            "Spotify", "Apple Music", "YouTube Music", "Amazon Music", 
            "Pandora", "Deezer", "SoundCloud", "Other"
        ])
        
        st.markdown("### 🎧 Listening Habits")
        while_working = st.selectbox("Do you listen to music while working?", ["Yes", "No"])
        instrumentalist = st.selectbox("Do you play any musical instruments?", ["No", "Yes"])
        composer = st.selectbox("Do you compose music?", ["No", "Yes"])
        exploratory = st.selectbox("Do you actively explore new music?", ["Yes", "No"])
        foreign_languages = st.selectbox("Do you listen to music in foreign languages?", ["Yes", "No"])
        
        # Target to improve
        st.markdown("### 🎯 Primary Focus")
        target_to_improve = st.selectbox(
            "Which mental health aspect would you most like to improve?",
            ["Anxiety", "Depression", "Insomnia", "OCD"],
            help="Select your primary concern for personalized recommendations"
        )
        
        # Recommendation mode
        st.markdown("### ⚙️ Recommendation Mode")
        use_target_optimization = st.selectbox(
            "Choose recommendation approach:",
            ["Target-Optimized (Recommended)", "Standard"],
            help="Target-Optimized uses advanced algorithms specifically designed for your chosen condition"
        )
    
    # Create user profile dictionary
    user_profile = {
        'Age': age,
        'Hours per day': hours_per_day,
        'BPM': bpm,
        'Anxiety': anxiety,
        'Depression': depression,
        'Insomnia': insomnia,
        'OCD': ocd,
        f'Primary streaming service_{primary_streaming}': True,
        f'While working_{while_working}': True,
        f'Instrumentalist_{instrumentalist}': True,
        f'Composer_{composer}': True,
        f'Exploratory_{exploratory}': True,
        f'Foreign languages_{foreign_languages}': True
    }
    
    return user_profile, target_to_improve, use_target_optimization

def display_user_profile(user_profile, target_to_improve):
    """Display user profile summary."""
    st.markdown("### 👤 Your Profile Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", f"{user_profile['Age']} years")
        st.metric("Daily Listening", f"{user_profile['Hours per day']} hours")
    
    with col2:
        st.metric("Anxiety", f"{user_profile['Anxiety']}/10")
        st.metric("Depression", f"{user_profile['Depression']}/10")
    
    with col3:
        st.metric("Insomnia", f"{user_profile['Insomnia']}/10")
        st.metric("OCD", f"{user_profile['OCD']}/10")
    
    with col4:
        st.metric("Preferred BPM", f"{user_profile['BPM']}")
        st.metric("Focus Area", target_to_improve)

def display_recommendations(recommendations_df, target_to_improve):
    """Display the target-optimized music recommendations."""
    if recommendations_df.empty:
        st.error("❌ No recommendations could be generated. Please try different inputs.")
        return
    
    st.markdown("### 🎵 Your Target-Optimized Music Recommendations")
    st.markdown(f"**Focus:** Specifically improving {target_to_improve}")
    
    # Display top recommendation prominently
    top_rec = recommendations_df.iloc[0]
    
    # Check if we have therapeutic score (target optimized) or standard recommendation
    if 'Therapeutic_Score' in top_rec:
        therapeutic_pct = top_rec['Therapeutic_Score'] * 100
        status_emoji = "🟢" if therapeutic_pct > 5 else "🟡" if therapeutic_pct > 1 else "🔴"
        status_text = "High" if therapeutic_pct > 5 else "Moderate" if therapeutic_pct > 1 else "Low"
        
        st.markdown(f"""
        <div class="success-box">
            <h3>{status_emoji} Top Recommendation: {top_rec['Genre']}</h3>
            <p><strong>{status_text} therapeutic potential for {target_to_improve}</strong></p>
            <p>Therapeutic Score: <strong>{therapeutic_pct:+.2f}%</strong></p>
            <p>Specifically optimized for <strong>{target_to_improve} improvement</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback for standard recommendations
        st.markdown(f"""
        <div class="success-box">
            <h3>🏆 Top Recommendation: {top_rec['Genre']}</h3>
            <p><strong>{top_rec['Improvement_Percentage']:.1f}% chance to improve your {target_to_improve.lower()}</strong></p>
            <p>Predicted effect: <strong>{top_rec['Predicted_Effect']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display all recommendations in a table
    st.markdown("#### 📊 Target-Optimized Recommendations")
    
    # Format the dataframe for display
    display_df = recommendations_df.copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    
    if 'Therapeutic_Score' in display_df.columns:
        # Target-optimized display
        display_df['Therapeutic Score'] = (display_df['Therapeutic_Score'] * 100).round(2).astype(str) + '%'
        display_df['Improvement %'] = display_df['Improvement_Percentage'].round(1).astype(str) + '%'
        display_df = display_df[['Rank', 'Genre', 'Therapeutic Score', 'Improvement %', 'Target_Optimized']]
        display_df.columns = ['Rank', 'Genre', f'{target_to_improve} Therapeutic Score', 'Overall Improvement %', 'Optimized For']
    else:
        # Standard display
        display_df['Improvement %'] = display_df['Improvement_Percentage'].round(1).astype(str) + '%'
        display_df = display_df[['Rank', 'Genre', 'Improvement %', 'Predicted_Effect']]
        display_df.columns = ['Rank', 'Genre', 'Improvement Probability', 'Predicted Effect']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Create visualization
    if 'Therapeutic_Score' in recommendations_df.columns:
        fig = px.bar(
            recommendations_df, 
            x='Genre', 
            y=[col for col in ['Therapeutic_Score'] if col in recommendations_df.columns][0],
            title=f"Target-Optimized Therapeutic Scores for {target_to_improve}",
            labels={'Therapeutic_Score': f'{target_to_improve} Therapeutic Score', 'Genre': 'Music Genre'},
            color='Therapeutic_Score',
            color_continuous_scale='RdYlGn'
        )
        # Update y-axis to show percentage
        fig.update_traces(y=recommendations_df['Therapeutic_Score'] * 100)
        fig.update_layout(yaxis_title=f"{target_to_improve} Therapeutic Score (%)")
    else:
        fig = px.bar(
            recommendations_df, 
            x='Genre', 
            y='Improvement_Percentage',
            title=f"Music Genre Recommendations for {target_to_improve} Improvement",
            labels={'Improvement_Percentage': 'Improvement Probability (%)', 'Genre': 'Music Genre'},
            color='Improvement_Percentage',
            color_continuous_scale='viridis'
        )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_genre_explanations(recommendations_df):
    """Display explanations for recommended genres."""
    st.markdown("### 💡 Why These Genres?")
    
    explanations = {
        'Classical': "Classical music's structured harmonies and moderate tempo have been scientifically shown to reduce cortisol levels and promote relaxation.",
        'Jazz': "Jazz music's complex yet soothing harmonies can provide emotional stimulation while maintaining therapeutic benefits.",
        'Ambient': "Ambient music's minimal structure and gentle soundscapes help reduce mental stimulation, making it ideal for anxiety relief.",
        'Folk': "Folk music's storytelling nature and acoustic instruments can provide emotional connection and comfort.",
        'New Age': "New Age music is specifically designed for relaxation and meditation, often incorporating nature sounds and slow tempos.",
        'World': "World music offers diverse cultural expressions that can provide new perspectives and emotional experiences.",
        'Alternative': "Alternative music's authentic expression can resonate with complex emotional states and provide cathartic relief.",
        'Rock': "Certain rock subgenres with moderate tempo can provide energizing effects while maintaining emotional authenticity.",
        'Pop': "Upbeat pop music can boost mood through familiar melodies and positive energy.",
        'Hip hop': "Hip hop's rhythmic elements and expressive lyrics can provide emotional outlet and empowerment.",
        'R&B': "R&B's smooth rhythms and soulful melodies can provide comfort and emotional connection.",
        'Country': "Country music's storytelling and emotional honesty can provide relatability and comfort.",
        'EDM': "Electronic dance music's rhythmic patterns can provide energy and mood elevation through movement.",
        'Latin': "Latin music's vibrant rhythms can promote physical movement and positive energy.",
        'Metal': "For some individuals, metal music's intensity can provide emotional release and catharsis.",
        'Lofi': "Lo-fi music's relaxed tempo and ambient qualities make it ideal for focus and anxiety reduction."
    }
    
    for _, row in recommendations_df.iterrows():
        genre = row['Genre']
        percentage = row['Improvement_Percentage']
        
        explanation = explanations.get(genre, f"{genre} music has shown positive therapeutic effects for mood regulation and stress reduction.")
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎼 {genre} ({percentage:.1f}% improvement chance)</h4>
            <p>{explanation}</p>
        </div>
        """, unsafe_allow_html=True)

def display_mental_health_radar(user_profile):
    """Display a radar chart of mental health indicators."""
    st.markdown("### 📊 Your Mental Health Profile")
    
    categories = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
    values = [user_profile[cat] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Profile',
        line_color='rgb(30, 136, 229)',
        fillcolor='rgba(30, 136, 229, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        title="Mental Health Indicators (0-10 scale)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    # Load model artifacts
    model_artifacts = load_model_artifacts()
    
    # Display header
    display_header()
    
    # Check if model loaded successfully
    if not model_artifacts['loaded']:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Model Not Available</strong><br>
            The trained machine learning model could not be loaded. 
            Please ensure you have run the modeling notebook and the model files exist in the /models/ directory.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get user input
    user_profile, target_to_improve, use_target_optimization = get_user_input()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display user profile
        display_user_profile(user_profile, target_to_improve)
        
        # Get recommendations button
        optimization_mode = use_target_optimization == "Target-Optimized (Recommended)"
        button_text = f"🎵 Get My {'Target-Optimized' if optimization_mode else 'Standard'} Recommendations"
        
        if st.button(button_text, type="primary", use_container_width=True):
            with st.spinner(f"🤖 {'Target-optimizing' if optimization_mode else 'Analyzing'} your profile and generating recommendations..."):
                recommendations = recommend_music_genres(
                    user_profile, 
                    target_to_improve, 
                    model_artifacts, 
                    top_n=5,
                    use_target_optimization=optimization_mode
                )
                
                if not recommendations.empty:
                    display_recommendations(recommendations, target_to_improve)
                    display_genre_explanations(recommendations)
                else:
                    st.error("Unable to generate recommendations. Please check your inputs.")
    
    with col2:
        # Display mental health radar chart
        display_mental_health_radar(user_profile)
        
        # Model information
        st.markdown("""
        <div class="info-box">
            <h4>🤖 About Our AI System</h4>
            <p><strong>Core Model:</strong> Random Forest Classifier</p>
            <p><strong>Optimization:</strong> Target-specific therapeutic scoring</p>
            <p><strong>Features:</strong> 200+ user characteristics</p>
            <p><strong>Training Data:</strong> MXMH Survey (736 responses)</p>
            <p><strong>Innovation:</strong> Simulates improvement scenarios for personalized recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Important Disclaimer</h4>
            <p>This tool provides music recommendations based on data patterns and should not replace professional mental health treatment. 
            If you're experiencing severe mental health symptoms, please consult with a healthcare professional.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()