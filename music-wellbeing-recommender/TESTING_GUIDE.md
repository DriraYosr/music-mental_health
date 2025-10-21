# 🧪 Testing Guide: Music Wellbeing Recommender Streamlit App

## 📋 Prerequisites

Before testing the Streamlit app, ensure you have:

1. **Trained Model**: The `03_modeling.ipynb` notebook has been executed and model artifacts are saved in `/models/`
2. **Dependencies**: All required packages are installed
3. **Python Environment**: Proper Python environment is configured

## 🚀 Step-by-Step Testing Instructions

### 1. Install Required Packages

First, install Streamlit and other dependencies if not already installed:

```bash
pip install streamlit plotly pandas numpy scikit-learn joblib
```

### 2. Navigate to Project Directory

Open your terminal/command prompt and navigate to the project root:

```bash
cd "c:\Users\FX506\Desktop\music&mental_health\music-wellbeing-recommender"
```

### 3. Run the Streamlit App

Execute the following command to start the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

**Alternative commands if the above doesn't work:**
```bash
python -m streamlit run app/streamlit_app.py
```

### 4. Access the App

After running the command, you should see output like:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open your browser and go to `http://localhost:8501`

## 🧪 Testing Scenarios

### Test Case 1: High Anxiety User
1. Navigate to the user input section
2. Enter the following profile:
   - **Age**: 25
   - **Daily listening hours**: 3
   - **Anxiety level**: 8
   - **Depression level**: 4
   - **Insomnia level**: 5
   - **OCD level**: 2
   - **Preferred BPM**: 120
   - **Primary streaming service**: Spotify
   - **Listen while working**: Yes
   - **Play instruments**: No
   - **Explore new music**: Yes

3. Click "Get Recommendations"
4. **Expected Results**: Should recommend calming genres like Classical, Ambient, etc.

### Test Case 2: Depression-Focused User
1. Enter profile:
   - **Age**: 30
   - **Daily listening hours**: 4
   - **Anxiety level**: 3
   - **Depression level**: 8
   - **Insomnia level**: 4
   - **OCD level**: 1
   - **Preferred BPM**: 100
   - **Primary streaming service**: Apple Music
   - **Listen while working**: No
   - **Play instruments**: Yes
   - **Explore new music**: Yes

2. Click "Get Recommendations"
3. **Expected Results**: Should recommend uplifting genres like Jazz, Folk, etc.

### Test Case 3: Insomnia User
1. Enter profile:
   - **Age**: 45
   - **Daily listening hours**: 1.5
   - **Anxiety level**: 5
   - **Depression level**: 3
   - **Insomnia level**: 9
   - **OCD level**: 1
   - **Preferred BPM**: 60
   - **Primary streaming service**: YouTube Music
   - **Listen while working**: No
   - **Play instruments**: No
   - **Explore new music**: No

2. Click "Get Recommendations"
3. **Expected Results**: Should recommend low-tempo, relaxing genres

## 🔍 Testing Checklist

### Functionality Tests
- [ ] App loads without errors
- [ ] All input fields are functional
- [ ] Form validation works (invalid inputs show errors)
- [ ] "Get Recommendations" button works
- [ ] Recommendations are displayed correctly
- [ ] Probability percentages are shown
- [ ] Explanations are provided for recommendations

### UI/UX Tests
- [ ] App layout is responsive
- [ ] Colors and styling are consistent
- [ ] Text is readable
- [ ] Navigation is intuitive
- [ ] Loading indicators work (if any)

### Model Integration Tests
- [ ] Model loads successfully
- [ ] Predictions are generated
- [ ] Feature engineering works correctly
- [ ] Genre recommendations are ranked properly
- [ ] Confidence scores are reasonable (0-100%)

### Error Handling Tests
- [ ] Invalid age (negative, too high) shows error
- [ ] Mental health scores outside 0-10 range show error
- [ ] Missing required fields show appropriate messages
- [ ] Model loading failures are handled gracefully

## 🛠️ Troubleshooting Common Issues

### Issue 1: "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue 2: "FileNotFoundError: Model not found"
**Solution**: 
1. Run the `03_modeling.ipynb` notebook completely
2. Ensure model files exist in `/models/` directory:
   - `music_effect_model.pkl`
   - `label_encoder.pkl`
   - `feature_columns.pkl`

### Issue 3: App doesn't start
**Solution**:
1. Check if port 8501 is already in use
2. Try a different port: `streamlit run app/streamlit_app.py --server.port 8502`
3. Check Python path and virtual environment

### Issue 4: Recommendations not showing
**Solution**:
1. Check browser console for JavaScript errors
2. Verify model predictions are working
3. Check data preprocessing steps

### Issue 5: Slow performance
**Solution**:
1. Check model size and loading time
2. Consider caching with `@st.cache_data`
3. Optimize recommendation algorithm

## 📊 Expected App Features

Your Streamlit app should include:

1. **User Profile Input**:
   - Demographic information (age)
   - Mental health indicators (anxiety, depression, insomnia, OCD)
   - Music preferences (BPM, streaming service, genres)
   - Listening habits (hours per day, while working, etc.)

2. **Recommendation Output**:
   - Top 5 recommended genres
   - Confidence percentages
   - Predicted effects (Improve/No effect/Worsen)
   - Explanations for each recommendation

3. **Visualizations**:
   - Bar chart of recommendation probabilities
   - Mental health profile radar chart
   - Music preference breakdown

4. **Additional Features**:
   - About section with model information
   - Data source information
   - Disclaimer about medical advice

## 🚀 Performance Benchmarks

- **App Loading Time**: < 5 seconds
- **Recommendation Generation**: < 3 seconds
- **Model Prediction**: < 1 second per genre
- **Memory Usage**: < 500MB

## 📝 Test Results Documentation

Create a test results file documenting:
- Test date and time
- Test cases executed
- Pass/fail status
- Screenshots of successful runs
- Performance metrics
- Any bugs or issues found

## 🔄 Continuous Testing

For ongoing development:
1. Test after each code change
2. Validate with different user profiles
3. Check model performance regularly
4. Monitor user feedback and adjust accordingly

Remember: The goal is to provide helpful, personalized music recommendations that could genuinely improve users' mental wellbeing!