# Crowdfunding Prediction Web App

A Flask web application that predicts crowdfunding campaign raised amounts using a Bayesian Ridge regression model.

## Features

- Clean, responsive web interface built with TailwindCSS v4
- Real-time prediction using machine learning
- Form validation and error handling
- Professional styling with smooth animations

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

## How It Works

1. The app loads the crowdfunding dataset and trains a Bayesian Ridge model
2. Users fill out a form with campaign parameters
3. The model predicts the expected raised amount based on the input
4. Results are displayed in real-time

## Input Parameters

- **Goal Amount**: Target funding amount in dollars
- **Duration Days**: Length of the campaign in days
- **Number of Backers**: Expected number of supporters
- **Owner Experience**: Years of experience of the campaign owner
- **Social Media Presence**: Number of social media followers
- **Number of Updates**: Planned campaign updates
- **Category**: Campaign category (Film, Games, Music, etc.)
- **Launch Month**: Month when the campaign will launch
- **Country**: Country where the campaign is based
- **Currency**: Currency for the campaign
- **Video Included**: Whether a video is included in the campaign

## Model Details

The application uses a Bayesian Ridge regression model trained on crowdfunding campaign data. The model considers various features including campaign parameters, categorical variables (one-hot encoded), and temporal factors to predict the raised amount.

### Key Feature Relationships

**Primary Predictors:**
- **Goal Amount**: Strongest predictor (~1:1 relationship with raised amount)
- **Number of Backers**: Negative correlation (fewer backers = higher individual contributions)
- **Owner Experience**: Negative impact (experienced owners may set realistic goals)

**Category-Specific Behavior:**
- **Technology campaigns**: Longer duration reduces funding success
- **Film/Music campaigns**: Longer duration improves funding success  
- **Games campaigns**: Moderate duration sensitivity

**Feature Interactions:**
- The model uses interaction features (e.g., `DurationDays × Category`) to capture how campaign behavior varies by category
- Different categories have distinct funding patterns and optimal strategies
- Social media presence and updates have moderate positive impact

**Model Performance:**
- Uses full dataset for training with 80/20 train-test split
- Includes feature importance analysis and categorical impact testing
- Interaction features significantly improve prediction accuracy over basic categorical encoding