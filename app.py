from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os
import time

app = Flask(__name__)

# Global variables for model and feature columns
model = None
scaler = None
feature_columns = None

def load_and_train_model():
    """Load data, train model with full dataset and enhanced features, and save it"""
    global model, scaler, feature_columns
    
    print("No pre-trained model found. Training new model...")
    print("Loading dataset...")
    
    # Load data
    df = pd.read_csv('https://raw.githubusercontent.com/ArchanaInsights/Datasets/refs/heads/main/crowdfunding_campaign.csv')
    print(f"Dataset shape: {df.shape}")
    print("Using full dataset for training")
    
    # Prepare data
    df_subset = df.drop(columns=['CampaignID', 'IsSuccessful'])
    
    # Check for missing values
    missing_values = df_subset.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Missing values found:\n{missing_values[missing_values > 0]}")
    
    # Analyze categorical distributions
    categorical_cols = df_subset.select_dtypes(include=['object', 'category']).columns
    print(f"Categorical columns: {list(categorical_cols)}")
    
    # One-hot encode categorical variables
    df_subset_encoded = pd.get_dummies(df_subset, columns=categorical_cols, drop_first=True)
    print(f"After encoding shape: {df_subset_encoded.shape}")
    
    # Create enhanced interaction features
    print("Creating enhanced interaction features...")
    
    # Goal amount interactions with categories
    if 'GoalAmount' in df_subset_encoded.columns:
        for col in df_subset_encoded.columns:
            if col.startswith('Category_'):
                df_subset_encoded[f'GoalAmount_x_{col}'] = df_subset_encoded['GoalAmount'] * df_subset_encoded[col]
    
    # Duration interactions with categories  
    if 'DurationDays' in df_subset_encoded.columns:
        for col in df_subset_encoded.columns:
            if col.startswith('Category_'):
                df_subset_encoded[f'DurationDays_x_{col}'] = df_subset_encoded['DurationDays'] * df_subset_encoded[col]
    
    # NumBackers interactions - very important for raised amount
    if 'NumBackers' in df_subset_encoded.columns:
        for col in df_subset_encoded.columns:
            if col.startswith('Category_'):
                df_subset_encoded[f'NumBackers_x_{col}'] = df_subset_encoded['NumBackers'] * df_subset_encoded[col]
        
        # NumBackers with other numerical features
        if 'GoalAmount' in df_subset_encoded.columns:
            df_subset_encoded['NumBackers_x_GoalAmount'] = df_subset_encoded['NumBackers'] * df_subset_encoded['GoalAmount']
        if 'DurationDays' in df_subset_encoded.columns:
            df_subset_encoded['NumBackers_x_DurationDays'] = df_subset_encoded['NumBackers'] * df_subset_encoded['DurationDays']
    
    # Social media and updates interactions
    if 'SocialMediaPresence' in df_subset_encoded.columns and 'NumUpdates' in df_subset_encoded.columns:
        df_subset_encoded['Social_x_Updates'] = df_subset_encoded['SocialMediaPresence'] * df_subset_encoded['NumUpdates']
        if 'NumBackers' in df_subset_encoded.columns:
            df_subset_encoded['Social_x_Backers'] = df_subset_encoded['SocialMediaPresence'] * df_subset_encoded['NumBackers']
    
    # Experience-based interactions
    if 'OwnerExperience' in df_subset_encoded.columns:
        if 'GoalAmount' in df_subset_encoded.columns:
            df_subset_encoded['Experience_x_Goal'] = df_subset_encoded['OwnerExperience'] * df_subset_encoded['GoalAmount']
        if 'NumBackers' in df_subset_encoded.columns:
            df_subset_encoded['Experience_x_Backers'] = df_subset_encoded['OwnerExperience'] * df_subset_encoded['NumBackers']
    
    # Country and Currency interactions
    for col in df_subset_encoded.columns:
        if col.startswith('Country_') or col.startswith('Currency_'):
            if 'GoalAmount' in df_subset_encoded.columns:
                df_subset_encoded[f'GoalAmount_x_{col}'] = df_subset_encoded['GoalAmount'] * df_subset_encoded[col]
    
    # Video interaction with categories
    for col in df_subset_encoded.columns:
        if col.startswith('VideoIncluded_'):
            if 'NumBackers' in df_subset_encoded.columns:
                df_subset_encoded[f'Video_x_Backers'] = df_subset_encoded[col] * df_subset_encoded['NumBackers']
            if 'SocialMediaPresence' in df_subset_encoded.columns:
                df_subset_encoded[f'Video_x_Social'] = df_subset_encoded[col] * df_subset_encoded['SocialMediaPresence']
    
    print(f"After interaction features shape: {df_subset_encoded.shape}")
    
    # Separate features and target
    X = df_subset_encoded.drop(columns=['RaisedAmount'])
    y = df_subset_encoded['RaisedAmount']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Store feature columns for later use
    feature_columns = X.columns.tolist()
    
    # Normalize features for better sensitivity
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train model with enhanced parameters
    print("Training BayesianRidge model with scaled features...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    model = BayesianRidge(
        n_iter=500,
        tol=1e-4,
        alpha_1=1e-5,
        alpha_2=1e-5,
        lambda_1=1e-5,
        lambda_2=1e-5,
        compute_score=True,
        fit_intercept=True
    )
    
    import time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate model
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Model Performance - R²: {test_r2:.4f}, RMSE: ${test_rmse:,.2f}, MAE: ${test_mae:,.2f}")
    
    # Save model and metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'metrics': {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'training_time': training_time,
            'n_iterations': model.n_iter_
        },
        'model_type': 'BayesianRidge'
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to 'model.pkl' with R² score: {test_r2:.4f}")
    
    return model, scaler, feature_columns

def load_model():
    """Load existing model or train new one"""
    global model, scaler, feature_columns
    
    if os.path.exists('model.pkl'):
        print("Loading existing model...")
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            scaler = data.get('scaler', None)  # Get scaler if it exists
            feature_columns = data['feature_columns']
            
            # Display model info if available
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"Loaded model - R²: {metrics.get('test_r2', 'N/A'):.4f}, "
                      f"Training time: {metrics.get('training_time', 'N/A'):.2f}s")
            else:
                print("Model loaded successfully")
    else:
        model, scaler, feature_columns = load_and_train_model()
    
    return model, scaler, feature_columns

@app.route('/')
def index():
    return render_template('index.html')

def calculate_multipliers(num_backers, duration_days, social_media, num_updates, owner_experience, category, video_included):
    """
    Calculate multipliers for custom formula based on business logic
    Returns individual multipliers and combined multiplier
    """
    
    # 1. Backer Multiplier (Most Critical)
    if num_backers == 0:
        backer_mult = 0.05  # Minimal baseline
    elif num_backers <= 10:
        backer_mult = 0.3  # Early stage
    elif num_backers <= 50:
        backer_mult = 0.6  # Gaining momentum
    elif num_backers <= 100:
        backer_mult = 0.8  # Active
    else:
        backer_mult = 1.0  # Full model prediction
    
    # 2. Duration Multiplier
    if duration_days == 0:
        duration_mult = 0.2  # No time to raise
    elif duration_days <= 7:
        duration_mult = 0.4  # Insufficient time
    elif duration_days <= 15:
        duration_mult = 0.6  # Short campaign
    elif duration_days <= 30:
        duration_mult = 0.8  # Standard
    else:
        duration_mult = 1.0  # Optimal (30+ days)
    
    # 3. Social Media Impact
    if social_media == 0:
        social_mult = 0.5  # No reach
    elif social_media <= 3:
        social_mult = 0.65  # Minimal presence
    elif social_media <= 7:
        social_mult = 0.8  # Good presence
    else:
        social_mult = 1.0  # Strong presence (8+)
    
    # 4. Campaign Updates Impact
    if num_updates == 0:
        updates_mult = 0.6  # Abandoned campaign
    elif num_updates <= 5:
        updates_mult = 0.75  # Minimal engagement
    elif num_updates <= 10:
        updates_mult = 0.9  # Active engagement
    else:
        updates_mult = 1.05  # Very active (10+)
    
    # 5. Owner Experience Impact
    if owner_experience == 0:
        exp_mult = 0.7  # First-time, risky
    elif owner_experience <= 3:
        exp_mult = 0.85  # Some track record
    elif owner_experience <= 7:
        exp_mult = 0.95  # Experienced
    else:
        exp_mult = 1.05  # Proven (8+)
    
    # 6. Category Multiplier
    category_multipliers = {
        'Technology': 1.0,
        'Games': 0.95,
        'Film': 0.85,
        'Music': 0.80,
        'Publishing': 0.75,
        'Other': 0.70
    }
    cat_mult = category_multipliers.get(category, 0.70)
    
    # 7. Video Presence Impact
    video_mult = 1.1 if video_included == 'Yes' else 0.7
    
    # 8. Country Adjustment (using default USA if not provided, can be enhanced later)
    country_mult = 1.0  # Default
    
    # Calculate combined multiplier
    total_multiplier = (backer_mult * duration_mult * social_mult * 
                        updates_mult * exp_mult * cat_mult * video_mult * country_mult)
    
    # Apply bounds to prevent extreme values
    if total_multiplier < 0.15:
        total_multiplier = 0.15  # Minimum threshold
    elif total_multiplier > 1.5:
        total_multiplier = 1.5  # Maximum threshold
    
    return {
        'backer': backer_mult,
        'duration': duration_mult,
        'social': social_mult,
        'updates': updates_mult,
        'experience': exp_mult,
        'category': cat_mult,
        'video': video_mult,
        'country': country_mult,
        'total': total_multiplier
    }

def apply_custom_formula(model_prediction, goal_amount, multipliers):
    """
    Apply custom calculation logic to model predictions
    Combines ML predictions with business logic for realistic results
    """
    # Start with model prediction
    adjusted_amount = model_prediction * multipliers['total']
    
    # Apply hard limits
    # Minimum viable amount - at least $100
    if adjusted_amount < 100:
        adjusted_amount = 100
    
    # Maximum realistic amount - can't raise more than 3x the goal (realistically ~80% of campaigns succeed)
    max_realistic_amount = goal_amount * 3
    if adjusted_amount > max_realistic_amount:
        adjusted_amount = max_realistic_amount
    
    # Round to nearest dollar
    adjusted_amount = round(adjusted_amount, 2)
    
    return adjusted_amount

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.get_json()
        
        # Extract numerical values
        goal_amount = float(data['goalAmount'])
        duration_days = int(data['durationDays'])
        num_backers = int(data['numBackers'])
        owner_experience = int(data['ownerExperience'])
        social_media = int(data['socialMediaPresence'])
        num_updates = int(data['numUpdates'])
        
        # Create input dataframe
        user_input = {
            'GoalAmount': [goal_amount],
            'DurationDays': [duration_days],
            'NumBackers': [num_backers],
            'OwnerExperience': [owner_experience],
            'SocialMediaPresence': [social_media],
            'NumUpdates': [num_updates]
        }
        
        # Handle categorical variables with one-hot encoding
        categories = ['Film', 'Games', 'Music', 'Other', 'Publishing', 'Technology']
        for cat in categories:
            is_cat = (data['category'] == cat)
            user_input[f'Category_{cat}'] = [is_cat]
            
            # Add GoalAmount interactions
            user_input[f'GoalAmount_x_Category_{cat}'] = [goal_amount * is_cat]
            # Add DurationDays interactions
            user_input[f'DurationDays_x_Category_{cat}'] = [duration_days * is_cat]
            # Add NumBackers interactions
            user_input[f'NumBackers_x_Category_{cat}'] = [num_backers * is_cat]
        
        months = ['August', 'December', 'February', 'January', 'July', 'June', 
                 'March', 'May', 'November', 'October', 'September']
        for month in months:
            user_input[f'LaunchMonth_{month}'] = [data['launchMonth'] == month]
        
        countries = ['Canada', 'Germany', 'UK', 'USA']
        for country in countries:
            is_country = (data['country'] == country)
            user_input[f'Country_{country}'] = [is_country]
            user_input[f'GoalAmount_x_Country_{country}'] = [goal_amount * is_country]
        
        currencies = ['CAD', 'EUR', 'GBP', 'USD']
        for currency in currencies:
            is_currency = (data['currency'] == currency)
            user_input[f'Currency_{currency}'] = [is_currency]
            user_input[f'GoalAmount_x_Currency_{currency}'] = [goal_amount * is_currency]
        
        is_video = (data['videoIncluded'] == 'Yes')
        user_input['VideoIncluded_Yes'] = [is_video]
        
        # Add all the interaction features
        user_input['NumBackers_x_GoalAmount'] = [num_backers * goal_amount]
        user_input['NumBackers_x_DurationDays'] = [num_backers * duration_days]
        user_input['Social_x_Updates'] = [social_media * num_updates]
        user_input['Social_x_Backers'] = [social_media * num_backers]
        user_input['Experience_x_Goal'] = [owner_experience * goal_amount]
        user_input['Experience_x_Backers'] = [owner_experience * num_backers]
        user_input['Video_x_Backers'] = [is_video * num_backers]
        user_input['Video_x_Social'] = [is_video * social_media]
        
        # Create DataFrame
        user_df = pd.DataFrame(user_input)
        
        # Add missing columns with 0 values (for other categories not selected)
        for col in feature_columns:
            if col not in user_df.columns:
                user_df[col] = [0]
        
        # Reorder columns to match training data
        user_df = user_df[feature_columns]
        
        # Scale the features if scaler is available
        if scaler is not None:
            user_df_scaled = scaler.transform(user_df)
            model_prediction = model.predict(user_df_scaled)[0]
        else:
            model_prediction = model.predict(user_df)[0]
        
        # Get category for multiplier calculations
        category = data.get('category', 'Other')
        video_included = data.get('videoIncluded', 'No')
        
        # Calculate multipliers based on business logic
        multipliers = calculate_multipliers(
            num_backers=num_backers,
            duration_days=duration_days,
            social_media=social_media,
            num_updates=num_updates,
            owner_experience=owner_experience,
            category=category,
            video_included=video_included
        )
        
        # Apply custom formula to get realistic prediction
        final_prediction = apply_custom_formula(
            model_prediction=model_prediction,
            goal_amount=goal_amount,
            multipliers=multipliers
        )
        
        return jsonify({
            'success': True,
            'prediction': final_prediction,
            'model_prediction': round(model_prediction, 2),
            'multiplier': round(multipliers['total'], 4),
            'adjustment_details': {
                'backer_multiplier': round(multipliers['backer'], 2),
                'duration_multiplier': round(multipliers['duration'], 2),
                'social_multiplier': round(multipliers['social'], 2),
                'updates_multiplier': round(multipliers['updates'], 2),
                'experience_multiplier': round(multipliers['experience'], 2),
                'category_multiplier': round(multipliers['category'], 2),
                'video_multiplier': round(multipliers['video'], 2)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Load or train model on startup
    load_model()
    # Get port from environment variable (Render sets this) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Bind to 0.0.0.0 to be accessible externally (required for Render)
    app.run(host='0.0.0.0', port=port, debug=False)