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
feature_columns = None

def load_and_train_model():
    """Load data, train model with full dataset and enhanced features, and save it"""
    global model, feature_columns
    
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
    
    # Create interaction features between important numerical and categorical features
    print("Creating interaction features...")
    
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
    
    print(f"After interaction features shape: {df_subset_encoded.shape}")
    
    # Separate features and target
    X = df_subset_encoded.drop(columns=['RaisedAmount'])
    y = df_subset_encoded['RaisedAmount']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Store feature columns for later use
    feature_columns = X.columns.tolist()
    
    # Train model with enhanced parameters
    print("Training BayesianRidge model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = BayesianRidge(
        n_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        compute_score=True
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
    
    return model, feature_columns

def load_model():
    """Load existing model or train new one"""
    global model, feature_columns
    
    if os.path.exists('model.pkl'):
        print("Loading existing model...")
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            feature_columns = data['feature_columns']
            
            # Display model info if available
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"Loaded model - R²: {metrics.get('test_r2', 'N/A'):.4f}, "
                      f"Training time: {metrics.get('training_time', 'N/A'):.2f}s")
            else:
                print("Model loaded successfully")
    else:
        model, feature_columns = load_and_train_model()
    
    return model, feature_columns

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.get_json()
        
        # Create input dataframe
        user_input = {
            'GoalAmount': [int(data['goalAmount'])],
            'DurationDays': [int(data['durationDays'])],
            'NumBackers': [int(data['numBackers'])],
            'OwnerExperience': [int(data['ownerExperience'])],
            'SocialMediaPresence': [int(data['socialMediaPresence'])],
            'NumUpdates': [int(data['numUpdates'])]
        }
        
        # Handle categorical variables with one-hot encoding
        categories = ['Film', 'Games', 'Music', 'Other', 'Publishing', 'Technology']
        for cat in categories:
            user_input[f'Category_{cat}'] = [data['category'] == cat]
        
        months = ['August', 'December', 'February', 'January', 'July', 'June', 
                 'March', 'May', 'November', 'October', 'September']
        for month in months:
            user_input[f'LaunchMonth_{month}'] = [data['launchMonth'] == month]
        
        countries = ['Canada', 'Germany', 'UK', 'USA']
        for country in countries:
            user_input[f'Country_{country}'] = [data['country'] == country]
        
        currencies = ['CAD', 'EUR', 'GBP', 'USD']
        for currency in currencies:
            user_input[f'Currency_{currency}'] = [data['currency'] == currency]
        
        user_input['VideoIncluded_Yes'] = [data['videoIncluded'] == 'Yes']
        
        # Create DataFrame and ensure column order matches training data
        user_df = pd.DataFrame(user_input)
        
        # Add missing columns with False/0 values
        for col in feature_columns:
            if col not in user_df.columns:
                user_df[col] = [False]
        
        # Reorder columns to match training data
        user_df = user_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(user_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Load or train model on startup
    load_model()
    app.run(debug=True)