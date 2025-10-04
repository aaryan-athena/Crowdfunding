import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time

def load_and_prepare_data():
    """Load and prepare the crowdfunding dataset"""
    print("Loading dataset...")
    df = pd.read_csv('https://raw.githubusercontent.com/ArchanaInsights/Datasets/refs/heads/main/crowdfunding_campaign.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Use full dataset for training
    print(f"Using full dataset for training")
    
    # Prepare data
    df_subset = df.drop(columns=['CampaignID', 'IsSuccessful'])
    
    # Check for missing values
    print(f"\nMissing values:\n{df_subset.isnull().sum()}")
    
    # Analyze categorical distributions before encoding
    categorical_cols = df_subset.select_dtypes(include=['object', 'category']).columns
    print(f"\nCategorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        print(f"\n{col} distribution:")
        print(df_subset[col].value_counts().head())
    
    # One-hot encode categorical variables
    df_subset_encoded = pd.get_dummies(df_subset, columns=categorical_cols, drop_first=True)
    print(f"After encoding shape: {df_subset_encoded.shape}")
    
    # Create interaction features between important numerical and categorical features
    print("\nCreating interaction features...")
    
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
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target statistics:\n{y.describe()}")
    
    return X, y

def train_bayesian_ridge_model(X, y):
    """Train BayesianRidge model and evaluate performance"""
    print("\n" + "="*50)
    print("TRAINING BAYESIAN RIDGE MODEL")
    print("="*50)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize model with different parameters to show training process
    model = BayesianRidge(
        n_iter=300,  # Maximum number of iterations
        tol=1e-3,    # Tolerance for stopping criterion
        alpha_1=1e-6,  # Hyperparameter for alpha prior
        alpha_2=1e-6,  # Hyperparameter for alpha prior
        lambda_1=1e-6, # Hyperparameter for lambda prior
        lambda_2=1e-6, # Hyperparameter for lambda prior
        compute_score=True  # Compute log marginal likelihood at each iteration
    )
    
    # Train the model
    print("\nTraining model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Number of iterations: {model.n_iter_}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Display results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Training Metrics:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  MAE: ${train_mae:,.2f}")
    print(f"  MSE: ${train_mse:,.2f}")
    
    print(f"\nTest Metrics:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  MAE: ${test_mae:,.2f}")
    print(f"  MSE: ${test_mse:,.2f}")
    
    # Model parameters
    print(f"\nModel Parameters:")
    print(f"  Alpha (precision of noise): {model.alpha_:.6f}")
    print(f"  Lambda (precision of weights): {model.lambda_:.6f}")
    
    # Feature importance analysis
    print(f"\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Get feature coefficients (weights)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Top 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Analyze categorical vs numerical features
    categorical_features = [col for col in X.columns if any(cat in col for cat in ['Category_', 'LaunchMonth_', 'Country_', 'Currency_', 'VideoIncluded_'])]
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    cat_importance = feature_importance[feature_importance['feature'].isin(categorical_features)]['abs_coefficient'].mean()
    num_importance = feature_importance[feature_importance['feature'].isin(numerical_features)]['abs_coefficient'].mean()
    
    print(f"\nFeature Type Analysis:")
    print(f"  Average importance of categorical features: {cat_importance:.6f}")
    print(f"  Average importance of numerical features: {num_importance:.6f}")
    print(f"  Ratio (numerical/categorical): {num_importance/cat_importance:.2f}x")
    
    # Interpretation
    print(f"\n" + "="*50)
    print("MODEL INTERPRETATION")
    print("="*50)
    accuracy_percentage = test_r2 * 100
    print(f"Model Accuracy (R²): {accuracy_percentage:.2f}%")
    
    if test_r2 > 0.8:
        print("✅ Excellent model performance!")
    elif test_r2 > 0.6:
        print("✅ Good model performance!")
    elif test_r2 > 0.4:
        print("⚠️  Moderate model performance")
    else:
        print("❌ Poor model performance - consider feature engineering")
    
    avg_error_percentage = (test_mae / y_test.mean()) * 100
    print(f"Average prediction error: {avg_error_percentage:.2f}% of actual value")
    
    if num_importance > cat_importance * 5:
        print("⚠️  Categorical features have low impact - consider feature engineering")
    elif cat_importance > num_importance * 2:
        print("✅ Categorical features are highly influential")
    else:
        print("✅ Balanced feature importance between categorical and numerical")
    
    return model, X.columns.tolist(), {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'training_time': training_time,
        'n_iterations': model.n_iter_
    }

def save_model(model, feature_columns, metrics):
    """Save the trained model and metadata"""
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'metrics': metrics,
        'model_type': 'BayesianRidge'
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to 'model.pkl'")
    print(f"   - Model: BayesianRidge")
    print(f"   - Features: {len(feature_columns)}")
    print(f"   - Test R²: {metrics['test_r2']:.4f}")

def test_categorical_impact(X, y):
    """Test the impact of categorical features by comparing models"""
    print(f"\n" + "="*50)
    print("CATEGORICAL FEATURE IMPACT TEST")
    print("="*50)
    
    # Identify categorical and numerical features
    categorical_features = [col for col in X.columns if any(cat in col for cat in ['Category_', 'LaunchMonth_', 'Country_', 'Currency_', 'VideoIncluded_'])]
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model with only numerical features
    model_num = BayesianRidge()
    model_num.fit(X_train[numerical_features], y_train)
    y_pred_num = model_num.predict(X_test[numerical_features])
    r2_num = r2_score(y_test, y_pred_num)
    
    # Model with only categorical features
    if len(categorical_features) > 0:
        model_cat = BayesianRidge()
        model_cat.fit(X_train[categorical_features], y_train)
        y_pred_cat = model_cat.predict(X_test[categorical_features])
        r2_cat = r2_score(y_test, y_pred_cat)
    else:
        r2_cat = 0
    
    # Model with all features
    model_all = BayesianRidge()
    model_all.fit(X_train, y_train)
    y_pred_all = model_all.predict(X_test)
    r2_all = r2_score(y_test, y_pred_all)
    
    print(f"\nModel Performance Comparison:")
    print(f"  Numerical features only: R² = {r2_num:.4f}")
    print(f"  Categorical features only: R² = {r2_cat:.4f}")
    print(f"  All features combined: R² = {r2_all:.4f}")
    
    improvement = r2_all - r2_num
    print(f"\nCategorical features improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    if improvement < 0.01:
        print("⚠️  Categorical features add minimal value - consider feature engineering")
    elif improvement > 0.05:
        print("✅ Categorical features significantly improve the model")
    else:
        print("✅ Categorical features provide moderate improvement")

def main():
    """Main training pipeline"""
    print("CROWDFUNDING PREDICTION MODEL TRAINING")
    print("="*50)
    
    try:
        # Load and prepare data
        X, y = load_and_prepare_data()
        
        # Test categorical feature impact
        test_categorical_impact(X, y)
        
        # Train model
        model, feature_columns, metrics = train_bayesian_ridge_model(X, y)
        
        # Save model
        save_model(model, feature_columns, metrics)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"You can now run your Flask app with: python app.py")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()