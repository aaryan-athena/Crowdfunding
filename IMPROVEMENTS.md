# Model Sensitivity Improvements - Summary

## Changes Made to Improve Prediction Sensitivity

### 1. Enhanced Feature Engineering

**Added comprehensive interaction features:**

- **NumBackers interactions** (most important for raised amount):
  - NumBackers × Category (each category)
  - NumBackers × GoalAmount
  - NumBackers × DurationDays
  
- **Social Media & Updates interactions**:
  - SocialMediaPresence × NumUpdates
  - SocialMediaPresence × NumBackers
  
- **Owner Experience interactions**:
  - OwnerExperience × GoalAmount
  - OwnerExperience × NumBackers
  
- **Country & Currency interactions**:
  - GoalAmount × Country (each country)
  - GoalAmount × Currency (each currency)
  
- **Video interactions**:
  - VideoIncluded × NumBackers
  - VideoIncluded × SocialMediaPresence

### 2. Feature Scaling (StandardScaler)

- Added StandardScaler to normalize all features
- This makes the model more sensitive to relative changes in features
- Prevents large-scale features (like GoalAmount) from dominating small-scale features

### 3. Improved Model Parameters

**Changed from:**
```python
BayesianRidge(
    n_iter=300,
    tol=1e-3,
    alpha_1=1e-6,
    alpha_2=1e-6,
    lambda_1=1e-6,
    lambda_2=1e-6
)
```

**Changed to:**
```python
BayesianRidge(
    n_iter=500,          # More iterations for better convergence
    tol=1e-4,            # Tighter tolerance
    alpha_1=1e-5,        # Adjusted regularization
    alpha_2=1e-5,
    lambda_1=1e-5,
    lambda_2=1e-5,
    fit_intercept=True   # Ensure intercept is fitted
)
```

### 4. Updated Prediction Function

- Now creates all interaction features during prediction (not just base features)
- Applies same scaling transformation as training
- Ensures consistency between training and prediction

## Why These Changes Improve Sensitivity

1. **Interaction Features**: Capture how features work together
   - Example: 100 backers for a Technology campaign behaves differently than 100 backers for a Film campaign
   - GoalAmount impact varies by category and country

2. **Feature Scaling**: Puts all features on same scale
   - Prevents bias toward large-magnitude features
   - Makes model coefficients more interpretable
   - Improves numerical stability

3. **More Iterations**: Allows model to find better coefficients
   - Better convergence to optimal solution
   - More accurate feature importance weights

## Expected Impact

- **NumBackers**: Should have STRONG positive impact (most important predictor)
- **GoalAmount**: Should have moderate impact (higher goals = higher raised amounts typically)
- **Categories**: Different categories should show different prediction patterns
- **Social Media + Updates**: Should show combined positive effect
- **Owner Experience**: Should show positive correlation
- **Video Included**: Should show positive impact
- **Country/Currency**: Should show regional variations

## Testing

Run the test script to verify sensitivity:
```bash
python test_sensitivity.py
```

This will test how predictions change when you vary each parameter individually.

## Files Modified

1. **app.py**: 
   - Enhanced feature engineering in `load_and_train_model()`
   - Added StandardScaler for feature normalization
   - Updated prediction function to create interaction features
   - Improved model parameters

2. **model.pkl**: 
   - Deleted (will be regenerated with new features on next run)
   - Now includes scaler object

## Next Steps

1. Delete old `model.pkl` (already done)
2. Restart Flask app - it will retrain with new features
3. Run `test_sensitivity.py` to verify improvements
4. Deploy to Render with updated code

## Deployment Notes

- All requirements already in `requirements.txt`
- `Procfile` configured for Gunicorn
- App configured for Render port binding
- Model will auto-train on first deployment if model.pkl missing
