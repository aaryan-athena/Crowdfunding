# Quick Start Guide - Testing Your Enhanced Model

## Step 1: Start the Flask App

Open a terminal and run:
```bash
cd "c:\Users\AIT 33\Desktop\Crowdfunding"
python app.py
```

**What will happen:**
- The app will detect that model.pkl is missing
- It will automatically download the dataset and train a new model
- This will take 1-2 minutes with the enhanced features
- You'll see training progress and final R² score
- The server will start on http://localhost:5000

## Step 2: Test the Model Sensitivity

Open a **second terminal** while the Flask app is running:
```bash
cd "c:\Users\AIT 33\Desktop\Crowdfunding"
python test_sensitivity.py
```

**What this does:**
- Tests how predictions change when you vary each parameter
- Shows the delta (Δ) - the change from baseline
- You should see SIGNIFICANT changes in predictions now!

## Expected Results

If the improvements worked, you should see:

### Strong Impact (Large Δ values):
- **NumBackers**: Changing from 50 to 250 should show $10,000+ difference
- **GoalAmount**: Higher goals should generally predict higher raised amounts
- **Category**: Different categories should predict different amounts

### Moderate Impact:
- **Social Media + Updates**: Combined effect should be noticeable
- **Owner Experience**: More experience = higher predictions
- **Video Included**: Yes vs No should show difference

### Smaller Impact:
- **Duration**: Some variation but not huge
- **Country/Currency**: Regional differences

## Step 3: Deploy to Render

Once you verify locally that predictions are changing properly:

1. **Commit changes to GitHub:**
```bash
git add .
git commit -m "Enhanced model with interaction features and scaling"
git push origin main
```

2. **Redeploy on Render:**
   - Render will auto-deploy when you push
   - Or manually trigger deployment in Render dashboard
   - First deployment will take longer as it trains the model

## Troubleshooting

### If test_sensitivity.py fails:
- Make sure Flask app is running first
- Check that it's on http://localhost:5000
- Wait for model training to complete

### If predictions still don't vary much:
- Check the test output - look for the Δ values
- If they're all near zero, there may be a data issue
- Review the training output for R² score (should be > 0.5)

### If deployment fails:
- Check Render logs for specific errors
- Ensure all files are committed to GitHub
- Verify requirements.txt has all dependencies

## Key Files

- `app.py` - Main application with enhanced features
- `test_sensitivity.py` - Test script to verify predictions vary
- `IMPROVEMENTS.md` - Detailed explanation of changes
- `requirements.txt` - Python dependencies (includes setuptools, wheel)
- `Procfile` - Tells Render how to start the app

## Success Criteria

✅ Model trains successfully  
✅ R² score > 0.5  
✅ Test script shows varying predictions  
✅ NumBackers has strong impact on predictions  
✅ Categories show different prediction patterns  
✅ App runs on Render without errors  

Good luck! 🚀
