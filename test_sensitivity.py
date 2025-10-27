"""
Test script to verify that model predictions are sensitive to input parameter changes
"""
import requests
import json
import time

# Wait for server to be ready
print("Make sure the Flask app is running on http://localhost:5000")
print("Testing model sensitivity to parameter changes...\n")

base_url = "http://localhost:5000/predict"

# Base test case
base_input = {
    "goalAmount": 10000,
    "durationDays": 30,
    "numBackers": 100,
    "ownerExperience": 5,
    "socialMediaPresence": 3,
    "numUpdates": 10,
    "category": "Technology",
    "launchMonth": "January",
    "country": "USA",
    "currency": "USD",
    "videoIncluded": "Yes"
}

def make_prediction(data):
    try:
        response = requests.post(base_url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('prediction'), result.get('multiplier', 1.0), result.get('model_prediction', 0)
        return None, None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

print("=" * 80)
print("BASE CASE")
print("=" * 80)
base_prediction, base_mult, base_model = make_prediction(base_input)
if base_prediction:
    print(f"Model prediction: ${base_model:,.2f}")
    print(f"Multiplier: {base_mult:.4f}")
    print(f"Final prediction: ${base_prediction:,.2f}\n")
else:
    print("Failed to get base prediction. Make sure Flask app is running!\n")
    exit(1)

# Test 1: NumBackers sensitivity (most important)
print("=" * 80)
print("TEST 1: NumBackers Impact (Very Important)")
print("=" * 80)
for backers in [0, 5, 10, 50, 100, 150, 200, 250]:
    test_data = base_input.copy()
    test_data['numBackers'] = backers
    pred, mult, model_pred = make_prediction(test_data)
    if pred:
        diff = pred - base_prediction
        print(f"NumBackers: {backers:3d} → Prediction: ${pred:10,.2f} (Δ ${diff:+10,.2f}) [Multiplier: {mult:.4f}]")

# Test 2: GoalAmount sensitivity
print("\n" + "=" * 80)
print("TEST 2: GoalAmount Impact")
print("=" * 80)
for goal in [5000, 10000, 20000, 30000, 50000]:
    test_data = base_input.copy()
    test_data['goalAmount'] = goal
    pred, mult, model_pred = make_prediction(test_data)
    if pred:
        diff = pred - base_prediction
        print(f"GoalAmount: ${goal:5d} → Prediction: ${pred:10,.2f} (Δ ${diff:+10,.2f}) [Multiplier: {mult:.4f}]")

# Test 3: Category sensitivity
print("\n" + "=" * 80)
print("TEST 3: Category Impact (Categorical)")
print("=" * 80)
categories = ["Technology", "Film", "Games", "Music", "Publishing", "Other"]
for cat in categories:
    test_data = base_input.copy()
    test_data['category'] = cat
    pred, mult, model_pred = make_prediction(test_data)
    if pred:
        diff = pred - base_prediction
        print(f"Category: {cat:15s} → Prediction: ${pred:10,.2f} (Δ ${diff:+10,.2f}) [Multiplier: {mult:.4f}]")

# Test 4: Social Media + Updates
print("\n" + "=" * 80)
print("TEST 4: Social Media & Updates Impact")
print("=" * 80)
for social, updates in [(0, 0), (1, 2), (3, 10), (5, 15), (7, 20), (10, 30)]:
    test_data = base_input.copy()
    test_data['socialMediaPresence'] = social
    test_data['numUpdates'] = updates
    pred, mult, model_pred = make_prediction(test_data)
    if pred:
        diff = pred - base_prediction
        print(f"Social: {social:2d}, Updates: {updates:2d} → Prediction: ${pred:10,.2f} (Δ ${diff:+10,.2f}) [Multiplier: {mult:.4f}]")

# Test 5: Owner Experience
print("\n" + "=" * 80)
print("TEST 5: Owner Experience Impact")
print("=" * 80)
for exp in [0, 1, 3, 5, 7, 10]:
    test_data = base_input.copy()
    test_data['ownerExperience'] = exp
    pred, mult, model_pred = make_prediction(test_data)
    if pred:
        diff = pred - base_prediction
        print(f"Experience: {exp:2d} → Prediction: ${pred:10,.2f} (Δ ${diff:+10,.2f}) [Multiplier: {mult:.4f}]")

# Test 6: Duration Days
print("\n" + "=" * 80)
print("TEST 6: Duration Days Impact")
print("=" * 80)
for days in [0, 5, 15, 30, 45, 60, 90]:
    test_data = base_input.copy()
    test_data['durationDays'] = days
    pred, mult, model_pred = make_prediction(test_data)
    if pred:
        diff = pred - base_prediction
        print(f"Duration: {days:2d} days → Prediction: ${pred:10,.2f} (Δ ${diff:+10,.2f}) [Multiplier: {mult:.4f}]")

# Test 7: Video Included
print("\n" + "=" * 80)
print("TEST 7: Video Included Impact")
print("=" * 80)
for video in ["No", "Yes"]:
    test_data = base_input.copy()
    test_data['videoIncluded'] = video
    pred, mult, model_pred = make_prediction(test_data)
    if pred:
        diff = pred - base_prediction
        print(f"Video: {video:3s} → Prediction: ${pred:10,.2f} (Δ ${diff:+10,.2f}) [Multiplier: {mult:.4f}]")

# Test 8: Edge case - All zeros
print("\n" + "=" * 80)
print("TEST 8: EDGE CASE - All Parameters at Zero")
print("=" * 80)
zero_input = {
    "goalAmount": 10000,
    "durationDays": 0,
    "numBackers": 0,
    "ownerExperience": 0,
    "socialMediaPresence": 0,
    "numUpdates": 0,
    "category": "Other",
    "launchMonth": "January",
    "country": "USA",
    "currency": "USD",
    "videoIncluded": "No"
}
pred, mult, model_pred = make_prediction(zero_input)
if pred:
    print(f"All zeros → Prediction: ${pred:10,.2f}")
    print(f"Model prediction: ${model_pred:10,.2f}")
    print(f"Multiplier: {mult:.4f}")
    print(f"Adjustment: {((pred/model_pred - 1) * 100):.1f}% reduction")

# Test 9: Strong campaign
print("\n" + "=" * 80)
print("TEST 9: STRONG CAMPAIGN - All Parameters Optimal")
print("=" * 80)
strong_input = {
    "goalAmount": 50000,
    "durationDays": 60,
    "numBackers": 200,
    "ownerExperience": 10,
    "socialMediaPresence": 10,
    "numUpdates": 30,
    "category": "Technology",
    "launchMonth": "January",
    "country": "USA",
    "currency": "USD",
    "videoIncluded": "Yes"
}
pred, mult, model_pred = make_prediction(strong_input)
if pred:
    print(f"Strong campaign → Prediction: ${pred:10,.2f}")
    print(f"Model prediction: ${model_pred:10,.2f}")
    print(f"Multiplier: {mult:.4f}")

# Test 8: Country/Currency
print("\n" + "=" * 80)
print("TEST 8: Country/Currency Impact")
print("=" * 80)
country_currency = [("USA", "USD"), ("UK", "GBP"), ("Germany", "EUR"), ("Canada", "CAD")]
for country, currency in country_currency:
    test_data = base_input.copy()
    test_data['country'] = country
    test_data['currency'] = currency
    pred = make_prediction(test_data)
    if pred:
        diff = pred - base_prediction
        print(f"Country: {country:10s}, Currency: {currency} → Prediction: ${pred:10,.2f} (Δ ${diff:+10,.2f})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("If you see significant changes (Δ values), the model is sensitive to inputs!")
print("If all predictions are similar, there may still be an issue.")
