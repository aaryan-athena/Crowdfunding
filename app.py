from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# ==========================================
# INDIA-FOCUSED CROWDFUNDING CALCULATOR
# Realistic algorithm based on actual Indian crowdfunding patterns
# Platforms reference: Ketto, Milaap, ImpactGuru, Wishberry
# ==========================================

# Average pledge amounts by category in INR (based on Indian crowdfunding data)
CATEGORY_AVG_PLEDGE_INR = {
    'Medical': 800,        # Medical emergencies get higher donations
    'Education': 500,      # Education causes get moderate support
    'Technology': 600,     # Tech products/startups
    'Creative': 400,       # Film, Music, Art projects
    'Social Cause': 450,   # NGO, Community projects
    'Sports': 350,         # Sports-related campaigns
    'Environment': 400,    # Environmental causes
    'Business': 550,       # Small business/startup funding
    'Personal': 300,       # Personal needs
    'Other': 350
}

# Conversion multipliers for different currencies (approximate rates)
CURRENCY_TO_INR = {
    'INR': 1,
    'USD': 83,
    'EUR': 90,
    'GBP': 105,
    'CAD': 62
}

# Platform fee percentages (average across Indian platforms)
PLATFORM_FEES = {
    'Ketto': 0.05,
    'Milaap': 0.05,
    'ImpactGuru': 0.06,
    'Wishberry': 0.10,
    'GoFundMe': 0.029,
    'Other': 0.05
}

def calculate_realistic_prediction(
    goal_amount,
    currency,
    category,
    num_backers,
    duration_days,
    social_followers,
    email_subscribers,
    num_updates,
    has_video,
    has_images,
    owner_experience,
    previous_campaigns,
    platform,
    city_tier,
    cause_urgency
):
    """
    Calculate realistic crowdfunding amount using a formula-based approach.
    This algorithm is designed for Indian crowdfunding scenarios.
    """
    
    # Convert goal to INR for calculations
    conversion_rate = CURRENCY_TO_INR.get(currency, 1)
    goal_inr = goal_amount * conversion_rate
    
    # Get base pledge amount for category
    base_pledge = CATEGORY_AVG_PLEDGE_INR.get(category, 350)
    
    # ==========================================
    # STEP 1: Calculate Base Amount from Backers
    # ==========================================
    # This is the primary driver - number of backers × average pledge
    base_amount = num_backers * base_pledge
    
    # ==========================================
    # STEP 2: Apply Multipliers
    # ==========================================
    
    # 2.1 Duration Effectiveness Multiplier
    # Optimal campaign duration is 30-45 days
    if duration_days <= 7:
        duration_mult = 0.4  # Too short
    elif duration_days <= 15:
        duration_mult = 0.65
    elif duration_days <= 30:
        duration_mult = 0.85
    elif duration_days <= 45:
        duration_mult = 1.0  # Optimal
    elif duration_days <= 60:
        duration_mult = 0.95
    else:
        duration_mult = 0.85  # Too long, fatigue sets in
    
    # 2.2 Social Media Reach Multiplier
    # Based on typical conversion rates (0.5-2% of followers donate)
    if social_followers == 0:
        social_mult = 0.6
        potential_social_backers = 0
    elif social_followers <= 500:
        social_mult = 0.7
        potential_social_backers = int(social_followers * 0.02)  # 2% conversion
    elif social_followers <= 2000:
        social_mult = 0.8
        potential_social_backers = int(social_followers * 0.015)
    elif social_followers <= 10000:
        social_mult = 0.9
        potential_social_backers = int(social_followers * 0.01)
    elif social_followers <= 50000:
        social_mult = 1.0
        potential_social_backers = int(social_followers * 0.008)
    else:
        social_mult = 1.1
        potential_social_backers = int(social_followers * 0.005)
    
    # 2.3 Email List Multiplier (email has higher conversion ~5-10%)
    if email_subscribers == 0:
        email_mult = 0.8
        potential_email_backers = 0
    elif email_subscribers <= 100:
        email_mult = 0.9
        potential_email_backers = int(email_subscribers * 0.08)
    elif email_subscribers <= 500:
        email_mult = 1.0
        potential_email_backers = int(email_subscribers * 0.06)
    else:
        email_mult = 1.1
        potential_email_backers = int(email_subscribers * 0.05)
    
    # 2.4 Campaign Updates Multiplier
    # Regular updates show commitment and build trust
    if num_updates == 0:
        updates_mult = 0.7
    elif num_updates <= 3:
        updates_mult = 0.85
    elif num_updates <= 7:
        updates_mult = 0.95
    elif num_updates <= 15:
        updates_mult = 1.05
    else:
        updates_mult = 1.1
    
    # 2.5 Media Content Multiplier
    media_mult = 1.0
    if has_video:
        media_mult += 0.25  # Video increases trust significantly
    if has_images:
        media_mult += 0.1
    if not has_video and not has_images:
        media_mult = 0.6  # No visuals is a big negative
    
    # 2.6 Owner Credibility Multiplier
    if owner_experience == 0:
        exp_mult = 0.75
    elif owner_experience <= 2:
        exp_mult = 0.85
    elif owner_experience <= 5:
        exp_mult = 0.95
    else:
        exp_mult = 1.05
    
    # 2.7 Previous Campaign Success Multiplier
    if previous_campaigns == 0:
        prev_mult = 0.85  # First-time campaigner
    elif previous_campaigns <= 2:
        prev_mult = 1.0
    else:
        prev_mult = 1.15  # Proven track record
    
    # 2.8 City Tier Multiplier (India-specific)
    # Tier 1: Mumbai, Delhi, Bangalore, etc.
    # Tier 2: Pune, Ahmedabad, Jaipur, etc.
    # Tier 3: Smaller cities and towns
    city_multipliers = {
        'Tier 1': 1.1,
        'Tier 2': 0.95,
        'Tier 3': 0.8,
        'Rural': 0.65
    }
    city_mult = city_multipliers.get(city_tier, 0.9)
    
    # 2.9 Cause Urgency Multiplier
    urgency_multipliers = {
        'Critical': 1.3,    # Life-threatening, immediate need
        'High': 1.15,       # Urgent but not critical
        'Medium': 1.0,      # Standard timeline
        'Low': 0.85         # No urgency, may affect donations
    }
    urgency_mult = urgency_multipliers.get(cause_urgency, 1.0)
    
    # 2.10 Platform Trust Multiplier
    platform_trust = {
        'Ketto': 1.05,
        'Milaap': 1.05,
        'ImpactGuru': 1.0,
        'Wishberry': 0.95,
        'GoFundMe': 0.9,  # Less popular in India
        'Other': 0.85
    }
    platform_mult = platform_trust.get(platform, 0.9)
    
    # ==========================================
    # STEP 3: Calculate Final Amount
    # ==========================================
    
    # Combine all multipliers
    total_multiplier = (
        duration_mult *
        social_mult *
        email_mult *
        updates_mult *
        media_mult *
        exp_mult *
        prev_mult *
        city_mult *
        urgency_mult *
        platform_mult
    )
    
    # Apply multiplier to base amount
    adjusted_amount = base_amount * total_multiplier
    
    # Add potential backers from social media and email
    additional_backers = potential_social_backers + potential_email_backers
    additional_amount = additional_backers * base_pledge * 0.5  # Lower pledge from acquired backers
    
    # Total predicted amount
    predicted_amount = adjusted_amount + additional_amount
    
    # ==========================================
    # STEP 4: Apply Reality Checks
    # ==========================================
    
    # Minimum viable amount
    min_amount = 1000  # At least ₹1000
    if predicted_amount < min_amount:
        predicted_amount = min_amount
    
    # Cap at realistic maximum (typically 150-200% of goal for successful campaigns)
    max_realistic = goal_inr * 2.5
    if predicted_amount > max_realistic:
        predicted_amount = max_realistic
    
    # Apply platform fees
    platform_fee_rate = PLATFORM_FEES.get(platform, 0.05)
    net_amount = predicted_amount * (1 - platform_fee_rate)
    
    # Calculate success probability
    success_ratio = predicted_amount / goal_inr if goal_inr > 0 else 0
    if success_ratio >= 1.0:
        success_probability = min(95, 60 + (success_ratio * 20))
    else:
        success_probability = success_ratio * 60
    
    # Convert back to original currency if needed
    final_amount = predicted_amount / conversion_rate
    net_amount_converted = net_amount / conversion_rate
    
    return {
        'predicted_amount': round(final_amount, 2),
        'predicted_amount_inr': round(predicted_amount, 2),
        'net_amount': round(net_amount_converted, 2),
        'net_amount_inr': round(net_amount, 2),
        'goal_amount': goal_amount,
        'goal_amount_inr': round(goal_inr, 2),
        'success_probability': round(success_probability, 1),
        'total_multiplier': round(total_multiplier, 3),
        'base_pledge': base_pledge,
        'estimated_backers': num_backers + additional_backers,
        'platform_fee_percent': platform_fee_rate * 100,
        'multipliers': {
            'duration': round(duration_mult, 2),
            'social_media': round(social_mult, 2),
            'email_list': round(email_mult, 2),
            'updates': round(updates_mult, 2),
            'media_content': round(media_mult, 2),
            'experience': round(exp_mult, 2),
            'previous_campaigns': round(prev_mult, 2),
            'city_tier': round(city_mult, 2),
            'urgency': round(urgency_mult, 2),
            'platform': round(platform_mult, 2)
        }
    }

def generate_recommendations(result, category, has_video, has_images, social_followers, num_updates):
    """Generate actionable recommendations based on the prediction"""
    recommendations = []
    
    # Success status
    success_ratio = result['predicted_amount'] / result['goal_amount'] if result['goal_amount'] > 0 else 0
    
    if success_ratio >= 1.0:
        recommendations.append({
            'type': 'success',
            'icon': '🎉',
            'message': f"Great news! You're likely to reach {round(success_ratio * 100)}% of your goal."
        })
    elif success_ratio >= 0.7:
        recommendations.append({
            'type': 'warning',
            'icon': '⚠️',
            'message': f"You may reach about {round(success_ratio * 100)}% of your goal. Consider the tips below."
        })
    else:
        recommendations.append({
            'type': 'alert',
            'icon': '🚨',
            'message': f"Your campaign may struggle. Expected to reach only {round(success_ratio * 100)}% of goal."
        })
    
    # Video recommendation
    if not has_video:
        recommendations.append({
            'type': 'tip',
            'icon': '🎥',
            'message': "Add a video! Campaigns with videos raise 4x more on average."
        })
    
    # Images recommendation
    if not has_images:
        recommendations.append({
            'type': 'tip',
            'icon': '📸',
            'message': "Add quality images to build trust and emotional connection."
        })
    
    # Social media recommendation
    if social_followers < 1000:
        recommendations.append({
            'type': 'tip',
            'icon': '📱',
            'message': "Build your social media presence. Share on WhatsApp groups, Facebook, and Instagram."
        })
    
    # Updates recommendation
    if num_updates < 5:
        recommendations.append({
            'type': 'tip',
            'icon': '📢',
            'message': "Plan regular updates (at least weekly) to keep backers engaged."
        })
    
    # Category-specific tips
    if category == 'Medical':
        recommendations.append({
            'type': 'tip',
            'icon': '🏥',
            'message': "Include medical documents and hospital bills for credibility."
        })
    elif category == 'Education':
        recommendations.append({
            'type': 'tip',
            'icon': '📚',
            'message': "Share admission letters, fee structures, and academic achievements."
        })
    elif category in ['Creative', 'Technology']:
        recommendations.append({
            'type': 'tip',
            'icon': '🎯',
            'message': "Offer rewards or early access to attract backers."
        })
    
    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract and validate inputs
        goal_amount = float(data.get('goalAmount', 0))
        currency = data.get('currency', 'INR')
        category = data.get('category', 'Other')
        num_backers = int(data.get('numBackers', 0))
        duration_days = int(data.get('durationDays', 30))
        social_followers = int(data.get('socialFollowers', 0))
        email_subscribers = int(data.get('emailSubscribers', 0))
        num_updates = int(data.get('numUpdates', 0))
        has_video = data.get('hasVideo', 'No') == 'Yes'
        has_images = data.get('hasImages', 'No') == 'Yes'
        owner_experience = int(data.get('ownerExperience', 0))
        previous_campaigns = int(data.get('previousCampaigns', 0))
        platform = data.get('platform', 'Other')
        city_tier = data.get('cityTier', 'Tier 2')
        cause_urgency = data.get('causeUrgency', 'Medium')
        
        # Calculate prediction
        result = calculate_realistic_prediction(
            goal_amount=goal_amount,
            currency=currency,
            category=category,
            num_backers=num_backers,
            duration_days=duration_days,
            social_followers=social_followers,
            email_subscribers=email_subscribers,
            num_updates=num_updates,
            has_video=has_video,
            has_images=has_images,
            owner_experience=owner_experience,
            previous_campaigns=previous_campaigns,
            platform=platform,
            city_tier=city_tier,
            cause_urgency=cause_urgency
        )
        
        # Generate recommendations
        recommendations = generate_recommendations(
            result=result,
            category=category,
            has_video=has_video,
            has_images=has_images,
            social_followers=social_followers,
            num_updates=num_updates
        )
        
        return jsonify({
            'success': True,
            'prediction': result['predicted_amount'],
            'prediction_inr': result['predicted_amount_inr'],
            'net_amount': result['net_amount'],
            'net_amount_inr': result['net_amount_inr'],
            'goal_amount': result['goal_amount'],
            'goal_amount_inr': result['goal_amount_inr'],
            'success_probability': result['success_probability'],
            'estimated_backers': result['estimated_backers'],
            'platform_fee_percent': result['platform_fee_percent'],
            'multipliers': result['multipliers'],
            'total_multiplier': result['total_multiplier'],
            'recommendations': recommendations,
            'currency': currency
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Return available categories and their average pledges"""
    return jsonify({
        'categories': list(CATEGORY_AVG_PLEDGE_INR.keys()),
        'avg_pledges': CATEGORY_AVG_PLEDGE_INR
    })

@app.route('/api/platforms', methods=['GET'])
def get_platforms():
    """Return available platforms and their fees"""
    return jsonify({
        'platforms': list(PLATFORM_FEES.keys()),
        'fees': PLATFORM_FEES
    })

if __name__ == '__main__':
    print("=" * 50)
    print("India Crowdfunding Calculator")
    print("Realistic algorithm-based prediction system")
    print("=" * 50)
    
    # Get port from environment variable (Render sets this) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Bind to 0.0.0.0 to be accessible externally
    app.run(host='0.0.0.0', port=port, debug=False)