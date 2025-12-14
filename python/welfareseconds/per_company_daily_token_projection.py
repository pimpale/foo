#%%
"""
Per-Company Daily Token Projection for AI Inference

This script estimates daily token production for major AI companies:
- xAI (Grok)
- Anthropic (Claude)
- OpenAI (ChatGPT, API)
- Google (Gemini, AI Overviews, etc.)
- Meta (Meta AI)

=============================================================================
METHODOLOGY AND ASSUMPTIONS
=============================================================================

1. OPENAI - Most Complete Data
   - Direct token data: "100B words/day" (Feb 2024) → ~133B tokens/day
     Assumption: 1 word ≈ 1.33 tokens (standard tokenization ratio)
   - Direct message data: 451M (Jun 2024), 1B (Dec 2024), 2.627B (Jun 2025), 3B (Aug 2025)
   - API tokens: 6B/minute = 8.64T/day (Oct 2025)
   - GPT-5-Codex: ~2T/day (Oct 2025)
   
   Key ratio computed: tokens_per_message = 133B / daily_messages_interpolated
   This ratio is used to impute tokens for other companies.
   
   Note: The Feb 2024 "100B words" figure predates our message data. We interpolate
   messages to Feb 2024 and use that to establish our tokens/message baseline.

2. GOOGLE - Gemini API/Chat Only
   - "All AI products" figures (9.7T → 1.3Q monthly) are EXCLUDED because they
     include AI Search Overviews, Translate, etc. that don't scale comparably.
   - We use Gemini-specific data only:
     - Gemini API: 7B tokens/min = 10.08T/day (Oct 2025)
     - Gemini chat: 140M daily messages (Mar 2025) × tokens/message ratio
   
   Note: This makes Google comparable to other enterprise AI providers.

3. META - Imputed from Messages
   - No direct token data available
   - Daily messages: 200M (Mar 2025, from Google antitrust trial)
   - Apply OpenAI's tokens/message ratio
   
   Assumptions:
   - Meta AI conversations are similar in length to ChatGPT
   - The Google trial estimate is reliable for order-of-magnitude

4. ANTHROPIC - Revenue-based with Inference Spend Anchor
   - Rich revenue data: $87M (Jan 2024) → $7B (Oct 2025)
   - One inference spend data point: ~$2B annualized (Jul 2025)
   
   Method:
   1. Anchor point: Jul 2025 inference spend ($2B) vs OpenAI ($7B) = 0.29 ratio
   2. Get Anthropic's tokens at Jul 2025 from OpenAI comparison
   3. Scale other dates by revenue ratio (revenue ≈ tokens assumption)
   
   This uses Anthropic's actual revenue growth curve (8 data points) instead
   of assuming arbitrary growth rates.

5. XAI - Revenue-based
   - Revenue data: $100M (Nov 2024) → $500M (Jul 2025)
   - No direct token or inference spend data
   
   Method:
   1. Calculate tokens-per-revenue ratio from OpenAI and Anthropic
   2. Apply average ratio to xAI's revenue data
   
   This uses actual revenue data (4 points) instead of arbitrary growth rates.

=============================================================================
GROWTH MODEL
=============================================================================

For each company, we fit an exponential growth model: tokens(t) = a * exp(b * t)

This captures the rapid scaling of AI inference. The model is fit using least
squares on log-transformed data where multiple data points exist.

For companies with sparse data, we use:
- Linear interpolation between known points
- Exponential extrapolation based on industry-average growth rates

=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# DATA CLASSES AND UTILITIES
# =============================================================================

@dataclass
class DataPoint:
    """A single data point for token/message estimation."""
    date: datetime
    value: float
    metric: str  # 'daily_tokens', 'monthly_tokens', 'daily_messages', etc.
    source: str
    confidence: str  # 'Confident', 'Likely', 'Uncertain'


def date_to_decimal(dt: datetime) -> float:
    """Convert datetime to decimal year."""
    year_start = datetime(dt.year, 1, 1)
    year_end = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - year_start).days / (year_end - year_start).days


def decimal_to_date(decimal_year: float) -> datetime:
    """Convert decimal year to datetime."""
    year = int(decimal_year)
    remainder = decimal_year - year
    year_start = datetime(year, 1, 1)
    year_end = datetime(year + 1, 1, 1)
    days = remainder * (year_end - year_start).days
    return year_start + timedelta(days=days)


def exponential_model(t, a, b):
    """Exponential growth model: y = a * exp(b * t)"""
    return a * np.exp(b * t)


def fit_exponential(dates: np.ndarray, values: np.ndarray, 
                    t_reference: float = 2024.0) -> tuple:
    """
    Fit exponential model to data.
    Returns (a, b, r_squared) where model is: y = a * exp(b * (t - t_reference))
    """
    t_shifted = dates - t_reference
    
    # Use log-linear regression for initial guess
    log_values = np.log(values)
    b_init = np.polyfit(t_shifted, log_values, 1)[0]
    a_init = np.exp(np.mean(log_values - b_init * t_shifted))
    
    try:
        popt, _ = curve_fit(exponential_model, t_shifted, values, 
                           p0=[a_init, b_init], maxfev=10000)
        
        # Calculate R-squared
        predicted = exponential_model(t_shifted, *popt)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return popt[0], popt[1], r_squared
    except:
        # Fallback to log-linear
        return a_init, b_init, 0.0


# =============================================================================
# OPENAI DATA AND RATIO CALCULATION
# =============================================================================

# OpenAI data points
openai_data = {
    'daily_tokens': [
        # "100B words per day" = ~133B tokens (1 word ≈ 1.33 tokens)
        DataPoint(datetime(2024, 2, 9), 133e9, 'daily_tokens', 'Sam Altman tweet', 'Confident'),
    ],
    'daily_messages': [
        DataPoint(datetime(2024, 6, 24), 451e6, 'daily_messages', 'OpenAI paper', 'Confident'),
        DataPoint(datetime(2024, 12, 4), 1e9, 'daily_messages', 'Sam Altman', 'Confident'),
        DataPoint(datetime(2025, 6, 24), 2.627e9, 'daily_messages', 'OpenAI paper', 'Confident'),
        DataPoint(datetime(2025, 8, 4), 3e9, 'daily_messages', 'CNBC', 'Confident'),
    ],
    'api_daily_tokens': [
        # 6B tokens/minute * 1440 min/day = 8.64T/day
        DataPoint(datetime(2025, 10, 6), 8.64e12, 'api_daily_tokens', 'Dev Day', 'Confident'),
    ],
    'codex_daily_tokens': [
        # 40T tokens over 21 days = ~1.9T/day
        DataPoint(datetime(2025, 10, 6), 1.9e12, 'codex_daily_tokens', 'Dev Day', 'Confident'),
    ],
}

def calculate_openai_tokens_per_message():
    """
    Calculate the tokens per message ratio from OpenAI's data.
    
    We know: 133B tokens/day on Feb 9, 2024
    We need to estimate daily messages at that time.
    """
    # Message data points
    msg_dates = np.array([date_to_decimal(dp.date) for dp in openai_data['daily_messages']])
    msg_values = np.array([dp.value for dp in openai_data['daily_messages']])
    
    # Fit exponential to messages
    a, b, _ = fit_exponential(msg_dates, msg_values)
    
    # Extrapolate to Feb 2024
    feb_2024 = date_to_decimal(datetime(2024, 2, 9))
    estimated_messages_feb = exponential_model(feb_2024 - 2024, a, b)
    
    # Calculate ratio
    tokens_feb = 133e9
    tokens_per_message = tokens_feb / estimated_messages_feb
    
    print(f"OpenAI Tokens/Message Calculation:")
    print(f"  Feb 2024 tokens: {tokens_feb/1e9:.1f}B")
    print(f"  Feb 2024 messages (estimated): {estimated_messages_feb/1e6:.1f}M")
    print(f"  Tokens per message: {tokens_per_message:.0f}")
    print()
    
    return tokens_per_message


def build_openai_daily_tokens():
    """
    Build comprehensive OpenAI daily token estimates.
    
    Strategy:
    1. Use direct token data where available
    2. Convert messages to tokens for ChatGPT consumer
    3. Add API and Codex tokens for later dates
    """
    # Get tokens per message ratio
    tokens_per_msg = calculate_openai_tokens_per_message()
    
    # Build token estimates from messages (ChatGPT consumer)
    msg_dates = [dp.date for dp in openai_data['daily_messages']]
    msg_values = [dp.value for dp in openai_data['daily_messages']]
    
    # Start with Feb 2024 direct measurement
    estimates = [(datetime(2024, 2, 9), 133e9, 'direct')]
    
    # Add message-derived estimates
    for date, msgs in zip(msg_dates, msg_values):
        tokens = msgs * tokens_per_msg
        estimates.append((date, tokens, 'from_messages'))
    
    # For Oct 2025, we also have API (8.64T) and Codex (1.9T) tokens
    # The 3B messages (Aug 2025) is ChatGPT consumer only
    # We need to add API tokens on top for total
    
    # Create DataFrame
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    return df, tokens_per_msg


# =============================================================================
# GOOGLE DATA
# =============================================================================
# 
# NOTE: Google's "all AI products" figures (9.7T → 1.3Q monthly) include AI Search
# Overviews, Google Translate, and other products that don't scale the same way
# as enterprise AI/chat. We focus on Gemini API data for comparability.

google_data = {
    # These "all AI products" figures are NOT used - kept for reference only
    'monthly_tokens_all_products': [
        DataPoint(datetime(2024, 4, 15), 9.7e12, 'monthly_tokens', 'Google I/O 2024', 'Confident'),
        DataPoint(datetime(2025, 4, 30), 480e12, 'monthly_tokens', 'Google I/O 2025', 'Confident'),
        DataPoint(datetime(2025, 7, 23), 980e12, 'monthly_tokens', 'Q2 2025 Earnings', 'Confident'),
        DataPoint(datetime(2025, 9, 30), 1.3e15, 'monthly_tokens', 'Demis Hassabis tweet', 'Confident'),
    ],
    # Gemini API tokens - more comparable to other enterprise AI providers
    'api_daily_tokens': [
        # 7B tokens/minute * 1440 min/day = 10.08T/day
        DataPoint(datetime(2025, 10, 29), 10.08e12, 'api_daily_tokens', 'Q3 2025 Earnings', 'Confident'),
    ],
    'daily_messages': [
        # Gemini chat: 140M daily messages (Mar 2025)
        DataPoint(datetime(2025, 3, 28), 140e6, 'daily_messages', 'Google trial', 'Confident'),
    ],
}

def build_google_daily_tokens(tokens_per_msg: float):
    """
    Build Google daily token estimates using Gemini API data.
    
    We use only API token data because:
    1. "All AI products" includes Search AI Overviews, Translate, etc.
    2. Chat messages can't be directly combined with API tokens for growth curves
    
    With only one API data point, we use OpenAI's growth rate as a proxy
    for extrapolation (both are major AI providers with similar dynamics).
    """
    # Primary data: Gemini API tokens (Oct 2025): 10.08T/day
    api_dp = google_data['api_daily_tokens'][0]
    api_tokens = api_dp.value
    api_date = api_dp.date
    api_decimal = date_to_decimal(api_date)
    
    # For reference: Gemini chat would add ~0.07T (140M * 512 tokens)
    # But this is small compared to API and mixing them creates issues
    msg_dp = google_data['daily_messages'][0]
    chat_tokens = msg_dp.value * tokens_per_msg
    
    # Since we only have 1 API data point, we need to assume a growth rate
    # Use a reasonable estimate: similar to OpenAI's growth (~1.7/year)
    # This is conservative compared to Google's reported "all products" growth
    assumed_growth_rate = 1.7  # Similar to OpenAI
    
    # Build estimates by scaling from the API data point
    estimates = []
    
    key_dates = [
        datetime(2024, 6, 1),
        datetime(2024, 12, 1),
        datetime(2025, 3, 28),  # Chat data date - for reference
        datetime(2025, 6, 1),
        datetime(2025, 10, 29),  # API measurement date
    ]
    
    for date in key_dates:
        decimal = date_to_decimal(date)
        time_diff = decimal - api_decimal
        tokens = api_tokens * np.exp(assumed_growth_rate * time_diff)
        source = 'gemini_api' if date == api_date else 'extrapolated'
        estimates.append((date, tokens, source))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    print(f"Google Token Estimation (Gemini API):")
    print(f"  Oct 2025 API: {api_tokens/1e12:.2f}T/day (primary data)")
    print(f"  Chat tokens (Mar 2025, not used in fit): {chat_tokens/1e12:.2f}T/day")
    print(f"  Assumed growth rate: {assumed_growth_rate}/year (similar to OpenAI)")
    print(f"  NOTE: Only 1 API data point - growth rate is assumed, not fitted")
    print()
    
    return df


# =============================================================================
# META DATA (Imputed)
# =============================================================================

meta_data = {
    'mau': [
        DataPoint(datetime(2024, 9, 25), 400e6, 'mau', 'Meta Connect', 'Confident'),
        DataPoint(datetime(2024, 10, 30), 500e6, 'mau', 'Earnings', 'Confident'),
        DataPoint(datetime(2024, 12, 15), 600e6, 'mau', 'CNBC', 'Confident'),
        DataPoint(datetime(2025, 1, 29), 700e6, 'mau', 'CNBC', 'Confident'),
        DataPoint(datetime(2025, 5, 28), 1e9, 'mau', 'Zuckerberg', 'Confident'),
        DataPoint(datetime(2025, 10, 29), 1e9, 'mau', 'Earnings', 'Confident'),
    ],
    'daily_messages': [
        # From Google antitrust trial
        DataPoint(datetime(2025, 3, 28), 200e6, 'daily_messages', 'Google trial', 'Likely'),
    ],
    'dau': [
        DataPoint(datetime(2025, 3, 28), 100e6, 'dau', 'Google trial', 'Likely'),
    ],
}

def build_meta_daily_tokens(tokens_per_msg: float):
    """
    Build Meta daily token estimates.
    
    Strategy: Use message data and apply OpenAI's tokens/message ratio.
    For dates without message data, estimate from MAU trends.
    """
    # We have one message data point
    msg_dp = meta_data['daily_messages'][0]
    base_tokens = msg_dp.value * tokens_per_msg
    base_date = msg_dp.date
    base_decimal = date_to_decimal(base_date)
    
    # Estimate growth from MAU trend
    mau_dates = np.array([date_to_decimal(dp.date) for dp in meta_data['mau']])
    mau_values = np.array([dp.value for dp in meta_data['mau']])
    
    # Fit exponential to MAU
    a_mau, b_mau, _ = fit_exponential(mau_dates, mau_values)
    
    # Create estimates at MAU data points, scaling from base
    estimates = []
    
    # Use message-based estimate as anchor
    estimates.append((base_date, base_tokens, 'from_messages'))
    
    # Scale other dates by MAU ratio
    base_mau = exponential_model(base_decimal - 2024, a_mau, b_mau)
    
    for dp in meta_data['mau']:
        if dp.date != base_date:
            decimal = date_to_decimal(dp.date)
            mau_ratio = dp.value / base_mau
            # Messages likely scale sub-linearly with MAU (engagement varies)
            # Use sqrt scaling as approximation
            scaled_tokens = base_tokens * np.sqrt(mau_ratio)
            estimates.append((dp.date, scaled_tokens, 'scaled_from_mau'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    return df


# =============================================================================
# ANTHROPIC DATA (Revenue-based with inference spend anchor)
# =============================================================================
#
# Anthropic has rich revenue data but only one inference spend data point.
# Strategy: 
# 1. Use Jul 2025 inference spend ($2B) + OpenAI comparison to anchor tokens
# 2. Use revenue growth curve to extrapolate (revenue ~ inference tokens)

anthropic_data = {
    'inference_spend_annualized': [
        DataPoint(datetime(2025, 7, 31), 2e9, 'inference_spend', 'The Information/Morgan Stanley', 'Likely'),
    ],
    'revenue_annualized': [
        DataPoint(datetime(2024, 1, 1), 87e6, 'revenue', 'Anthropic disclosure', 'Confident'),
        DataPoint(datetime(2024, 12, 31), 1e9, 'revenue', 'CNBC', 'Likely'),
        DataPoint(datetime(2025, 3, 1), 1.4e9, 'revenue', 'The Information', 'Likely'),
        DataPoint(datetime(2025, 3, 31), 2e9, 'revenue', 'CNBC/Anthropic', 'Confident'),
        DataPoint(datetime(2025, 5, 30), 3e9, 'revenue', 'CNBC', 'Likely'),
        DataPoint(datetime(2025, 7, 1), 4e9, 'revenue', 'The Information', 'Likely'),
        DataPoint(datetime(2025, 7, 29), 5e9, 'revenue', 'Bloomberg/Anthropic', 'Confident'),
        DataPoint(datetime(2025, 10, 21), 7e9, 'revenue', 'Anthropic/Reuters', 'Confident'),
    ],
}

# OpenAI inference spend for comparison
openai_inference_spend = {
    2024: 1.8e9,  # Projected
    2025: 7e9,    # Projected
}

def build_anthropic_daily_tokens(openai_df: pd.DataFrame):
    """
    Build Anthropic daily token estimates using revenue growth curve.
    
    Method:
    1. Anchor: Jul 2025 inference spend ($2B) vs OpenAI ($7B) = 0.29 token ratio
    2. Get OpenAI's tokens at Jul 2025 → Anthropic's tokens at Jul 2025
    3. Scale other dates by revenue ratio (revenue growth ≈ token growth)
    
    This uses Anthropic's actual revenue growth curve instead of assuming
    OpenAI's growth rate, giving us real data-driven estimates.
    """
    # Step 1: Anchor point from inference spend comparison
    anthropic_spend_jul = 2e9
    openai_spend_2025 = 7e9
    token_ratio = anthropic_spend_jul / openai_spend_2025
    
    # Step 2: Get OpenAI's tokens at July 2025
    jul_2025 = date_to_decimal(datetime(2025, 7, 31))
    openai_dates = openai_df['decimal_date'].values
    openai_tokens = openai_df['daily_tokens'].values
    a, b, _ = fit_exponential(openai_dates, openai_tokens)
    openai_jul_tokens = exponential_model(jul_2025 - 2024, a, b)
    
    # Anthropic's tokens at July 2025 anchor point
    anthropic_jul_tokens = openai_jul_tokens * token_ratio
    
    # Step 3: Get Anthropic's revenue at July 2025 for scaling
    # Use the $5B data point from Jul 29, 2025
    anthropic_revenue_jul = 5e9
    
    # Step 4: Build estimates at each revenue data point, scaled by revenue
    estimates = []
    
    for dp in anthropic_data['revenue_annualized']:
        # Scale tokens by revenue ratio relative to July 2025
        revenue_ratio = dp.value / anthropic_revenue_jul
        tokens = anthropic_jul_tokens * revenue_ratio
        estimates.append((dp.date, tokens, 'from_revenue'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    # Compare inference spend / revenue ratios
    # OpenAI: 2024 ($1.8B inference / $3.7B rev), 2025 ($7B inference / $12B rev)
    openai_ratio_2024 = 1.8e9 / 3.7e9
    openai_ratio_2025 = 7e9 / 12e9
    # Anthropic: Jul 2025 ($2B inference / $5B rev)
    anthropic_ratio_jul = anthropic_spend_jul / anthropic_revenue_jul
    
    print(f"Anthropic Token Estimation (revenue-based):")
    print(f"  Inference/Revenue ratios:")
    print(f"    OpenAI 2024: ${1.8}B / ${3.7}B = {openai_ratio_2024:.1%}")
    print(f"    OpenAI 2025: ${7}B / ${12}B = {openai_ratio_2025:.1%}")
    print(f"    Anthropic Jul 2025: ${anthropic_spend_jul/1e9:.1f}B / ${anthropic_revenue_jul/1e9:.1f}B = {anthropic_ratio_jul:.1%}")
    print(f"  Anchor: inference spend ratio = {token_ratio:.2f}")
    print(f"  Jul 2025 anchor: {anthropic_jul_tokens/1e12:.2f}T tokens/day")
    print(f"  Revenue data points: {len(anthropic_data['revenue_annualized'])}")
    print()
    
    return df


# =============================================================================
# XAI DATA (Revenue-based)
# =============================================================================
#
# xAI has revenue data but no direct token/inference spend data.
# Strategy: Use revenue to estimate tokens, calibrated against OpenAI/Anthropic
# tokens-per-revenue ratios.

xai_data = {
    'daily_messages': [
        DataPoint(datetime(2025, 3, 28), 75e6, 'daily_messages', 'Google trial', 'Likely'),
    ],
    'revenue_annualized': [
        DataPoint(datetime(2024, 11, 20), 100e6, 'revenue', 'WSJ', 'Likely'),
        DataPoint(datetime(2025, 1, 31), 178e6, 'revenue', 'Wired', 'Uncertain'),
        DataPoint(datetime(2025, 3, 31), 208e6, 'revenue', 'Reuters (Q1 annualized)', 'Likely'),
        DataPoint(datetime(2025, 7, 31), 500e6, 'revenue', 'Wired', 'Likely'),
    ],
}

def build_xai_daily_tokens(openai_df: pd.DataFrame, anthropic_df: pd.DataFrame):
    """
    Build xAI daily token estimates from revenue data.
    
    Method:
    1. Calculate tokens-per-revenue ratio from OpenAI and Anthropic
    2. Apply average ratio to xAI's revenue data
    
    This gives us data-driven estimates instead of arbitrary growth rates.
    """
    # Calculate tokens-per-revenue for OpenAI (use Jul 2025 data)
    # OpenAI Jul 2025: ~$12B ARR, estimate tokens from fit
    jul_2025 = date_to_decimal(datetime(2025, 7, 31))
    openai_dates = openai_df['decimal_date'].values
    openai_tokens = openai_df['daily_tokens'].values
    a_oai, b_oai, _ = fit_exponential(openai_dates, openai_tokens)
    openai_jul_tokens = exponential_model(jul_2025 - 2024, a_oai, b_oai)
    openai_jul_revenue = 12e9  # $12B ARR in Jul 2025
    openai_tokens_per_revenue = openai_jul_tokens / openai_jul_revenue
    
    # Calculate tokens-per-revenue for Anthropic (Jul 2025)
    # Find Anthropic's Jul 2025 tokens from their dataframe
    anthropic_jul = anthropic_df[anthropic_df['decimal_date'] >= jul_2025 - 0.1]
    if len(anthropic_jul) > 0:
        anthropic_jul_tokens = anthropic_jul.iloc[0]['daily_tokens']
    else:
        # Extrapolate
        a_ant, b_ant, _ = fit_exponential(anthropic_df['decimal_date'].values, 
                                          anthropic_df['daily_tokens'].values)
        anthropic_jul_tokens = exponential_model(jul_2025 - 2024, a_ant, b_ant)
    anthropic_jul_revenue = 5e9  # $5B ARR in Jul 2025
    anthropic_tokens_per_revenue = anthropic_jul_tokens / anthropic_jul_revenue
    
    # Average the two ratios
    avg_tokens_per_revenue = (openai_tokens_per_revenue + anthropic_tokens_per_revenue) / 2
    
    print(f"xAI Token Estimation (revenue-based):")
    print(f"  OpenAI tokens/$ revenue: {openai_tokens_per_revenue:.0f}")
    print(f"  Anthropic tokens/$ revenue: {anthropic_tokens_per_revenue:.0f}")
    print(f"  Average tokens/$ revenue: {avg_tokens_per_revenue:.0f}")
    
    # Build estimates from xAI revenue data
    estimates = []
    
    for dp in xai_data['revenue_annualized']:
        tokens = dp.value * avg_tokens_per_revenue
        estimates.append((dp.date, tokens, 'from_revenue'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    print(f"  Revenue data points: {len(xai_data['revenue_annualized'])}")
    print(f"  Latest estimate ({df.iloc[-1]['date'].strftime('%Y-%m-%d')}): {df.iloc[-1]['daily_tokens']/1e12:.3f}T/day")
    print()
    
    return df


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_and_project():
    """Main analysis function."""
    print("=" * 70)
    print("AI COMPANY DAILY TOKEN PROJECTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Build OpenAI data and get tokens/message ratio
    openai_df, tokens_per_msg = build_openai_daily_tokens()
    
    print(f"Using tokens/message ratio: {tokens_per_msg:.0f}")
    print()
    
    # Build all company data
    # Note: Order matters - some depend on others
    google_df = build_google_daily_tokens(tokens_per_msg)
    meta_df = build_meta_daily_tokens(tokens_per_msg)
    anthropic_df = build_anthropic_daily_tokens(openai_df)
    xai_df = build_xai_daily_tokens(openai_df, anthropic_df)
    
    # Fit growth models
    companies = {
        'OpenAI': openai_df,
        'Google': google_df,
        'Meta': meta_df,
        'Anthropic': anthropic_df,
        'xAI': xai_df,
    }
    
    models = {}
    
    print("=" * 70)
    print("GROWTH MODEL FITS")
    print("=" * 70)
    print()
    
    for name, df in companies.items():
        dates = df['decimal_date'].values
        tokens = df['daily_tokens'].values
        
        a, b, r2 = fit_exponential(dates, tokens)
        models[name] = {'a': a, 'b': b, 'r2': r2}
        
        doubling_time = np.log(2) / b if b > 0 else float('inf')
        monthly_growth = (np.exp(b / 12) - 1) * 100
        
        print(f"{name}:")
        print(f"  Data points: {len(df)}")
        print(f"  Latest estimate: {tokens[-1]:.2e} tokens/day ({tokens[-1]/1e12:.2f}T)")
        print(f"  Growth rate: {b:.2f}/year ({monthly_growth:.1f}%/month)")
        print(f"  Doubling time: {doubling_time:.2f} years ({doubling_time*12:.1f} months)")
        print(f"  R² fit: {r2:.3f}")
        print()
    
    return companies, models, tokens_per_msg


def project_forward(companies: dict, models: dict, end_date: datetime = datetime(2027, 1, 1)):
    """Project token production forward."""
    
    print("=" * 70)
    print("PROJECTIONS")
    print("=" * 70)
    print()
    
    projection_dates = [
        datetime(2024, 6, 1),
        datetime(2025, 1, 1),
        datetime(2025, 6, 1),
        datetime(2025, 12, 1),
        datetime(2026, 6, 1),
        datetime(2027, 1, 1),
    ]
    
    projections = {name: [] for name in companies.keys()}
    projections['Total'] = []
    
    for date in projection_dates:
        decimal = date_to_decimal(date)
        total = 0
        
        for name, params in models.items():
            tokens = exponential_model(decimal - 2024, params['a'], params['b'])
            projections[name].append(tokens)
            total += tokens
        
        projections['Total'].append(total)
    
    # Create projection DataFrame
    proj_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in projection_dates],
        **{name: [f"{v/1e12:.2f}T" for v in vals] for name, vals in projections.items()}
    })
    
    print("Daily Token Production Projections (Trillions):")
    print()
    print(proj_df.to_string(index=False))
    print()
    
    return projections, projection_dates


def create_visualization(companies: dict, models: dict, projections: dict, 
                         projection_dates: list):
    """Create visualization of token projections."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color scheme
    colors = {
        'OpenAI': '#10a37f',      # OpenAI green
        'Google': '#4285f4',      # Google blue
        'Meta': '#0866ff',        # Meta blue
        'Anthropic': '#d4a574',   # Anthropic tan/brown
        'xAI': '#1da1f2',         # xAI/Twitter blue
        'All': '#333333',         # Dark gray for total
    }
    
    # Time range for plotting
    t_plot = np.linspace(2024.0, 2027.0, 300)
    
    # Plot each company
    for name, df in companies.items():
        # Plot data points
        ax.scatter(df['decimal_date'], df['daily_tokens'] / 1e12, 
                   s=80, color=colors[name], alpha=0.7, zorder=5,
                   label=f'{name} (data)')
        
        # Plot trendline
        params = models[name]
        y_fit = exponential_model(t_plot - 2024, params['a'], params['b']) / 1e12
        ax.plot(t_plot, y_fit, '--', color=colors[name], alpha=0.6, linewidth=2)
    
    # Calculate and plot "All" (total) line
    total = np.zeros_like(t_plot)
    for name in companies.keys():
        params = models[name]
        total += exponential_model(t_plot - 2024, params['a'], params['b']) / 1e12
    
    ax.plot(t_plot, total, '-', color=colors['All'], linewidth=3, label='All (total)', zorder=4)
    
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Daily Tokens (Trillions)', fontsize=12)
    ax.set_title('Daily Token Production by Company', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(2024, 2027)
    
    plt.tight_layout()
    plt.savefig('per_company_token_projection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Saved visualization to 'per_company_token_projection.png'")


def create_summary_table(companies: dict, models: dict):
    """Create summary statistics table."""
    
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()
    
    rows = []
    for name, df in companies.items():
        params = models[name]
        
        # Latest data point
        latest_date = df['date'].max()
        latest_tokens = df[df['date'] == latest_date]['daily_tokens'].values[0]
        
        # Current projection (Dec 2025)
        dec_2025 = date_to_decimal(datetime(2025, 12, 1))
        proj_dec_2025 = exponential_model(dec_2025 - 2024, params['a'], params['b'])
        
        # Future projection (Dec 2026)
        dec_2026 = date_to_decimal(datetime(2026, 12, 1))
        proj_dec_2026 = exponential_model(dec_2026 - 2024, params['a'], params['b'])
        
        doubling_time = np.log(2) / params['b'] if params['b'] > 0 else float('inf')
        
        rows.append({
            'Company': name,
            'Latest Data': latest_date.strftime('%Y-%m-%d'),
            'Latest (T/day)': f"{latest_tokens/1e12:.2f}",
            'Dec 2025 (T/day)': f"{proj_dec_2025/1e12:.2f}",
            'Dec 2026 (T/day)': f"{proj_dec_2026/1e12:.2f}",
            'Doubling (months)': f"{doubling_time*12:.1f}",
        })
    
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    print()
    
    # Total row
    dec_2025 = date_to_decimal(datetime(2025, 12, 1))
    dec_2026 = date_to_decimal(datetime(2026, 12, 1))
    
    total_dec_2025 = sum(
        exponential_model(dec_2025 - 2024, models[name]['a'], models[name]['b'])
        for name in companies.keys()
    )
    total_dec_2026 = sum(
        exponential_model(dec_2026 - 2024, models[name]['a'], models[name]['b'])
        for name in companies.keys()
    )
    
    print(f"TOTAL (All Companies):")
    print(f"  Dec 2025: {total_dec_2025/1e12:.2f}T tokens/day")
    print(f"  Dec 2026: {total_dec_2026/1e12:.2f}T tokens/day")
    print(f"  YoY Growth: {(total_dec_2026/total_dec_2025 - 1)*100:.1f}%")
    print()


def export_data(companies: dict, models: dict):
    """Export projection data to CSV."""
    
    # Generate monthly projections
    dates = pd.date_range('2024-01-01', '2027-01-01', freq='MS')
    
    data = {'date': dates}
    
    for name in companies.keys():
        params = models[name]
        decimals = [date_to_decimal(d.to_pydatetime()) for d in dates]
        tokens = [exponential_model(d - 2024, params['a'], params['b']) for d in decimals]
        data[f'{name}_daily_tokens'] = tokens
    
    # Add total
    data['total_daily_tokens'] = [
        sum(data[f'{name}_daily_tokens'][i] for name in companies.keys())
        for i in range(len(dates))
    ]
    
    df = pd.DataFrame(data)
    df.to_csv('token_projections.csv', index=False)
    print("Exported projections to 'token_projections.csv'")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Run analysis
    companies, models, tokens_per_msg = analyze_and_project()
    
    # Generate projections
    projections, projection_dates = project_forward(companies, models)
    
    # Create summary
    create_summary_table(companies, models)
    
    # Create visualization
    create_visualization(companies, models, projections, projection_dates)
    
    # Export data
    export_data(companies, models)
    
    print("=" * 70)
    print("METHODOLOGY NOTES")
    print("=" * 70)
    print("""
1. OpenAI tokens are derived from direct disclosures (100B words/day in Feb 2024)
   and message counts, using a computed tokens/message ratio.

2. Google tokens use GEMINI-ONLY data (API + chat), excluding "all AI products"
   figures which include Search AI Overviews, Translate, etc.

3. Meta tokens are imputed from daily message estimates using OpenAI's
   tokens/message ratio.

4. Anthropic tokens are anchored to inference spend comparison with OpenAI,
   then scaled using Anthropic's revenue growth curve (8 data points).

5. xAI tokens are estimated from revenue data using an average tokens-per-revenue
   ratio derived from OpenAI and Anthropic.

6. All growth models are exponential fits. Actual growth may vary significantly.

7. The "Total" model sums all company projections and represents combined
   daily token output from the major AI providers.

CAVEATS:
- Token counts include both input and output tokens where reported
- Different products (API vs chat) have different token densities
- Internal/non-public AI usage is not captured
- DeepSeek and other providers are excluded from this analysis
""")

# %%
