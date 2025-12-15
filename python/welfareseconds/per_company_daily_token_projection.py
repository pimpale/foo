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

1. OPENAI - Multiple Estimation Methods
   - Direct message data: 451M (Jun 2024), 1B (Dec 2024), 2.627B (Jun 2025), 3B (Aug 2025)
   - API tokens: 6B/minute = 8.64T/day (Oct 2025)
   - GPT-5-Codex: ~2T/day (Oct 2025)
   - Direct token data: "100B words/day" (Feb 2024) → ~133B tokens/day (Sam Altman tweet)
   
   Four OpenAI series are computed:
   a) ChatGPT: message counts × CHAT_TOKENS_PER_MSG constant
   b) API: direct API token measurements with growth extrapolation
   c) OpenAI (Inference): scaled by inference compute spend, anchored to Sam's tweet
   d) OpenAI (Revenue): scaled by revenue, anchored to Sam's tweet
   
   The Inference and Revenue series use data from ai_companies_compute_spend.csv
   and ai_companies_revenue.csv respectively, providing different views on growth.

2. GOOGLE - Gemini API/Chat Only
   - "All AI products" figures (9.7T → 1.3Q monthly) are EXCLUDED because they
     include AI Search Overviews, Translate, etc. that don't scale comparably.
   - We use Gemini-specific data only:
     - Gemini API: 7B tokens/min = 10.08T/day (Oct 2025)
     - Gemini chat: 140M daily messages (Mar 2025) × tokens/message ratio
   
   Note: This makes Google comparable to other enterprise AI providers.

3. META - MAU-based with messages/MAU ratio
   - Rich MAU data: 400M (Sep 2024) → 1B (May 2025, plateaued)
   - Daily messages: 200M (Mar 2025, from Google antitrust trial)
   
   Method:
   1. Interpolate MAU at Mar 2025 (~850M) from surrounding data points
   2. Compute messages/MAU ratio: 200M / 850M = 0.235 messages/MAU/day
   3. Apply ratio linearly to all MAU data points
   4. Convert messages to tokens using CHAT_TOKENS_PER_MSG constant

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

import os
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
# CONSTANTS
# =============================================================================

# Assumed tokens per chat message for consumer AI assistants
# This is a simplification - actual values vary by product and conversation length
CHAT_TOKENS_PER_MSG = 512

# Human tokens per second estimate
# Based on: 230 words per minute reading/speaking speed, 4/3 tokens per word
# (230 / 60) * (4/3) ≈ 5.11 tokens/second
HUMAN_TOKENS_PER_SECOND = (230 / 60) * (4 / 3)

# Industry-wide growth rate (per year) for token production
# Used for extrapolating single data point estimates
# This is calculated dynamically by averaging growth rates from companies with multiple data points
# (see calculate_industry_growth_rate() function)

# Date range for plotting and export (decimal years)
PLOT_BEGIN_DATE = 2024.0
PLOT_END_DATE = 2030.0

# Output folder for individual company plots
PLOTS_FOLDER = 'plots'


def tokens_per_revenue():
    """
    Calculate the ratio of daily tokens per dollar of annualized revenue.
    
    This is computed from the OpenAI anchor point:
    - Feb 2024: 133B tokens/day (Sam Altman "100B words/day" tweet)
    - Revenue at Feb 2024: interpolated from revenue data
    
    Returns:
        float: Daily tokens per dollar of annualized revenue
    
    Note: This ratio implies that revenue scales linearly with token production,
    which is a simplification. In reality, pricing, efficiency, and product mix
    all affect this relationship.
    """
    # Anchor point: Feb 2024 from Sam Altman tweet
    anchor_tokens = 133e9  # 133B tokens/day
    anchor_date = date_to_decimal(datetime(2024, 2, 9))
    
    # Get OpenAI revenue data points to interpolate anchor revenue
    rev_dates = np.array([date_to_decimal(dp.date) for dp in openai_data['revenue_annualized']])
    rev_values = np.array([dp.value for dp in openai_data['revenue_annualized']])
    
    # Fit exponential to interpolate anchor revenue
    a, b, _ = fit_exponential(rev_dates, rev_values)
    anchor_revenue = exponential_model(anchor_date - 2024, a, b)
    
    # Simple ratio: tokens per dollar of revenue
    return anchor_tokens / anchor_revenue

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
        # This is overall inference (ChatGPT + API combined)
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
    # Inference compute spend (from ai_companies_compute_spend.csv)
    'inference_spend_annualized': [
        DataPoint(datetime(2024, 12, 31), 1.8e9, 'inference_spend', 'The Information', 'Likely'),
        DataPoint(datetime(2025, 12, 31), 7e9, 'inference_spend', 'The Information', 'Likely'),
    ],
    # Revenue (from ai_companies_revenue.csv)
    'revenue_annualized': [
        DataPoint(datetime(2023, 12, 31), 2e9, 'revenue', 'FT', 'Likely'),
        DataPoint(datetime(2024, 6, 12), 3.4e9, 'revenue', 'The Information', 'Likely'),
        DataPoint(datetime(2024, 8, 15), 3.6e9, 'revenue', 'NYT', 'Likely'),
        DataPoint(datetime(2024, 9, 12), 4e9, 'revenue', 'WSJ', 'Likely'),
        DataPoint(datetime(2024, 12, 31), 5.5e9, 'revenue', 'CNBC', 'Likely'),
        DataPoint(datetime(2025, 6, 9), 10e9, 'revenue', 'CNBC', 'Confident'),
        DataPoint(datetime(2025, 7, 30), 12e9, 'revenue', 'CNBC/The Information', 'Confident'),
        DataPoint(datetime(2025, 8, 1), 13e9, 'revenue', 'NYT', 'Likely'),
    ],
}

def build_openai_chatgpt_tokens():
    """
    Build OpenAI ChatGPT (consumer) daily token estimates.
    
    Uses message data converted to tokens via CHAT_TOKENS_PER_MSG constant.
    """
    # Build estimates from message data
    estimates = []
    
    for dp in openai_data['daily_messages']:
        tokens = dp.value * CHAT_TOKENS_PER_MSG
        estimates.append((dp.date, tokens, 'from_messages'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    print(f"OpenAI ChatGPT Token Estimation:")
    print(f"  Using {CHAT_TOKENS_PER_MSG} tokens/message")
    print(f"  Message data points: {len(openai_data['daily_messages'])}")
    print()
    
    return df


def build_openai_api_tokens():
    """
    Build OpenAI API daily token estimates.
    
    Uses only the single direct API token measurement.
    Growth rate is applied later in model fitting using averaged industry growth rate.
    """
    api_dp = openai_data['api_daily_tokens'][0]
    api_tokens = api_dp.value
    api_date = api_dp.date
    
    # Single data point only
    estimates = [(api_date, api_tokens, 'api_direct')]
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    
    print(f"OpenAI API Token Estimation:")
    print(f"  {api_date.strftime('%Y-%m-%d')}: {api_tokens/1e12:.2f}T/day (direct measurement)")
    print(f"  NOTE: Single point - will use averaged industry growth rate for trendline")
    print()
    
    return df


def build_openai_inference_scaled_tokens():
    """
    Build OpenAI Combined daily token estimates scaled by INFERENCE SPEND.
    
    Method:
    1. Anchor: Feb 2024 "100B words/day" = 133B tokens/day (Sam Altman tweet)
    2. Interpolate inference spend at Feb 2024 from data points
    3. Scale tokens by inference spend ratio at each data point
    """
    # Anchor point: Feb 2024 from Sam Altman tweet
    anchor_tokens = 133e9  # 133B tokens/day
    anchor_date = date_to_decimal(datetime(2024, 2, 9))
    
    # Get inference spend data points
    spend_dates = np.array([date_to_decimal(dp.date) for dp in openai_data['inference_spend_annualized']])
    spend_values = np.array([dp.value for dp in openai_data['inference_spend_annualized']])
    
    # Fit exponential to interpolate anchor spend
    a, b, _ = fit_exponential(spend_dates, spend_values)
    anchor_spend = exponential_model(anchor_date - 2024, a, b)
    
    # Build estimates at each inference spend data point
    estimates = []
    
    for dp in openai_data['inference_spend_annualized']:
        # Scale tokens by spend ratio relative to anchor
        spend_ratio = dp.value / anchor_spend
        tokens = anchor_tokens * spend_ratio
        estimates.append((dp.date, tokens, 'from_inference_spend'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    # Calculate tokens per $ inference spend
    tokens_per_spend = anchor_tokens / anchor_spend
    
    print(f"OpenAI (Inference-Scaled) Token Estimation:")
    print(f"  Anchor: Feb 2024 = {anchor_tokens/1e9:.0f}B tokens/day (Sam Altman tweet)")
    print(f"  Anchor inference spend (interpolated): ${anchor_spend/1e9:.2f}B")
    print(f"  Tokens per $ inference: {tokens_per_spend:.0f}")
    print(f"  Inference data points: {len(openai_data['inference_spend_annualized'])}")
    print(f"  Latest ({df.iloc[-1]['date'].strftime('%Y-%m-%d')}): {df.iloc[-1]['daily_tokens']/1e12:.2f}T tokens/day")
    print()
    
    return df


def build_openai_revenue_scaled_tokens():
    """
    Build OpenAI Combined daily token estimates scaled by REVENUE.
    
    Uses tokens_per_revenue() ratio to convert revenue to tokens.
    """
    # Get the tokens-per-revenue ratio
    tpr = tokens_per_revenue()
    
    # Build estimates at each revenue data point
    estimates = []
    
    for dp in openai_data['revenue_annualized']:
        tokens = dp.value * tpr
        estimates.append((dp.date, tokens, 'from_revenue'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    print(f"OpenAI (Revenue-Scaled) Token Estimation:")
    print(f"  Using tokens_per_revenue = {tpr:.2f} tokens/day per $")
    print(f"  Revenue data points: {len(openai_data['revenue_annualized'])}")
    print(f"  Latest ({df.iloc[-1]['date'].strftime('%Y-%m-%d')}): {df.iloc[-1]['daily_tokens']/1e12:.2f}T tokens/day")
    print()
    
    return df


# =============================================================================
# GOOGLE DATA - Split into Gemini Assistant and Gemini API
# =============================================================================

google_data = {
    # "All AI products" figures - NOT used (includes Search AI Overviews, Translate, etc.)
    'monthly_tokens_all_products': [
        DataPoint(datetime(2024, 4, 15), 9.7e12, 'monthly_tokens', 'Google I/O 2024', 'Confident'),
        DataPoint(datetime(2025, 4, 30), 480e12, 'monthly_tokens', 'Google I/O 2025', 'Confident'),
        DataPoint(datetime(2025, 7, 23), 980e12, 'monthly_tokens', 'Q2 2025 Earnings', 'Confident'),
        DataPoint(datetime(2025, 9, 30), 1.3e15, 'monthly_tokens', 'Demis Hassabis tweet', 'Confident'),
    ],
    # Gemini Assistant MAU data
    'gemini_mau': [
        DataPoint(datetime(2025, 3, 15), 350e6, 'mau', 'The Information', 'Confident'),
        DataPoint(datetime(2025, 5, 20), 400e6, 'mau', 'Google I/O', 'Confident'),
        DataPoint(datetime(2025, 7, 23), 450e6, 'mau', 'Q2 Earnings', 'Confident'),
        DataPoint(datetime(2025, 10, 29), 650e6, 'mau', 'Q3 Earnings', 'Confident'),
    ],
    # Gemini Assistant messages (Mar 2025)
    'gemini_daily_messages': [
        DataPoint(datetime(2025, 3, 28), 140e6, 'daily_messages', 'Google trial', 'Confident'),
    ],
    # Gemini API tokens
    'api_daily_tokens': [
        # 7B tokens/minute * 1440 min/day = 10.08T/day
        DataPoint(datetime(2025, 10, 29), 10.08e12, 'api_daily_tokens', 'Q3 2025 Earnings', 'Confident'),
    ],
}

def build_gemini_assistant_tokens():
    """
    Build Gemini Assistant (consumer) daily token estimates.
    
    Method: Use MAU data with messages/MAU ratio from Google trial.
    """
    # Compute messages/MAU ratio from Mar 2025 data
    # 140M messages / 350M MAU = 0.4 messages/MAU/day
    messages_mar = 140e6
    mau_mar = 350e6
    messages_per_mau = messages_mar / mau_mar
    
    estimates = []
    
    for dp in google_data['gemini_mau']:
        daily_messages = dp.value * messages_per_mau
        daily_tokens = daily_messages * CHAT_TOKENS_PER_MSG
        estimates.append((dp.date, daily_tokens, 'from_mau'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    print(f"Gemini Assistant Token Estimation (MAU-based):")
    print(f"  Messages/MAU ratio: {messages_per_mau:.3f} (from Mar 2025: {messages_mar/1e6:.0f}M msgs / {mau_mar/1e6:.0f}M MAU)")
    print(f"  Tokens/message: {CHAT_TOKENS_PER_MSG}")
    print(f"  MAU data points: {len(google_data['gemini_mau'])}")
    print(f"  Latest ({df.iloc[-1]['date'].strftime('%Y-%m-%d')}): {df.iloc[-1]['daily_tokens']/1e12:.2f}T tokens/day")
    print()
    
    return df


def build_gemini_api_tokens():
    """
    Build Gemini API daily token estimates.
    
    Uses only the single direct API token measurement.
    Growth rate is applied later in model fitting using averaged industry growth rate.
    """
    api_dp = google_data['api_daily_tokens'][0]
    api_tokens = api_dp.value
    api_date = api_dp.date
    
    # Single data point only
    estimates = [(api_date, api_tokens, 'api_direct')]
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    
    print(f"Gemini API Token Estimation:")
    print(f"  {api_date.strftime('%Y-%m-%d')}: {api_tokens/1e12:.2f}T/day (direct measurement)")
    print(f"  NOTE: Single point - will use averaged industry growth rate for trendline")
    print()
    
    return df


# =============================================================================
# META DATA (MAU-based with messages/MAU ratio from Google trial)
# =============================================================================
#
# Strategy: Use Google trial data to compute messages/MAU ratio, then apply
# linearly to all MAU data points.

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

def build_meta_daily_tokens():
    """
    Build Meta daily token estimates using messages/MAU ratio.
    
    Strategy:
    1. Interpolate MAU at Mar 2025 (Google trial date)
    2. Compute messages/MAU ratio from Google trial data
    3. Apply ratio linearly to all MAU data points
    4. Convert messages to tokens
    """
    # Google trial data: 200M messages on Mar 28, 2025
    messages_mar_2025 = 200e6
    
    # Interpolate MAU at Mar 2025 (between Jan 700M and May 1B)
    # Jan 29 → Mar 28 is ~2 months, Jan 29 → May 28 is ~4 months
    # Linear interpolation: 700M + (1B - 700M) * (2/4) = 850M
    mau_mar_2025 = 850e6
    
    # Compute messages per MAU per day (linear relationship)
    messages_per_mau = messages_mar_2025 / mau_mar_2025
    
    print(f"Meta Token Estimation (MAU-based):")
    print(f"  Google trial (Mar 2025): {messages_mar_2025/1e6:.0f}M messages/day")
    print(f"  Interpolated MAU (Mar 2025): {mau_mar_2025/1e6:.0f}M")
    print(f"  Messages per MAU per day: {messages_per_mau:.4f}")
    print(f"  Tokens/message: {CHAT_TOKENS_PER_MSG}")
    
    # Apply ratio to all MAU data points
    estimates = []
    
    for dp in meta_data['mau']:
        daily_messages = dp.value * messages_per_mau
        daily_tokens = daily_messages * CHAT_TOKENS_PER_MSG
        estimates.append((dp.date, daily_tokens, 'from_mau'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    print(f"  MAU data points: {len(meta_data['mau'])}")
    print(f"  Latest ({df.iloc[-1]['date'].strftime('%Y-%m-%d')}): {df.iloc[-1]['daily_tokens']/1e12:.2f}T tokens/day")
    print()
    
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

def build_anthropic_daily_tokens():
    """
    Build Anthropic daily token estimates using revenue and tokens_per_revenue.
    
    Uses the same tokens-per-revenue ratio as OpenAI.
    """
    # Get the tokens-per-revenue ratio
    tpr = tokens_per_revenue()
    
    # Build estimates at each revenue data point
    estimates = []
    
    for dp in anthropic_data['revenue_annualized']:
        tokens = dp.value * tpr
        estimates.append((dp.date, tokens, 'from_revenue'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    print(f"Anthropic Token Estimation (revenue-based):")
    print(f"  Using tokens_per_revenue = {tpr:.2f} tokens/day per $")
    print(f"  Revenue data points: {len(anthropic_data['revenue_annualized'])}")
    print(f"  Latest ({df.iloc[-1]['date'].strftime('%Y-%m-%d')}): {df.iloc[-1]['daily_tokens']/1e12:.2f}T tokens/day")
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

def build_xai_daily_tokens():
    """
    Build xAI daily token estimates from revenue data.
    
    Uses tokens_per_revenue() ratio to convert revenue to tokens.
    """
    # Get the tokens-per-revenue ratio
    tpr = tokens_per_revenue()
    
    # Build estimates from xAI revenue data
    estimates = []
    
    for dp in xai_data['revenue_annualized']:
        tokens = dp.value * tpr
        estimates.append((dp.date, tokens, 'from_revenue'))
    
    df = pd.DataFrame(estimates, columns=['date', 'daily_tokens', 'source'])
    df['decimal_date'] = df['date'].apply(date_to_decimal)
    df = df.sort_values('decimal_date')
    
    print(f"xAI Token Estimation (revenue-based):")
    print(f"  Using tokens_per_revenue = {tpr:.2f} tokens/day per $")
    print(f"  Revenue data points: {len(xai_data['revenue_annualized'])}")
    print(f"  Latest estimate ({df.iloc[-1]['date'].strftime('%Y-%m-%d')}): {df.iloc[-1]['daily_tokens']/1e12:.3f}T/day")
    print()
    
    return df


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def calculate_industry_growth_rate(companies: dict) -> float:
    """
    Calculate industry-wide growth rate by averaging growth rates from companies
    with multiple data points.
    
    Returns:
        float: Average growth rate (per year) across companies with fitted models
    """
    growth_rates = []
    
    for name, df in companies.items():
        if len(df) > 1:
            dates = df['decimal_date'].values
            tokens = df['daily_tokens'].values
            _, b, _ = fit_exponential(dates, tokens)
            growth_rates.append(b)
    
    avg_growth_rate = np.mean(growth_rates)
    return avg_growth_rate


def analyze_and_project():
    """Main analysis function."""
    print("=" * 70)
    print("AI COMPANY DAILY TOKEN PROJECTION ANALYSIS")
    print("=" * 70)
    print()
    print(f"Using CHAT_TOKENS_PER_MSG = {CHAT_TOKENS_PER_MSG}")
    print()
    
    # Build OpenAI ChatGPT data
    openai_chatgpt_df = build_openai_chatgpt_tokens()
    
    # Build OpenAI API data
    openai_api_df = build_openai_api_tokens()
    
    # Build OpenAI Combined - two methods using Sam's tweet as anchor
    openai_inference_df = build_openai_inference_scaled_tokens()
    openai_revenue_df = build_openai_revenue_scaled_tokens()
    
    # Build Google Gemini data (split into Assistant and API)
    gemini_assistant_df = build_gemini_assistant_tokens()
    gemini_api_df = build_gemini_api_tokens()
    
    # Build other company data
    meta_df = build_meta_daily_tokens()
    # Revenue-based estimates using teratokens_per_bil_revenue()
    anthropic_df = build_anthropic_daily_tokens()
    xai_df = build_xai_daily_tokens()
    
    # Fit growth models - now with split categories
    companies = {
        'OpenAI ChatGPT': openai_chatgpt_df,
        'OpenAI API': openai_api_df,
        'OpenAI (Inference)': openai_inference_df,
        'OpenAI (Revenue)': openai_revenue_df,
        'Gemini Assistant': gemini_assistant_df,
        'Gemini API': gemini_api_df,
        'Meta': meta_df,
        'Anthropic': anthropic_df,
        'xAI': xai_df,
    }
    
    # Calculate industry growth rate by averaging growth rates from multi-point companies
    industry_growth_rate = calculate_industry_growth_rate(companies)
    
    print("=" * 70)
    print("INDUSTRY GROWTH RATE CALCULATION")
    print("=" * 70)
    print()
    print(f"Averaging growth rates from companies with multiple data points:")
    for name, df in companies.items():
        if len(df) > 1:
            dates = df['decimal_date'].values
            tokens = df['daily_tokens'].values
            _, b, _ = fit_exponential(dates, tokens)
            print(f"  {name}: {b:.2f}/year")
    print(f"\nAverage industry growth rate: {industry_growth_rate:.2f}/year")
    print()
    
    models = {}
    
    print("=" * 70)
    print("GROWTH MODEL FITS")
    print("=" * 70)
    print()
    
    for name, df in companies.items():
        dates = df['decimal_date'].values
        tokens = df['daily_tokens'].values
        
        if len(df) == 1:
            # Single data point: use industry growth rate (averaged from other companies)
            # Calculate 'a' so that model passes through the single point
            # Model: tokens = a * exp(b * (t - 2024))
            # So: a = tokens / exp(b * (t - 2024))
            b = industry_growth_rate
            t = dates[0]
            a = tokens[0] / np.exp(b * (t - 2024))
            r2 = float('nan')  # Can't compute R² with 1 point
            
            print(f"{name}:")
            print(f"  Data points: {len(df)} (using averaged industry growth rate)")
            print(f"  Estimate: {tokens[0]:.2e} tokens/day ({tokens[0]/1e12:.2f}T)")
            print(f"  Growth rate: {b:.2f}/year (averaged from other companies)")
        else:
            # Multiple data points: fit exponential
            a, b, r2 = fit_exponential(dates, tokens)
            
            doubling_time = np.log(2) / b if b > 0 else float('inf')
            monthly_growth = (np.exp(b / 12) - 1) * 100
            
            print(f"{name}:")
            print(f"  Data points: {len(df)}")
            print(f"  Latest estimate: {tokens[-1]:.2e} tokens/day ({tokens[-1]/1e12:.2f}T)")
            print(f"  Growth rate: {b:.2f}/year ({monthly_growth:.1f}%/month)")
            print(f"  Doubling time: {doubling_time:.2f} years ({doubling_time*12:.1f} months)")
            print(f"  R² fit: {r2:.3f}")
        
        models[name] = {'a': a, 'b': b, 'r2': r2}
        print()
    
    return companies, models


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
        
        # Calculate each series
        series_values = {}
        for name, params in models.items():
            tokens = exponential_model(decimal - 2024, params['a'], params['b'])
            projections[name].append(tokens)
            series_values[name] = tokens
        
        # Average company estimates before summing
        # OpenAI: average 3 methods: (ChatGPT + API), Inference, Revenue
        openai_chatgpt = series_values.get('OpenAI ChatGPT', 0)
        openai_api = series_values.get('OpenAI API', 0)
        openai_inference = series_values.get('OpenAI (Inference)', 0)
        openai_revenue = series_values.get('OpenAI (Revenue)', 0)
        # Average 3 methods: (ChatGPT + API), Inference, Revenue
        openai_avg = ((openai_chatgpt + openai_api) + openai_inference + openai_revenue) / 3
        
        # Google: sum products
        google_total = series_values.get('Gemini Assistant', 0) + series_values.get('Gemini API', 0)
        
        # Others: single series
        meta_total = series_values.get('Meta', 0)
        anthropic_total = series_values.get('Anthropic', 0)
        xai_total = series_values.get('xAI', 0)
        
        total = openai_avg + google_total + meta_total + anthropic_total + xai_total
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
    
    # Company colors (same color for same company)
    company_colors = {
        'OpenAI': '#10a37f',      # OpenAI green
        'Google': '#4285f4',      # Google blue
        'Meta': '#800080',        # Meta purple
        'Anthropic': '#d4a574',   # Anthropic tan/brown
        'xAI': '#1da1f2',         # xAI/Twitter blue
        'All': '#333333',         # Dark gray for total
    }
    
    # Map each category to its company color
    colors = {
        'OpenAI ChatGPT': company_colors['OpenAI'],
        'OpenAI API': company_colors['OpenAI'],
        'OpenAI (Inference)': company_colors['OpenAI'],
        'OpenAI (Revenue)': company_colors['OpenAI'],
        'Gemini Assistant': company_colors['Google'],
        'Gemini API': company_colors['Google'],
        'Meta': company_colors['Meta'],
        'Anthropic': company_colors['Anthropic'],
        'xAI': company_colors['xAI'],
    }
    
    # Markers: Chat (circle), API (square), Inference-scaled (plus), Revenue-scaled (star)
    markers = {
        'OpenAI ChatGPT': 'o',      # Chat - circle
        'OpenAI API': 's',          # API - square
        'OpenAI (Inference)': 'P',  # Inference-scaled - filled plus
        'OpenAI (Revenue)': '*',    # Revenue-scaled - star
        'Gemini Assistant': 'o',    # Chat - circle
        'Gemini API': 's',          # API - square
        'Meta': 'o',                # Chat - circle
        'Anthropic': '*',           # Revenue-based - star
        'xAI': '*',                 # Revenue-based - star
    }
    
    # Line styles: Chat (solid), API (dashed), Inference (dashdot), Revenue (dotted)
    linestyles = {
        'OpenAI ChatGPT': '-',      # Chat - solid
        'OpenAI API': '--',         # API - dashed
        'OpenAI (Inference)': '-.',  # Inference-scaled - dashdot
        'OpenAI (Revenue)': ':',    # Revenue-scaled - dotted
        'Gemini Assistant': '-',    # Chat - solid
        'Gemini API': '--',         # API - dashed
        'Meta': '-',                # Chat - solid
        'Anthropic': ':',           # Combined - dotted
        'xAI': ':',                 # Combined - dotted
    }
    
    # Time range for plotting
    t_plot = np.linspace(PLOT_BEGIN_DATE, PLOT_END_DATE, 300)
    
    # Plot each company
    for name, df in companies.items():
        # Plot data points
        ax.scatter(df['decimal_date'], df['daily_tokens'] / 1e12, 
                   s=80, color=colors[name], marker=markers[name], 
                   alpha=0.7, zorder=5, label=f'{name}')
        
        # Plot trendline
        params = models[name]
        y_fit = exponential_model(t_plot - 2024, params['a'], params['b']) / 1e12
        ax.plot(t_plot, y_fit, linestyle=linestyles[name], color=colors[name], 
                alpha=0.6, linewidth=2)
    
    # Plot Google "All AI Products" data points for reference (no trendline)
    # These include Search AI Overviews, Translate, etc. which don't scale the same way
    google_all_dates = [date_to_decimal(dp.date) for dp in google_data['monthly_tokens_all_products']]
    google_all_daily_tokens = [dp.value / 30 / 1e12 for dp in google_data['monthly_tokens_all_products']]  # monthly -> daily, in trillions
    ax.scatter(google_all_dates, google_all_daily_tokens, 
               s=100, color=company_colors['Google'], marker='X', 
               alpha=0.7, zorder=5, label='Google (All Products)')
    
    # Calculate and plot "All" (total) line
    # Group series by company and average before summing
    # OpenAI has 3 estimation methods: (ChatGPT + API), Inference-scaled, Revenue-scaled
    # Google has 2 products: Gemini Assistant + Gemini API (sum these)
    # Meta, Anthropic, xAI each have 1 series
    
    company_estimates = {
        'OpenAI': [],
        'Google': [],
        'Meta': [],
        'Anthropic': [],
        'xAI': [],
    }
    
    # Calculate each series
    for name in companies.keys():
        params = models[name]
        series = exponential_model(t_plot - 2024, params['a'], params['b']) / 1e12
        
        if name.startswith('OpenAI'):
            company_estimates['OpenAI'].append((name, series))
        elif name.startswith('Gemini'):
            company_estimates['Google'].append((name, series))
        elif name == 'Meta':
            company_estimates['Meta'].append((name, series))
        elif name == 'Anthropic':
            company_estimates['Anthropic'].append((name, series))
        elif name == 'xAI':
            company_estimates['xAI'].append((name, series))
    
    # For OpenAI: average 3 methods: (ChatGPT + API), Inference, Revenue
    openai_chatgpt = next((s for n, s in company_estimates['OpenAI'] if 'ChatGPT' in n), None)
    openai_api = next((s for n, s in company_estimates['OpenAI'] if n == 'OpenAI API'), None)
    openai_inference = next((s for n, s in company_estimates['OpenAI'] if 'Inference' in n), None)
    openai_revenue = next((s for n, s in company_estimates['OpenAI'] if 'Revenue' in n), None)
    
    openai_method1 = openai_chatgpt + openai_api if openai_chatgpt is not None and openai_api is not None else np.zeros_like(t_plot)
    openai_method2 = openai_inference if openai_inference is not None else np.zeros_like(t_plot)
    openai_method3 = openai_revenue if openai_revenue is not None else np.zeros_like(t_plot)
    openai_avg = (openai_method1 + openai_method2 + openai_method3) / 3
    
    # For Google: sum Gemini Assistant + Gemini API
    google_total = sum(s for _, s in company_estimates['Google'])
    
    # For others: just use single series
    meta_total = sum(s for _, s in company_estimates['Meta'])
    anthropic_total = sum(s for _, s in company_estimates['Anthropic'])
    xai_total = sum(s for _, s in company_estimates['xAI'])
    
    # Load population data and calculate human token-equivalents per day
    # Formula: population * 16 hours/day * 60 min/hr * 60 sec/min * HUMAN_TOKENS_PER_SECOND
    pop_df = pd.read_csv('population.csv')
    pop_years = pop_df['Year'].values
    pop_values = pop_df['all years'].values
    
    # Interpolate population at each t_plot point
    pop_interpolated = np.interp(t_plot, pop_years, pop_values)
    human_token_equivalents = pop_interpolated * 16 * 60 * 60 * HUMAN_TOKENS_PER_SECOND / 1e12  # in trillions
    ax.plot(t_plot, human_token_equivalents, color='red', linestyle='--', linewidth=2, label='Human Token-Equivalents')

    # Sum all companies
    total = openai_avg + google_total + meta_total + anthropic_total + xai_total
    
    ax.plot(t_plot, total, '-', color=company_colors['All'], linewidth=3, 
            label='All (avg per company)', zorder=4)
    
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Daily Tokens (Trillions)', fontsize=12)
    ax.set_title('Daily Token Production by Company/Product', fontsize=14, fontweight='bold')
    
    # Custom legend with marker explanations
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    # Add marker legend as text annotation
    ax.text(0.98, 0.02, '○ Chat  □ API  ✚ Inference  ★ Revenue', 
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(PLOT_BEGIN_DATE, PLOT_END_DATE)
    
    plt.tight_layout()
    plt.savefig('per_company_token_projection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Saved visualization to 'per_company_token_projection.png'")


def create_individual_company_plots(companies: dict, models: dict):
    """Create individual plots for each company/product."""
    
    # Create output folder if it doesn't exist
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    
    # Company colors
    company_colors = {
        'OpenAI': '#10a37f',
        'Google': '#4285f4',
        'Meta': '#800080',
        'Anthropic': '#d4a574',
        'xAI': '#1da1f2',
    }
    
    # Map each category to its company color
    colors = {
        'OpenAI ChatGPT': company_colors['OpenAI'],
        'OpenAI API': company_colors['OpenAI'],
        'OpenAI (Inference)': company_colors['OpenAI'],
        'OpenAI (Revenue)': company_colors['OpenAI'],
        'Gemini Assistant': company_colors['Google'],
        'Gemini API': company_colors['Google'],
        'Meta': company_colors['Meta'],
        'Anthropic': company_colors['Anthropic'],
        'xAI': company_colors['xAI'],
    }
    
    # Markers
    markers = {
        'OpenAI ChatGPT': 'o',
        'OpenAI API': 's',
        'OpenAI (Inference)': 'P',
        'OpenAI (Revenue)': '*',
        'Gemini Assistant': 'o',
        'Gemini API': 's',
        'Meta': 'o',
        'Anthropic': '*',
        'xAI': '*',
    }
    
    # Time range for plotting
    t_plot = np.linspace(PLOT_BEGIN_DATE, PLOT_END_DATE, 300)
    
    # ==========================================================================
    # Create combined OpenAI plot with all methods
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get OpenAI data and models
    openai_series = {name: (companies[name], models[name]) 
                     for name in companies.keys() if name.startswith('OpenAI')}
    
    # Define colors and styles for each OpenAI method
    openai_styles = {
        'OpenAI ChatGPT': {'color': '#2ecc71', 'marker': 'o', 'linestyle': '-', 'label': 'ChatGPT'},
        'OpenAI API': {'color': '#3498db', 'marker': 's', 'linestyle': '-', 'label': 'API'},
        'OpenAI (Inference)': {'color': '#e74c3c', 'marker': 'P', 'linestyle': '--', 'label': 'Inference-scaled'},
        'OpenAI (Revenue)': {'color': '#9b59b6', 'marker': '*', 'linestyle': '--', 'label': 'Revenue-scaled'},
    }
    
    # Plot each OpenAI method
    openai_projections = {}
    for name, (df, params) in openai_series.items():
        style = openai_styles[name]
        
        # Plot data points
        ax.scatter(df['decimal_date'], df['daily_tokens'] / 1e12, 
                   s=80, color=style['color'], marker=style['marker'], 
                   alpha=0.7, zorder=5)
        
        # Plot trendline
        y_fit = exponential_model(t_plot - 2024, params['a'], params['b']) / 1e12
        ax.plot(t_plot, y_fit, linestyle=style['linestyle'], color=style['color'], 
                alpha=0.7, linewidth=2, label=style['label'])
        
        openai_projections[name] = y_fit
    
    # Calculate and plot ChatGPT + API combined
    chatgpt_api_combined = openai_projections['OpenAI ChatGPT'] + openai_projections['OpenAI API']
    ax.plot(t_plot, chatgpt_api_combined, linestyle='-', color='#f39c12', 
            linewidth=2.5, label='ChatGPT + API', alpha=0.8)
    
    # Calculate and plot average of 3 methods: (ChatGPT + API), Inference, Revenue
    openai_avg = (chatgpt_api_combined + openai_projections['OpenAI (Inference)'] + 
                  openai_projections['OpenAI (Revenue)']) / 3
    ax.plot(t_plot, openai_avg, linestyle='-', color='#1a1a1a', 
            linewidth=3, label='Average (3 methods)', alpha=0.9)
    
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Daily Tokens (Trillions)', fontsize=12)
    ax.set_title('OpenAI - Daily Token Production (Multiple Estimation Methods)', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(PLOT_BEGIN_DATE, PLOT_END_DATE)
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_FOLDER, 'openai_combined_projection.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filepath}")
    
    # ==========================================================================
    # Create individual plots for non-OpenAI companies
    # ==========================================================================
    for name, df in companies.items():
        # Skip OpenAI - we already made a combined plot
        if name.startswith('OpenAI'):
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data points
        ax.scatter(df['decimal_date'], df['daily_tokens'] / 1e12, 
                   s=100, color=colors[name], marker=markers[name], 
                   alpha=0.8, zorder=5, label=f'{name} (data)')
        
        # Plot trendline
        params = models[name]
        y_fit = exponential_model(t_plot - 2024, params['a'], params['b']) / 1e12
        ax.plot(t_plot, y_fit, '-', color=colors[name], alpha=0.7, linewidth=2.5, 
                label=f'{name} (projection)')
        
        ax.set_yscale('log')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Daily Tokens (Trillions)', fontsize=12)
        ax.set_title(f'{name} - Daily Token Production Projection', fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(PLOT_BEGIN_DATE, PLOT_END_DATE)
        
        # Add growth rate annotation
        doubling_time = np.log(2) / params['b'] if params['b'] > 0 else float('inf')
        monthly_growth = (np.exp(params['b'] / 12) - 1) * 100
        ax.text(0.98, 0.02, f"Growth: {params['b']:.2f}/yr ({monthly_growth:.1f}%/mo)\nDoubling: {doubling_time*12:.1f} months", 
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Create safe filename
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        filepath = os.path.join(PLOTS_FOLDER, f'{safe_name}_projection.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved {filepath}")
    
    # Load population data for human token-equivalents plot
    pop_df = pd.read_csv('population.csv')
    pop_years = pop_df['Year'].values
    pop_values = pop_df['all years'].values
    pop_interpolated = np.interp(t_plot, pop_years, pop_values)
    human_token_equivalents = pop_interpolated * 16 * 60 * 60 * HUMAN_TOKENS_PER_SECOND / 1e12
    
    # Create a human token-equivalents only plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_plot, human_token_equivalents, color='red', linewidth=2.5, label='Human Token-Equivalents')
    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Daily Tokens (Trillions)', fontsize=12)
    ax.set_title('Human Token-Equivalents per Day (Population-Based)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(PLOT_BEGIN_DATE, PLOT_END_DATE)
    
    # Add methodology note
    ax.text(0.98, 0.02, f"Based on: population × 16 hrs/day × {HUMAN_TOKENS_PER_SECOND:.2f} tokens/sec\n(230 WPM × 4/3 tokens/word)", 
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filepath = os.path.join(PLOTS_FOLDER, 'human_token_equivalents.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filepath}")
    
    print(f"\nAll individual plots saved to '{PLOTS_FOLDER}/' folder")


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
    
    # Total row - average company estimates before summing
    dec_2025 = date_to_decimal(datetime(2025, 12, 1))
    dec_2026 = date_to_decimal(datetime(2026, 12, 1))
    
    def calc_total(decimal):
        series = {name: exponential_model(decimal - 2024, models[name]['a'], models[name]['b'])
                  for name in companies.keys()}
        # OpenAI: average 3 methods: (ChatGPT + API), Inference, Revenue
        openai_product = series.get('OpenAI ChatGPT', 0) + series.get('OpenAI API', 0)
        openai_avg = (openai_product + series.get('OpenAI (Inference)', 0) + series.get('OpenAI (Revenue)', 0)) / 3
        google_total = series.get('Gemini Assistant', 0) + series.get('Gemini API', 0)
        return openai_avg + google_total + series.get('Meta', 0) + series.get('Anthropic', 0) + series.get('xAI', 0)
    
    total_dec_2025 = calc_total(dec_2025)
    total_dec_2026 = calc_total(dec_2026)
    
    print(f"TOTAL (All Companies, averaged per company):")
    print(f"  Dec 2025: {total_dec_2025/1e12:.2f}T tokens/day")
    print(f"  Dec 2026: {total_dec_2026/1e12:.2f}T tokens/day")
    print(f"  YoY Growth: {(total_dec_2026/total_dec_2025 - 1)*100:.1f}%")
    print()


def export_data(companies: dict, models: dict):
    """Export projection data to CSV."""
    
    # Generate monthly projections using PLOT_BEGIN_DATE and PLOT_END_DATE
    begin_date = decimal_to_date(PLOT_BEGIN_DATE)
    end_date = decimal_to_date(PLOT_END_DATE)
    dates = pd.date_range(begin_date, end_date, freq='MS')
    
    data = {'date': dates}
    
    for name in companies.keys():
        params = models[name]
        decimals = [date_to_decimal(d.to_pydatetime()) for d in dates]
        tokens = [exponential_model(d - 2024, params['a'], params['b']) for d in decimals]
        data[f'{name}_daily_tokens'] = tokens
    
    # Add total - average company estimates before summing
    total_tokens = []
    for i in range(len(dates)):
        # OpenAI: average 3 methods
        openai_chatgpt = data.get('OpenAI ChatGPT_daily_tokens', [0]*len(dates))[i]
        openai_api = data.get('OpenAI API_daily_tokens', [0]*len(dates))[i]
        openai_inference = data.get('OpenAI (Inference)_daily_tokens', [0]*len(dates))[i]
        openai_revenue = data.get('OpenAI (Revenue)_daily_tokens', [0]*len(dates))[i]
        # Average 3 methods: (ChatGPT + API), Inference, Revenue
        openai_avg = ((openai_chatgpt + openai_api) + openai_inference + openai_revenue) / 3
        
        google_total = data.get('Gemini Assistant_daily_tokens', [0]*len(dates))[i] + data.get('Gemini API_daily_tokens', [0]*len(dates))[i]
        meta_total = data.get('Meta_daily_tokens', [0]*len(dates))[i]
        anthropic_total = data.get('Anthropic_daily_tokens', [0]*len(dates))[i]
        xai_total = data.get('xAI_daily_tokens', [0]*len(dates))[i]
        
        total_tokens.append(openai_avg + google_total + meta_total + anthropic_total + xai_total)
    
    data['total_daily_tokens'] = total_tokens
    
    df = pd.DataFrame(data)
    df.to_csv('token_projections.csv', index=False)
    print("Exported projections to 'token_projections.csv'")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Run analysis
    companies, models = analyze_and_project()
    
    # Generate projections
    projections, projection_dates = project_forward(companies, models)
    
    # Create summary
    create_summary_table(companies, models)
    
  
    print("=" * 70)
    print("METHODOLOGY NOTES")
    print("=" * 70)
    print(f"""
1. Consumer chat tokens (OpenAI ChatGPT, Gemini Assistant, Meta) are computed
   from message counts using CHAT_TOKENS_PER_MSG = {CHAT_TOKENS_PER_MSG} tokens/message.

2. Google tokens use GEMINI-ONLY data (API + chat), excluding "all AI products"
   figures which include Search AI Overviews, Translate, etc.

3. Meta tokens use a messages/MAU ratio (0.235) derived from Google trial data,
   applied linearly to Meta's reported MAU figures.

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

    # Create combined visualization
    create_visualization(companies, models, projections, projection_dates)
    
    # Create individual company plots
    print("\nGenerating individual company plots...")
    create_individual_company_plots(companies, models)
    
    # Export data
    export_data(companies, models)
  

# %%
