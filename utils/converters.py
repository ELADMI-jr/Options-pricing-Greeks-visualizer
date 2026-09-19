"""
FX Options Converters
Convert between different quoting conventions
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum
from scipy.optimize import newton
from scipy.stats import norm


class QuoteConvention(Enum):
    """FX options quoting conventions"""
    SPOT = "SPOT"
    FORWARD = "FORWARD"
    DELTA_SPOT = "DELTA_SPOT"
    DELTA_FORWARD = "DELTA_FORWARD"


class OptionStyle(Enum):
    """Option style"""
    CALL = "CALL"
    PUT = "PUT"


def strike_to_delta(
    spot: float,
    strike: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    option_style: OptionStyle = OptionStyle.CALL,
    convention: QuoteConvention = QuoteConvention.DELTA_FORWARD
) -> float:
    """
    Convert strike to delta
    
    Args:
        spot: Current spot rate
        strike: Option strike
        T: Time to maturity (years)
        r_d: Domestic risk-free rate
        r_f: Foreign risk-free rate
        vol: Implied volatility
        option_style: CALL or PUT
        convention: DELTA_SPOT or DELTA_FORWARD
    
    Returns:
        Delta value
    """
    forward = spot * np.exp((r_d - r_f) * T)
    
    if convention == QuoteConvention.DELTA_SPOT:
        # Spot delta
        d1 = (np.log(spot / strike) + (r_d - r_f + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        
        if option_style == OptionStyle.CALL:
            return np.exp(-r_f * T) * norm.cdf(d1)
        else:
            return -np.exp(-r_f * T) * norm.cdf(-d1)
    
    else:  # DELTA_FORWARD
        # Forward delta
        d1_forward = (np.log(forward / strike) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        
        if option_style == OptionStyle.CALL:
            return np.exp(-r_f * T) * norm.cdf(d1_forward)
        else:
            return -np.exp(-r_f * T) * norm.cdf(-d1_forward)


def delta_to_strike(
    spot: float,
    delta: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    option_style: OptionStyle = OptionStyle.CALL,
    convention: QuoteConvention = QuoteConvention.DELTA_FORWARD,
    max_iter: int = 100,
    tol: float = 1e-8
) -> float:
    """
    Convert delta to strike using Newton-Raphson
    
    Args:
        spot: Current spot rate
        delta: Target delta
        T: Time to maturity (years)
        r_d: Domestic risk-free rate
        r_f: Foreign risk-free rate
        vol: Implied volatility
        option_style: CALL or PUT
        convention: DELTA_SPOT or DELTA_FORWARD
        max_iter: Maximum iterations
        tol: Tolerance
    
    Returns:
        Strike value
    """
    forward = spot * np.exp((r_d - r_f) * T)
    
    # Initial guess
    if abs(delta - 0.5) < 0.1:
        strike_guess = forward  # ATM
    elif delta < 0.5:
        # OTM put / ITM call
        if option_style == OptionStyle.CALL:
            strike_guess = forward * 1.2  # OTM call
        else:
            strike_guess = forward * 0.8  # ITM put
    else:
        # ITM call / OTM put
        if option_style == OptionStyle.CALL:
            strike_guess = forward * 0.8  # ITM call
        else:
            strike_guess = forward * 1.2  # OTM put
    
    def objective(K):
        return strike_to_delta(spot, K, T, r_d, r_f, vol, option_style, convention) - delta
    
    try:
        strike = newton(objective, strike_guess, maxiter=max_iter, tol=tol)
        return float(strike)
    except RuntimeError:
        # Fallback to bisection
        low = forward * 0.1
        high = forward * 10
        
        for _ in range(max_iter):
            mid = (low + high) / 2
            delta_mid = strike_to_delta(spot, mid, T, r_d, r_f, vol, option_style, convention)
            
            if abs(delta_mid - delta) < tol:
                return mid
            
            if (delta_mid - delta) * (strike_to_delta(spot, low, T, r_d, r_f, vol, option_style, convention) - delta) < 0:
                high = mid
            else:
                low = mid
        
        return (low + high) / 2


def forward_to_atm_strike(
    forward: float,
    T: float,
    r_d: float,
    r_f: float
) -> float:
    """
    Convert forward to ATM strike (in FX, ATM is typically at forward)
    
    Args:
        forward: Forward rate
        T: Time to maturity
        r_d: Domestic rate
        r_f: Foreign rate
    
    Returns:
        ATM strike
    """
    # In FX markets, ATM strike is typically the forward rate
    return forward


def convert_vol_quote_convention(
    vol: float,
    from_convention: QuoteConvention,
    to_convention: QuoteConvention,
    spot: float,
    strike: float,
    T: float,
    r_d: float,
    r_f: float
) -> float:
    """
    Convert volatility between different quoting conventions
    
    Note: This is approximate as it depends on the relationship between
    spot and forward deltas
    """
    if from_convention == to_convention:
        return vol
    
    forward = spot * np.exp((r_d - r_f) * T)
    
    # For simplicity, we use the relationship:
    # forward_delta ≈ spot_delta * exp(r_f * T)
    # This is approximate
    
    if from_convention == QuoteConvention.DELTA_SPOT and to_convention == QuoteConvention.DELTA_FORWARD:
        # Convert spot delta vol to forward delta vol
        # This requires knowing the strike, which we don't have
        # Return unchanged as approximation
        return vol
    
    elif from_convention == QuoteConvention.DELTA_FORWARD and to_convention == QuoteConvention.DELTA_SPOT:
        # Convert forward delta vol to spot delta vol
        return vol
    
    return vol


def strike_to_moneyness(
    strike: float,
    forward: float
) -> float:
    """
    Convert strike to moneyness (K/F)
    
    Args:
        strike: Option strike
        forward: Forward rate
    
    Returns:
        Moneyness (K/F)
    """
    return strike / forward


def moneyness_to_strike(
    moneyness: float,
    forward: float
) -> float:
    """
    Convert moneyness to strike
    
    Args:
        moneyness: K/F ratio
        forward: Forward rate
    
    Returns:
        Strike
    """
    return moneyness * forward


def log_moneyness(
    strike: float,
    forward: float
) -> float:
    """
    Calculate log moneyness: ln(K/F)
    
    Args:
        strike: Option strike
        forward: Forward rate
    
    Returns:
        Log moneyness
    """
    return np.log(strike / forward)
