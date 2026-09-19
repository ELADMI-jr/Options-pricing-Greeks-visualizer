"""
FX Options Validators
Validate inputs and check for arbitrage conditions
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """Validation error"""
    pass


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


def validate_fx_params(
    spot: float,
    strike: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float
) -> ValidationResult:
    """
    Validate FX option parameters
    
    Args:
        spot: Spot rate
        strike: Strike
        T: Time to maturity (years)
        r_d: Domestic rate
        r_f: Foreign rate
        vol: Volatility
    
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    # Check positive values
    if spot <= 0:
        errors.append("Spot must be positive")
    
    if strike <= 0:
        errors.append("Strike must be positive")
    
    if T <= 0:
        errors.append("Time to maturity must be positive")
    elif T < 1/365:  # Less than 1 day
        warnings.append("Very short time to maturity")
    
    if vol < 0:
        errors.append("Volatility cannot be negative")
    elif vol > 5.0:  # 500%
        warnings.append("Extremely high volatility")
    
    if vol == 0:
        warnings.append("Zero volatility - option has no time value")
    
    # Check interest rates
    if abs(r_d) > 1.0:  # 100%
        warnings.append("Extremely high domestic interest rate")
    
    if abs(r_f) > 1.0:
        warnings.append("Extremely high foreign interest rate")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def check_arbitrage_conditions(
    strikes: List[float],
    call_vols: List[float],
    put_vols: List[float],
    forward: float
) -> Dict[str, bool]:
    """
    Check for arbitrage violations in volatility surface
    
    Args:
        strikes: List of strikes
        call_vols: List of call volatilities
        put_vols: List of put volatilities
        forward: Forward rate
    
    Returns:
        Dictionary of arbitrage check results
    """
    results = {}
    
    # Check 1: Call-put parity for ATM
    atm_idx = np.argmin(np.abs(np.array(strikes) - forward))
    call_atm = call_vols[atm_idx]
    put_atm = put_vols[atm_idx]
    
    # ATM call and put should have similar volatility
    results["atm_call_put_parity"] = abs(call_atm - put_atm) < 0.01
    
    # Check 2: No butterfly arbitrage
    # For any three strikes K1 < K2 < K3:
    # sigma(K2) <= 0.5 * (sigma(K1) + sigma(K3)) if K2 = (K1 + K3)/2
    butterfly_ok = True
    sorted_indices = np.argsort(strikes)
    sorted_strikes = np.array(strikes)[sorted_indices]
    sorted_call_vols = np.array(call_vols)[sorted_indices]
    
    for i in range(1, len(sorted_strikes) - 1):
        K1, K2, K3 = sorted_strikes[i-1], sorted_strikes[i], sorted_strikes[i+1]
        v1, v2, v3 = sorted_call_vols[i-1], sorted_call_vols[i], sorted_call_vols[i+1]
        
        # Check if K2 is approximately the midpoint
        if abs(K2 - (K1 + K3)/2) < 0.01 * (K3 - K1):
            if v2 > 0.5 * (v1 + v3) + 1e-6:
                butterfly_ok = False
                break
    
    results["no_butterfly_arbitrage"] = butterfly_ok
    
    # Check 3: Monotonicity (no calendar spread arbitrage)
    # Volatility should not decrease too sharply with strike
    # This is a soft check
    max_vol_change = np.max(np.abs(np.diff(sorted_call_vols)))
    results["monotonicity_ok"] = max_vol_change < 0.5  # Less than 50% change between adjacent strikes
    
    # Check 4: Positive volatility
    results["positive_volatility"] = all(v > 0 for v in call_vols + put_vols)
    
    return results


def validate_volatility_surface(
    vol_surface: Dict[str, Dict[float, float]],
    min_vol: float = 0.01,
    max_vol: float = 2.0
) -> ValidationResult:
    """
    Validate a volatility surface
    
    Args:
        vol_surface: {tenor: {delta: vol}}
        min_vol: Minimum acceptable volatility
        max_vol: Maximum acceptable volatility
    
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    for tenor, smile in vol_surface.items():
        for delta, vol in smile.items():
            if vol < 0:
                errors.append(f"Negative volatility for {tenor} at {delta}Δ")
            elif vol < min_vol:
                warnings.append(f"Very low volatility ({vol:.4f}) for {tenor} at {delta}Δ")
            elif vol > max_vol:
                warnings.append(f"Very high volatility ({vol:.4f}) for {tenor} at {delta}Δ")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def check_svi_arbitrage(
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float
) -> Tuple[bool, List[str]]:
    """
    Check SVI parameters for arbitrage violations
    
    Args:
        a: Level parameter
        b: Slope parameter
        rho: Correlation parameter
        m: Skew parameter
        sigma: Convexity parameter
    
    Returns:
        (is_valid, list of violations)
    """
    violations = []
    
    # Check 1: a >= 0
    if a < 0:
        violations.append("a must be non-negative")
    
    # Check 2: b >= 0
    if b < 0:
        violations.append("b must be non-negative")
    
    # Check 3: |rho| <= 1
    if abs(rho) > 1:
        violations.append("|rho| must be <= 1")
    
    # Check 4: sigma > 0
    if sigma <= 0:
        violations.append("sigma must be positive")
    
    # Check 5: Andersen-Brotherton-Ratcliffe condition
    # For SVI-J: a >= sigma^2 / 4
    if a < sigma**2 / 4:
        violations.append(f"a < sigma^2/4: {a:.6f} < {sigma**2/4:.6f}")
    
    # Check 6: b * (1 - |rho|) >= 0
    if b * (1 - abs(rho)) < 0:
        violations.append("b * (1 - |rho|) must be non-negative")
    
    return len(violations) == 0, violations


def validate_market_data(
    spot: float,
    forward: float,
    r_d: float,
    r_f: float,
    T: float
) -> ValidationResult:
    """
    Validate market data inputs
    
    Args:
        spot: Spot rate
        forward: Forward rate
        r_d: Domestic rate
        r_f: Foreign rate
        T: Time to maturity
    
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    if spot <= 0:
        errors.append("Spot must be positive")
    
    if forward <= 0:
        errors.append("Forward must be positive")
    
    # Check forward consistency
    calculated_forward = spot * np.exp((r_d - r_f) * T)
    if abs(forward - calculated_forward) / calculated_forward > 0.01:
        warnings.append(f"Forward ({forward:.6f}) differs from calculated ({calculated_forward:.6f}) by >1%")
    
    if T <= 0:
        errors.append("Time to maturity must be positive")
    
    if abs(r_d) > 1.0:
        warnings.append("Very high domestic interest rate")
    
    if abs(r_f) > 1.0:
        warnings.append("Very high foreign interest rate")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


class FXOptionValidator:
    """
    Comprehensive FX option validator
    """
    
    def __init__(self):
        self.epsilon = 1e-10
    
    def validate_all(
        self,
        spot: float,
        strike: float,
        T: float,
        r_d: float,
        r_f: float,
        vol: float
    ) -> ValidationResult:
        """
        Validate all FX option parameters
        """
        return validate_fx_params(spot, strike, T, r_d, r_f, vol)
    
    def validate_volatility_surface(
        self,
        vol_surface: Dict[str, Dict[float, float]]
    ) -> ValidationResult:
        """
        Validate volatility surface
        """
        return validate_volatility_surface(vol_surface)
    
    def check_arbitrage(
        self,
        strikes: List[float],
        call_vols: List[float],
        put_vols: List[float],
        forward: float
    ) -> Dict[str, bool]:
        """
        Check for arbitrage violations
        """
        return check_arbitrage_conditions(strikes, call_vols, put_vols, forward)
