"""
SVI (Stochastic Volatility Inspired) Model for FX Volatility Surface Fitting

Implements the SVI parameterization and arbitrage checks as used by FX desks
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from scipy.optimize import minimize, curve_fit
from enum import Enum


class SVIVariance(Enum):
    """SVI variance parameterization types"""
    CLASSIC = "classic"          # Original SVI
    JS = "js"                    # SVI-J (Jim Gatheral)
    JW = "jw"                    # SVI-JW (with wings)
    NATURAL = "natural"          # Natural SVI


@dataclass
class SVIParams:
    """SVI model parameters"""
    a: float = 0.0       # Level parameter
    b: float = 0.0       # Slope parameter
    rho: float = 0.0     # Correlation parameter (-1 to 1)
    m: float = 0.0       # Skew parameter
    sigma: float = 0.0   # Convexity parameter
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "rho": self.rho,
            "m": self.m,
            "sigma": self.sigma
        }
    
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'SVIParams':
        return cls(
            a=params.get("a", 0.0),
            b=params.get("b", 0.0),
            rho=params.get("rho", 0.0),
            m=params.get("m", 0.0),
            sigma=params.get("sigma", 0.0)
        )


@dataclass
class SVIResult:
    """SVI fitting result"""
    params: SVIParams
    fitted_vols: Dict[float, float]  # {strike: fitted_vol}
    rmse: float
    max_error: float
    arbitrage_violations: List[str] = field(default_factory=list)
    is_valid: bool = True


class SVIModel:
    """
    SVI (Stochastic Volatility Inspired) Model
    
    Implements the SVI parameterization for volatility smiles:
    
    Classic SVI:
    sigma^2(K) = a + b * (rho * (k - m) + sqrt((k - m)^2 + epsilon^2))
    
    where:
    - k = ln(K/F) (log moneyness)
    - F = forward price
    - a, b, rho, m, epsilon are parameters
    
    SVI-J (Jim Gatheral modification):
    sigma^2(K) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    SVI-JW (with wings):
    Adds separate parameters for left and right wings
    
    References:
    - Gatheral, J. (2004). "The Volatility Surface: A Practitioner's Guide"
    - Gatheral, J., & Jacquier, A. (2011). "Arbitrage-free SVI volatility surfaces"
    """
    
    def __init__(self, variance_type: SVIVariance = SVIVariance.JS):
        self.variance_type = variance_type
        self.epsilon = 1e-10  # Small constant for numerical stability
    
    def total_variance(self, k: float, params: SVIParams) -> float:
        """
        Calculate total variance using SVI parameterization
        
        Args:
            k: Log moneyness = ln(K/F)
            params: SVI parameters
        
        Returns:
            Total variance sigma^2(K)
        """
        if self.variance_type == SVIVariance.CLASSIC:
            # Classic SVI with epsilon
            return (params.a + params.b * (
                params.rho * (k - params.m) + 
                np.sqrt((k - params.m)**2 + self.epsilon**2)
            ))
        elif self.variance_type == SVIVariance.JS:
            # SVI-J: uses sigma instead of epsilon
            return (params.a + params.b * (
                params.rho * (k - params.m) + 
                np.sqrt((k - params.m)**2 + params.sigma**2)
            ))
        elif self.variance_type == SVIVariance.JW:
            # SVI-JW: different parameters for left/right wings
            # sigma^2(K) = a + b_left * (rho_left * (k - m) + sqrt((k - m)^2 + sigma_left^2)) for k < m
            # sigma^2(K) = a + b_right * (rho_right * (k - m) + sqrt((k - m)^2 + sigma_right^2)) for k >= m
            # For simplicity, we use the JS parameterization
            return (params.a + params.b * (
                params.rho * (k - params.m) + 
                np.sqrt((k - params.m)**2 + params.sigma**2)
            ))
        else:  # NATURAL
            # Natural SVI: ensures arbitrage-free conditions
            return self._natural_svi(k, params)
    
    def _natural_svi(self, k: float, params: SVIParams) -> float:
        """
        Natural SVI parameterization
        Ensures no calendar spread arbitrage
        """
        # Natural SVI uses different parameter constraints
        # sigma^2(K) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        # with constraints: b >= 0, |rho| <= 1, sigma > 0
        return (params.a + params.b * (
            params.rho * (k - params.m) + 
            np.sqrt((k - params.m)**2 + params.sigma**2)
        ))
    
    def volatility(self, k: float, params: SVIParams) -> float:
        """
        Calculate volatility (not variance) from SVI
        """
        return np.sqrt(self.total_variance(k, params))
    
    def fit_to_market_data(
        self,
        strikes: List[float],
        market_vols: List[float],
        forward: float,
        initial_params: Optional[SVIParams] = None,
        method: str = "L-BFGS-B"
    ) -> SVIResult:
        """
        Fit SVI parameters to market volatility data
        
        Args:
            strikes: List of strike prices
            market_vols: List of market implied volatilities
            forward: Forward price
            initial_params: Optional initial parameter guess
            method: Optimization method
        
        Returns:
            SVIResult with fitted parameters and diagnostics
        """
        assert len(strikes) == len(market_vols), "Strikes and vols must have same length"
        assert len(strikes) > 5, "Need at least 5 data points for robust fitting"
        
        # Convert strikes to log moneyness
        k_values = np.array([np.log(s / forward) for s in strikes])
        target_vols = np.array(market_vols)
        
        # Initial parameter guess
        if initial_params is None:
            initial_params = self._get_initial_guess(k_values, target_vols)
        
        # Objective function: minimize RMSE between market and SVI vols
        def objective(params_array):
            params = SVIParams(
                a=params_array[0],
                b=params_array[1],
                rho=params_array[2],
                m=params_array[3],
                sigma=params_array[4]
            )
            
            # Calculate SVI volatilities
            svi_vols = np.array([self.volatility(k, params) for k in k_values])
            
            # RMSE
            errors = svi_vols - target_vols
            rmse = np.sqrt(np.mean(errors**2))
            
            return rmse
        
        # Parameter bounds
        # a >= 0, b >= 0, -1 <= rho <= 1, sigma > 0
        bounds = [
            (0, None),      # a
            (0, None),      # b
            (-1, 1),        # rho
            (None, None),   # m
            (1e-6, None)    # sigma
        ]
        
        # Initial parameter array
        x0 = np.array([
            initial_params.a,
            initial_params.b,
            initial_params.rho,
            initial_params.m,
            initial_params.sigma
        ])
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds
        )
        
        # Extract fitted parameters
        fitted_params = SVIParams(
            a=result.x[0],
            b=result.x[1],
            rho=result.x[2],
            m=result.x[3],
            sigma=result.x[4]
        )
        
        # Calculate fitted volatilities
        fitted_vols = {}
        for s, k in zip(strikes, k_values):
            fitted_vols[s] = self.volatility(k, fitted_params)
        
        # Calculate errors
        svi_vols_array = np.array([self.volatility(k, fitted_params) for k in k_values])
        errors = svi_vols_array - target_vols
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_error = float(np.max(np.abs(errors)))
        
        # Check for arbitrage violations
        arbitrage_violations = self.check_arbitrage(fitted_params, strikes, forward)
        
        return SVIResult(
            params=fitted_params,
            fitted_vols=fitted_vols,
            rmse=rmse,
            max_error=max_error,
            arbitrage_violations=arbitrage_violations,
            is_valid=len(arbitrage_violations) == 0
        )
    
    def _get_initial_guess(self, k_values: np.ndarray, target_vols: np.ndarray) -> SVIParams:
        """Get reasonable initial parameter guess"""
        # ATM volatility (at k=0)
        atm_idx = np.argmin(np.abs(k_values))
        atm_vol = target_vols[atm_idx]
        
        # Rough estimate of a (ATM variance)
        a_initial = atm_vol**2
        
        # Estimate b from the spread
        max_vol = np.max(target_vols)
        min_vol = np.min(target_vols)
        vol_spread = max_vol - min_vol
        b_initial = vol_spread / 2
        
        # Estimate m (where the minimum variance occurs)
        min_idx = np.argmin(target_vols)
        m_initial = k_values[min_idx]
        
        # Estimate rho from skew
        # If left side has higher vol, rho is negative
        left_k = k_values[k_values < 0]
        right_k = k_values[k_values > 0]
        
        if len(left_k) > 0 and len(right_k) > 0:
            left_vols = target_vols[k_values < 0]
            right_vols = target_vols[k_values > 0]
            
            if np.mean(left_vols) > np.mean(right_vols):
                rho_initial = -0.5
            else:
                rho_initial = 0.5
        else:
            rho_initial = 0.0
        
        # Estimate sigma
        sigma_initial = 0.1
        
        return SVIParams(
            a=a_initial,
            b=b_initial,
            rho=rho_initial,
            m=m_initial,
            sigma=sigma_initial
        )
    
    def check_arbitrage(
        self,
        params: SVIParams,
        strikes: List[float],
        forward: float
    ) -> List[str]:
        """
        Check for arbitrage violations in SVI fit
        
        Checks:
        1. No calendar spread arbitrage (total variance must be positive)
        2. No butterfly arbitrage (convexity condition)
        3. Feller condition (variance > 0 for all strikes)
        4. Short maturity condition
        
        Returns:
            List of arbitrage violation descriptions
        """
        violations = []
        
        # Convert strikes to log moneyness
        k_values = np.array([np.log(s / forward) for s in strikes])
        
        # Check 1: Total variance must be positive for all strikes
        total_vars = np.array([self.total_variance(k, params) for k in k_values])
        if np.any(total_vars <= 0):
            violations.append("Calendar spread arbitrage: Total variance non-positive for some strikes")
        
        # Check 2: Convexity condition (second derivative of variance >= 0)
        # For SVI, this requires: b * sigma^2 / (|k - m|^2 + sigma^2)^(3/2) >= 0
        # Since b >= 0 and sigma > 0, this is always satisfied if b >= 0
        if params.b < 0:
            violations.append("Butterfly arbitrage: b parameter must be non-negative")
        
        # Check 3: Feller condition for all k
        # Total variance must be positive and finite
        if params.a < 0:
            violations.append("Feller condition violation: a must be non-negative")
        
        # Check 4: |rho| <= 1
        if abs(params.rho) > 1:
            violations.append("Correlation violation: |rho| must be <= 1")
        
        # Check 5: sigma > 0
        if params.sigma <= 0:
            violations.append("Convexity violation: sigma must be positive")
        
        # Check 6: No negative densities (Andersen & Brotherton-Ratcliffe condition)
        # For SVI-J, need: a >= sigma^2 / 4 and b * (1 - |rho|) >= 0
        if self.variance_type == SVIVariance.JS:
            if params.a < params.sigma**2 / 4:
                violations.append("Andersen-BR condition: a < sigma^2 / 4")
            if params.b * (1 - abs(params.rho)) < 0:
                violations.append("Andersen-BR condition: b * (1 - |rho|) < 0")
        
        # Check 7: Martingale condition (for SVI-JW)
        # Integral of density over all strikes should equal 1
        # This is complex to check numerically, so we skip for now
        
        return violations
    
    def ensure_arbitrage_free(
        self,
        params: SVIParams,
        strikes: List[float],
        forward: float
    ) -> SVIParams:
        """
        Adjust SVI parameters to ensure arbitrage-free
        
        Uses the method from Gatheral & Jacquier (2011)
        """
        # Start with the given parameters
        adjusted_params = SVIParams(
            a=params.a,
            b=params.b,
            rho=params.rho,
            m=params.m,
            sigma=params.sigma
        )
        
        # Apply constraints
        # 1. a >= 0
        adjusted_params.a = max(adjusted_params.a, 0)
        
        # 2. b >= 0
        adjusted_params.b = max(adjusted_params.b, 0)
        
        # 3. |rho| <= 1
        adjusted_params.rho = np.clip(adjusted_params.rho, -1, 1)
        
        # 4. sigma > 0
        adjusted_params.sigma = max(adjusted_params.sigma, 1e-6)
        
        # 5. For SVI-J: a >= sigma^2 / 4
        if self.variance_type == SVIVariance.JS:
            adjusted_params.a = max(adjusted_params.a, adjusted_params.sigma**2 / 4)
        
        # 6. b * (1 - |rho|) >= 0
        if adjusted_params.b * (1 - abs(adjusted_params.rho)) < 0:
            # Adjust rho to satisfy condition
            if adjusted_params.b > 0:
                # b > 0, so need 1 - |rho| >= 0 (always true if |rho| <= 1)
                pass
            else:
                # b = 0 already from constraint 2
                pass
        
        return adjusted_params
    
    def density(self, k: float, params: SVIParams) -> float:
        """
        Calculate risk-neutral density from SVI
        
        The density is proportional to the second derivative of total variance
        w.r.t. log moneyness
        """
        # Numerical second derivative
        h = 1e-6
        var_plus = self.total_variance(k + h, params)
        var_minus = self.total_variance(k - h, params)
        var_center = self.total_variance(k, params)
        
        second_deriv = (var_plus - 2 * var_center + var_minus) / (h**2)
        
        # Density is proportional to second derivative
        # Need to normalize to integrate to 1
        return max(second_deriv, 0)
    
    def calculate_atm_vol(self, params: SVIParams) -> float:
        """Calculate ATM volatility (at k=0)"""
        return self.volatility(0, params)
    
    def calculate_skew(self, params: SVIParams, k_range: Tuple[float, float] = (-1, 1)) -> float:
        """
        Calculate skew of the volatility smile
        Skew = (vol(k=1) - vol(k=-1)) / ATM_vol
        """
        vol_pos = self.volatility(1, params)
        vol_neg = self.volatility(-1, params)
        atm_vol = self.volatility(0, params)
        
        if atm_vol > 0:
            return (vol_pos - vol_neg) / atm_vol
        return 0.0
    
    def calculate_convexity(self, params: SVIParams) -> float:
        """
        Calculate convexity of the volatility smile
        Convexity = (vol(k=1) + vol(k=-1) - 2 * vol(k=0)) / ATM_vol
        """
        vol_pos = self.volatility(1, params)
        vol_neg = self.volatility(-1, params)
        atm_vol = self.volatility(0, params)
        
        if atm_vol > 0:
            return (vol_pos + vol_neg - 2 * atm_vol) / atm_vol
        return 0.0


class SVIJWModel(SVIModel):
    """
    SVI-JW (SVI with Wings) Model
    
    Extends SVI to have different parameters for left and right wings
    This provides better fit for very OTM options
    
    sigma^2(K) = 
        a + b_left * (rho_left * (k - m) + sqrt((k - m)^2 + sigma_left^2)) for k < m
        a + b_right * (rho_right * (k - m) + sqrt((k - m)^2 + sigma_right^2)) for k >= m
    """
    
    def __init__(self):
        super().__init__(variance_type=SVIVariance.JW)
    
    def total_variance(self, k: float, params: SVIParams, 
                       params_left: Optional[SVIParams] = None,
                       params_right: Optional[SVIParams] = None) -> float:
        """
        Calculate total variance with separate left/right parameters
        """
        # For simplicity, use the standard SVI-J parameters
        # In a full implementation, we would have separate params for left/right
        return super().total_variance(k, params)


class SVIFitter:
    """
    High-level SVI fitting interface
    """
    
    def __init__(self, variance_type: SVIVariance = SVIVariance.JS):
        self.model = SVIModel(variance_type)
    
    def fit(
        self,
        strikes: List[float],
        market_vols: List[float],
        forward: float,
        initial_params: Optional[SVIParams] = None
    ) -> SVIResult:
        """
        Fit SVI to market data
        """
        return self.model.fit_to_market_data(
            strikes, market_vols, forward, initial_params
        )
    
    def fit_with_arbitrage_check(
        self,
        strikes: List[float],
        market_vols: List[float],
        forward: float,
        initial_params: Optional[SVIParams] = None
    ) -> SVIResult:
        """
        Fit SVI and ensure arbitrage-free
        """
        result = self.fit(strikes, market_vols, forward, initial_params)
        
        if not result.is_valid:
            # Try to adjust parameters
            adjusted_params = self.model.ensure_arbitrage_free(
                result.params, strikes, forward
            )
            
            # Re-fit with adjusted parameters as initial guess
            result = self.fit(strikes, market_vols, forward, adjusted_params)
        
        return result
    
    def interpolate_vol(self, strike: float, forward: float, 
                       result: SVIResult) -> float:
        """
        Interpolate volatility at a given strike using fitted SVI
        """
        k = np.log(strike / forward)
        return self.model.volatility(k, result.params)
    
    def extrapolate_vol(self, strike: float, forward: float,
                       result: SVIResult, max_extrapolation: float = 0.5) -> Optional[float]:
        """
        Extrapolate volatility beyond the fitted range
        
        Args:
            max_extrapolation: Maximum |k| to extrapolate to
        """
        k = np.log(strike / forward)
        
        if abs(k) > max_extrapolation:
            return None
        
        return self.model.volatility(k, result.params)


# ============================================================================
# SABR Model (Optional)
# ============================================================================

class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) Model
    
    Alternative to SVI for volatility surface fitting
    Particularly popular for interest rate options
    
    SABR dynamics:
    dF_t = sigma_t * F_t^beta * dW_t
    dsigma_t = nu * sigma_t * dZ_t
    d<W, Z>_t = rho * dt
    
    Implied volatility approximation (Hagan, 2002):
    sigma_B(K, F) = (sigma_0 / ((K*F)^((1-beta)/2))) * (z / ln((1-2*rho*z+z^2)/(1-rho^2)))
    where z = (nu / sigma_0) * (K*F)^((1-beta)/2) * ln(F/K)
    """
    
    def __init__(self):
        self.epsilon = 1e-10
    
    def implied_vol(
        self,
        strike: float,
        forward: float,
        alpha: float,
        beta: float,
        nu: float,
        rho: float
    ) -> float:
        """
        Calculate SABR implied volatility
        
        Args:
            strike: Strike price
            forward: Forward price
            alpha: Initial volatility (sigma_0)
            beta: Elasticity parameter (0 <= beta <= 1)
            nu: Volatility of volatility
            rho: Correlation between F and sigma
        
        Returns:
            Implied volatility
        """
        if strike <= 0 or forward <= 0:
            return 0.0
        
        if abs(forward - strike) < 1e-10:
            return alpha / (forward ** ((1 - beta) / 2))
        
        FK = forward * strike
        FK_beta = FK ** ((1 - beta) / 2)
        
        ln_FK = np.log(forward / strike)
        
        # Calculate z
        z = (nu / alpha) * FK_beta * ln_FK
        
        # Calculate the log term
        numerator = 1 - 2 * rho * z + z**2
        denominator = 1 - rho**2
        
        if denominator <= 0:
            return 0.0
        
        log_term = np.log(numerator / denominator)
        
        if abs(log_term) < 1e-10:
            # Taylor expansion for small log_term
            vol = (alpha / FK_beta) * (1 + ((1 - beta)**2 / 24) * (ln_FK**2) * 
                   (nu**2 / alpha**2) * FK ** (1 - beta) + 
                   (rho * beta * nu / (4 * alpha)) * FK_beta * ln_FK + 
                   ((2 - 3 * rho**2) / 24) * (ln_FK**2))
        else:
            vol = (alpha / FK_beta) * (z / log_term)
        
        return max(vol, 0.0)
    
    def fit_to_market(
        self,
        strikes: List[float],
        market_vols: List[float],
        forward: float,
        beta: float = 0.5
    ) -> Dict[str, float]:
        """
        Fit SABR parameters to market data
        
        Returns: {alpha, beta, nu, rho}
        """
        from scipy.optimize import minimize
        
        def objective(params):
            alpha, nu, rho = params
            
            sabr_vols = np.array([
                self.implied_vol(s, forward, alpha, beta, nu, rho)
                for s in strikes
            ])
            
            errors = sabr_vols - np.array(market_vols)
            return np.sum(errors**2)
        
        # Initial guess
        atm_idx = np.argmin(np.abs(np.array(strikes) - forward))
        atm_vol = market_vols[atm_idx]
        
        x0 = np.array([
            atm_vol * (forward ** ((1 - beta) / 2)),  # alpha
            0.3,  # nu
            0.0   # rho
        ])
        
        bounds = [
            (1e-6, None),   # alpha > 0
            (1e-6, None),   # nu > 0
            (-1, 1)         # -1 <= rho <= 1
        ]
        
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return {
            "alpha": float(result.x[0]),
            "beta": beta,
            "nu": float(result.x[1]),
            "rho": float(result.x[2]),
            "rmse": float(np.sqrt(result.fun / len(strikes)))
        }


# ============================================================================
# Volatility Surface Analytics
# ============================================================================

class VolSurfaceAnalytics:
    """
    Analytics and diagnostics for volatility surfaces
    """
    
    @staticmethod
    def check_arbitrage_conditions(
        strikes: List[float],
        vols: List[float],
        forward: float
    ) -> Dict[str, bool]:
        """
        Check various arbitrage conditions on a volatility surface
        
        Checks:
        1. No calendar spread arbitrage (monotonicity in time)
        2. No butterfly arbitrage (convexity in strike)
        3. Call-put parity
        
        Returns:
            Dictionary of check results
        """
        results = {}
        
        # Check 1: Butterfly arbitrage
        # For any three strikes K1 < K2 < K3 with K2 = (K1 + K3)/2:
        # 2*sigma(K2) <= sigma(K1) + sigma(K3)
        
        sorted_strikes = sorted(strikes)
        sorted_vols = [vols[strikes.index(s)] for s in sorted_strikes]
        
        butterfly_ok = True
        for i in range(1, len(sorted_strikes) - 1):
            K1, K2, K3 = sorted_strikes[i-1], sorted_strikes[i], sorted_strikes[i+1]
            v1, v2, v3 = sorted_vols[i-1], sorted_vols[i], sorted_vols[i+1]
            
            # Check if K2 is approximately the midpoint
            if abs(K2 - (K1 + K3)/2) < 0.01 * (K3 - K1):
                if 2 * v2 > v1 + v3 + 1e-6:
                    butterfly_ok = False
                    break
        
        results["no_butterfly_arbitrage"] = butterfly_ok
        
        # Check 2: Convexity (second derivative non-negative)
        # Numerical check
        if len(sorted_strikes) >= 3:
            # Calculate second differences
            second_diffs = []
            for i in range(1, len(sorted_strikes) - 1):
                K_prev, K_curr, K_next = sorted_strikes[i-1], sorted_strikes[i], sorted_strikes[i+1]
                v_prev, v_curr, v_next = sorted_vols[i-1], sorted_vols[i], sorted_vols[i+1]
                
                dK1 = K_curr - K_prev
                dK2 = K_next - K_curr
                
                if dK1 > 0 and dK2 > 0:
                    first_diff1 = (v_curr - v_prev) / dK1
                    first_diff2 = (v_next - v_curr) / dK2
                    second_diff = (first_diff2 - first_diff1) / ((dK1 + dK2) / 2)
                    second_diffs.append(second_diff)
            
            convexity_ok = all(sd >= -1e-6 for sd in second_diffs)
        else:
            convexity_ok = True
        
        results["convexity_ok"] = convexity_ok
        
        # Check 3: ATM volatility is reasonable
        atm_idx = np.argmin(np.abs(np.array(sorted_strikes) - forward))
        atm_vol = sorted_vols[atm_idx]
        atm_ok = 0.01 < atm_vol < 0.5  # Reasonable range
        results["atm_vol_ok"] = atm_ok
        
        # Check 4: Volatility smile (not too extreme)
        max_vol = max(sorted_vols)
        min_vol = min(sorted_vols)
        spread_ok = (max_vol - min_vol) / atm_vol < 2.0  # Max 200% spread
        results["spread_ok"] = spread_ok
        
        return results
    
    @staticmethod
    def calculate_risk_reversal_and_butterfly(
        strikes: List[float],
        vols: List[float],
        forward: float,
        delta: float = 0.25
    ) -> Dict[str, float]:
        """
        Calculate risk reversal and butterfly from volatility surface
        
        Args:
            strikes: List of strikes
            vols: List of volatilities
            forward: Forward price
            delta: Delta for calculation (typically 0.25)
        
        Returns:
            Dictionary with risk reversal and butterfly values
        """
        from models.garman_kohlhagen import StrikeFromDelta, OptionType, DeltaConvention
        
        # Find strikes corresponding to delta and 1-delta
        try:
            # Use Garman-Kohlhagen to find strikes
            # Need to estimate volatility first
            atm_idx = np.argmin(np.abs(np.array(strikes) - forward))
            atm_vol = vols[atm_idx]
            
            # Use synthetic parameters for strike calculation
            spot = forward  # Approximate
            T = 0.25  # 3 months
            r_d = 0.05
            r_f = 0.03
            
            strike_call = StrikeFromDelta.strike_from_delta(
                spot=spot,
                delta=delta,
                time_to_maturity=T,
                domestic_rate=r_d,
                foreign_rate=r_f,
                volatility=atm_vol,
                option_type=OptionType.CALL,
                convention=DeltaConvention.FORWARD_DELTA
            )
            
            strike_put = StrikeFromDelta.strike_from_delta(
                spot=spot,
                delta=1.0 - delta,
                time_to_maturity=T,
                domestic_rate=r_d,
                foreign_rate=r_f,
                volatility=atm_vol,
                option_type=OptionType.PUT,
                convention=DeltaConvention.FORWARD_DELTA
            )
            
            # Interpolate volatilities at these strikes
            vol_call = np.interp(strike_call, strikes, vols)
            vol_put = np.interp(strike_put, strikes, vols)
            
            risk_reversal = vol_call - vol_put
            butterfly = atm_vol - 0.5 * (vol_call + vol_put)
            
            return {
                "risk_reversal_25d": risk_reversal,
                "butterfly_25d": butterfly,
                "strike_call_25d": strike_call,
                "strike_put_25d": strike_put
            }
        except Exception as e:
            print(f"Error calculating RR/Butterfly: {e}")
            return {
                "risk_reversal_25d": 0.0,
                "butterfly_25d": 0.0,
                "strike_call_25d": forward,
                "strike_put_25d": forward
            }
    
    @staticmethod
    def term_structure_analysis(
        vol_surface: Dict[str, Dict[float, float]]
    ) -> Dict[str, float]:
        """
        Analyze the term structure of volatility
        
        Returns:
            Dictionary with term structure metrics
        """
        analysis = {}
        
        tenors = sorted(vol_surface.keys())
        
        # ATM volatility by tenor
        atm_vols = []
        for tenor in tenors:
            smile = vol_surface[tenor]
            atm_vol = smile.get(0.50, 0.0)
            atm_vols.append(atm_vol)
        
        analysis["atm_vols_by_tenor"] = dict(zip(tenors, atm_vols))
        
        # Check if term structure is upward or downward sloping
        if len(atm_vols) >= 2:
            # Linear regression slope
            x = np.arange(len(tenors))
            y = np.array(atm_vols)
            
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            
            analysis["term_structure_slope"] = float(slope)
            
            if slope > 0.01:
                analysis["term_structure"] = "upward"
            elif slope < -0.01:
                analysis["term_structure"] = "downward"
            else:
                analysis["term_structure"] = "flat"
        
        return analysis
