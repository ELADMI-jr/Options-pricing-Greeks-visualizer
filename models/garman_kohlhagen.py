"""
Garman-Kohlhagen Model for FX Options Pricing
Extended Black-Scholes for foreign exchange with two interest rates
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass
class FXOptionParams:
    """FX Option parameters in Garman-Kohlhagen framework"""
    spot: float          # Current spot FX rate (domestic/foreign)
    strike: float        # Strike price
    time_to_maturity: float  # Time to maturity in years
    domestic_rate: float     # Domestic risk-free rate (r_d)
    foreign_rate: float      # Foreign risk-free rate (r_f)
    volatility: float        # Implied volatility (sigma)
    option_type: OptionType  # CALL or PUT


@dataclass
class FXOptionResult:
    """Result container for FX option pricing"""
    premium: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho_domestic: float
    rho_foreign: float
    forward: float
    d1: float
    d2: float


class GarmanKohlhagen:
    """
    Garman-Kohlhagen Model (1983)
    FX options pricing with two interest rates
    
    Formula:
    call = S * e^(-r_f * T) * N(d1) - K * e^(-r_d * T) * N(d2)
    put  = K * e^(-r_d * T) * N(-d2) - S * e^(-r_f * T) * N(-d1)
    
    where:
    d1 = [ln(S/K) + (r_d - r_f + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    
    def __init__(self, params: FXOptionParams):
        self.params = params
        self._validate_params()
    
    def _validate_params(self):
        """Validate input parameters"""
        assert self.params.spot > 0, "Spot must be positive"
        assert self.params.strike > 0, "Strike must be positive"
        assert self.params.time_to_maturity > 0, "Time to maturity must be positive"
        assert self.params.volatility > 0, "Volatility must be positive"
    
    @property
    def forward(self) -> float:
        """Calculate forward FX rate: F = S * exp((r_d - r_f) * T)"""
        return self.params.spot * np.exp(
            (self.params.domestic_rate - self.params.foreign_rate) * 
            self.params.time_to_maturity
        )
    
    @property
    def d1(self) -> float:
        """Calculate d1 parameter"""
        S = self.params.spot
        K = self.params.strike
        T = self.params.time_to_maturity
        r_d = self.params.domestic_rate
        r_f = self.params.foreign_rate
        sigma = self.params.volatility
        
        numerator = np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T
        denominator = sigma * np.sqrt(T)
        return numerator / denominator
    
    @property
    def d2(self) -> float:
        """Calculate d2 parameter"""
        return self.d1 - self.params.volatility * np.sqrt(self.params.time_to_maturity)
    
    def price(self) -> float:
        """Calculate option premium using Garman-Kohlhagen formula"""
        S = self.params.spot
        K = self.params.strike
        T = self.params.time_to_maturity
        r_d = self.params.domestic_rate
        r_f = self.params.foreign_rate
        sigma = self.params.volatility
        
        d1 = self.d1
        d2 = self.d2
        
        if self.params.option_type == OptionType.CALL:
            # Call: S * e^(-r_f * T) * N(d1) - K * e^(-r_d * T) * N(d2)
            premium = (S * np.exp(-r_f * T) * norm.cdf(d1) - 
                      K * np.exp(-r_d * T) * norm.cdf(d2))
        else:
            # Put: K * e^(-r_d * T) * N(-d2) - S * e^(-r_f * T) * N(-d1)
            premium = (K * np.exp(-r_d * T) * norm.cdf(-d2) - 
                      S * np.exp(-r_f * T) * norm.cdf(-d1))
        
        return premium
    
    def delta(self) -> float:
        """Delta: Sensitivity to spot price (dPremium/dS)"""
        if self.params.option_type == OptionType.CALL:
            return np.exp(-self.params.foreign_rate * self.params.time_to_maturity) * norm.cdf(self.d1)
        else:
            return -np.exp(-self.params.foreign_rate * self.params.time_to_maturity) * norm.cdf(-self.d1)
    
    def gamma(self) -> float:
        """Gamma: Second derivative of premium w.r.t. spot"""
        S = self.params.spot
        T = self.params.time_to_maturity
        sigma = self.params.volatility
        
        d1 = self.d1
        # gamma = e^(-r_f * T) * N'(d1) / (S * sigma * sqrt(T))
        return (np.exp(-self.params.foreign_rate * T) * norm.pdf(d1) / 
                (S * sigma * np.sqrt(T)))
    
    def theta(self) -> float:
        """Theta: Sensitivity to time decay (dPremium/dT), per calendar day"""
        S = self.params.spot
        K = self.params.strike
        T = self.params.time_to_maturity
        r_d = self.params.domestic_rate
        r_f = self.params.foreign_rate
        sigma = self.params.volatility
        
        d1 = self.d1
        d2 = self.d2
        
        if self.params.option_type == OptionType.CALL:
            # Theta for call
            term1 = -S * np.exp(-r_f * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            term2 = -r_f * S * np.exp(-r_f * T) * norm.cdf(d1)
            term3 = r_d * K * np.exp(-r_d * T) * norm.cdf(d2)
            theta_per_year = term1 + term2 + term3
        else:
            # Theta for put
            term1 = -S * np.exp(-r_f * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            term2 = r_f * S * np.exp(-r_f * T) * norm.cdf(-d1)
            term3 = -r_d * K * np.exp(-r_d * T) * norm.cdf(-d2)
            theta_per_year = term1 + term2 + term3
        
        # Convert to per calendar day
        return theta_per_year / 365.0
    
    def vega(self) -> float:
        """Vega: Sensitivity to volatility (dPremium/dSigma)"""
        S = self.params.spot
        T = self.params.time_to_maturity
        
        d1 = self.d1
        # vega = S * e^(-r_f * T) * N'(d1) * sqrt(T)
        return S * np.exp(-self.params.foreign_rate * T) * norm.pdf(d1) * np.sqrt(T)
    
    def rho_domestic(self) -> float:
        """Rho: Sensitivity to domestic interest rate"""
        K = self.params.strike
        T = self.params.time_to_maturity
        
        d2 = self.d2
        
        if self.params.option_type == OptionType.CALL:
            return -K * T * np.exp(-self.params.domestic_rate * T) * norm.cdf(d2)
        else:
            return K * T * np.exp(-self.params.domestic_rate * T) * norm.cdf(-d2)
    
    def rho_foreign(self) -> float:
        """Rho: Sensitivity to foreign interest rate"""
        S = self.params.spot
        T = self.params.time_to_maturity
        
        d1 = self.d1
        
        if self.params.option_type == OptionType.CALL:
            return S * T * np.exp(-self.params.foreign_rate * T) * norm.cdf(d1)
        else:
            return -S * T * np.exp(-self.params.foreign_rate * T) * norm.cdf(-d1)
    
    def calculate_all(self) -> FXOptionResult:
        """Calculate all option metrics"""
        return FXOptionResult(
            premium=self.price(),
            delta=self.delta(),
            gamma=self.gamma(),
            theta=self.theta(),
            vega=self.vega(),
            rho_domestic=self.rho_domestic(),
            rho_foreign=self.rho_foreign(),
            forward=self.forward,
            d1=self.d1,
            d2=self.d2
        )
    
    @staticmethod
    def from_forward(forward: float, strike: float, time_to_maturity: float,
                    volatility: float, option_type: OptionType,
                    domestic_rate: float = 0.0) -> 'GarmanKohlhagen':
        """
        Create Garman-Kohlhagen instance from forward rate
        Useful when working with FX market conventions
        """
        # From forward: F = S * exp((r_d - r_f) * T)
        # We need to extract spot and foreign rate or make assumptions
        # For simplicity, we'll assume spot = forward (approximate for short T)
        # and foreign_rate = domestic_rate (symmetric)
        params = FXOptionParams(
            spot=forward,
            strike=strike,
            time_to_maturity=time_to_maturity,
            domestic_rate=domestic_rate,
            foreign_rate=domestic_rate,
            volatility=volatility,
            option_type=option_type
        )
        return GarmanKohlhagen(params)


# ============================================================================
# Delta-based quoting conventions
# ============================================================================

class DeltaConvention(Enum):
    """FX market delta quoting conventions"""
    SPOT_DELTA = "SPOT_DELTA"      # Delta w.r.t. spot (standard)
    FORWARD_DELTA = "FORWARD_DELTA"  # Delta w.r.t. forward (common in FX)


class StrikeFromDelta:
    """
    Convert between strike and delta for FX options
    FX desks quote volatility by delta, not by strike
    """
    
    @staticmethod
    def strike_from_delta(
        spot: float,
        delta: float,
        time_to_maturity: float,
        domestic_rate: float,
        foreign_rate: float,
        volatility: float,
        option_type: OptionType,
        convention: DeltaConvention = DeltaConvention.FORWARD_DELTA
    ) -> float:
        """
        Calculate strike from delta
        For FX options, this is the primary quoting method
        
        For ATM: delta = 0.5 for call, 0.5 for put (in forward delta convention)
        For 25-delta call: strike > forward (OTM call)
        For 25-delta put: strike < forward (OTM put)
        """
        # Iterative approach using Newton-Raphson
        # We need to find K such that delta(K) = target_delta
        
        F = spot * np.exp((domestic_rate - foreign_rate) * time_to_maturity)
        
        # Initial guess based on ATM
        strike_guess = F
        
        # For forward delta convention, we work with forward delta
        # Forward delta = exp(-r_f * T) * N(d1) for call
        # where d1 = [ln(F/K) + sigma^2/2 * T] / (sigma * sqrt(T))
        
        # Use scipy optimization for robustness
        from scipy.optimize import newton
        
        def forward_delta_func(K, target_delta, S, T, r_d, r_f, sigma, call=True):
            """Forward delta as function of strike"""
            F_val = S * np.exp((r_d - r_f) * T)
            d1 = (np.log(F_val / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
            
            if call:
                # Forward delta for call
                return np.exp(-r_f * T) * norm.cdf(d1) - target_delta
            else:
                # Forward delta for put
                return -np.exp(-r_f * T) * norm.cdf(-d1) - target_delta
        
        def spot_delta_func(K, target_delta, S, T, r_d, r_f, sigma, call=True):
            """Spot delta as function of strike"""
            d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            if call:
                return np.exp(-r_f * T) * norm.cdf(d1) - target_delta
            else:
                return -np.exp(-r_f * T) * norm.cdf(-d1) - target_delta
        
        is_call = option_type == OptionType.CALL
        
        if convention == DeltaConvention.FORWARD_DELTA:
            func = lambda K: forward_delta_func(
                K, delta, spot, time_to_maturity, 
                domestic_rate, foreign_rate, volatility, is_call
            )
        else:
            func = lambda K: spot_delta_func(
                K, delta, spot, time_to_maturity,
                domestic_rate, foreign_rate, volatility, is_call
            )
        
        # Use Newton-Raphson to find root
        try:
            strike = newton(func, strike_guess, maxiter=100, tol=1e-8)
            return float(strike)
        except RuntimeError:
            # Fallback: use bisection or return approximate
            # For very OTM options, use approximation
            if delta < 0.01:
                # Deep OTM
                if is_call:
                    return F * 1.5  # Far above forward for call
                else:
                    return F * 0.5  # Far below forward for put
            elif delta > 0.99:
                # Deep ITM
                if is_call:
                    return F * 0.5
                else:
                    return F * 1.5
            else:
                # Linear approximation around ATM
                return F
    
    @staticmethod
    def delta_from_strike(
        spot: float,
        strike: float,
        time_to_maturity: float,
        domestic_rate: float,
        foreign_rate: float,
        volatility: float,
        option_type: OptionType,
        convention: DeltaConvention = DeltaConvention.FORWARD_DELTA
    ) -> float:
        """Calculate delta from strike"""
        model = GarmanKohlhagen(FXOptionParams(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            volatility=volatility,
            option_type=option_type
        ))
        
        if convention == DeltaConvention.FORWARD_DELTA:
            # Forward delta = spot delta * exp(r_f * T)
            # Actually, forward delta is the delta of the option w.r.t. forward
            # In GK model: forward_delta = exp(-r_f * T) * N(d1) for call
            F = model.forward
            d1 = (np.log(F / strike) + 0.5 * volatility**2 * time_to_maturity) / \
                 (volatility * np.sqrt(time_to_maturity))
            
            if option_type == OptionType.CALL:
                return np.exp(-foreign_rate * time_to_maturity) * norm.cdf(d1)
            else:
                return -np.exp(-foreign_rate * time_to_maturity) * norm.cdf(-d1)
        else:
            return model.delta()
    
    @staticmethod
    def atm_strike(spot: float, time_to_maturity: float,
                   domestic_rate: float, foreign_rate: float) -> float:
        """ATM strike in FX convention (usually = forward)"""
        return spot * np.exp((domestic_rate - foreign_rate) * time_to_maturity)


# ============================================================================
# Market quote conventions
# ============================================================================

@dataclass
class MarketQuote:
    """FX options market quote (volatility by delta)"""
    delta: float           # Delta (e.g., 0.25 for 25-delta)
    call_vol: Optional[float] = None  # Call volatility at this delta
    put_vol: Optional[float] = None   # Put volatility at this delta
    
    @property
    def is_straddle(self) -> bool:
        return self.delta == 0.5 and self.call_vol == self.put_vol


class VolatilitySmile:
    """
    FX volatility smile representation
    Stores volatility by delta (market convention)
    """
    
    def __init__(self):
        self.quotes: list[MarketQuote] = []
    
    def add_quote(self, quote: MarketQuote):
        self.quotes.append(quote)
    
    def get_vol_at_delta(self, delta: float, option_type: OptionType) -> Optional[float]:
        """Get volatility at a specific delta"""
        for quote in self.quotes:
            if abs(quote.delta - delta) < 1e-6:
                if option_type == OptionType.CALL:
                    return quote.call_vol
                else:
                    return quote.put_vol
        return None
    
    def risk_reversal(self, delta: float) -> Optional[float]:
        """
        Risk Reversal = Call Vol - Put Vol at same delta
        Measures skew of the volatility smile
        """
        for quote in self.quotes:
            if abs(quote.delta - delta) < 1e-6:
                if quote.call_vol is not None and quote.put_vol is not None:
                    return quote.call_vol - quote.put_vol
        return None
    
    def butterfly(self, delta: float) -> Optional[float]:
        """
        Butterfly = ATM Vol - 0.5 * (Call Vol + Put Vol) at delta
        Measures convexity/smile of volatility
        """
        atm_vol = self.get_vol_at_delta(0.5, OptionType.CALL)
        if atm_vol is None:
            return None
        
        for quote in self.quotes:
            if abs(quote.delta - delta) < 1e-6:
                if quote.call_vol is not None and quote.put_vol is not None:
                    return atm_vol - 0.5 * (quote.call_vol + quote.put_vol)
        return None


# ============================================================================
# Finite Difference Greeks
# ============================================================================

class FiniteDifferenceGreeks:
    """
    Calculate Greeks using finite difference (bump-and-reprice)
    """
    
    def __init__(self, base_params: FXOptionParams, bump_size: float = 0.01):
        self.base_params = base_params
        self.bump_size = bump_size
        self.base_model = GarmanKohlhagen(base_params)
    
    def delta_fd(self) -> float:
        """Delta via central finite difference"""
        bump = self.bump_size * self.base_params.spot
        
        # Bump spot up
        params_up = FXOptionParams(
            spot=self.base_params.spot + bump,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_up = GarmanKohlhagen(params_up).price()
        
        # Bump spot down
        params_down = FXOptionParams(
            spot=self.base_params.spot - bump,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_down = GarmanKohlhagen(params_down).price()
        
        return (premium_up - premium_down) / (2 * bump)
    
    def gamma_fd(self) -> float:
        """Gamma via central finite difference"""
        bump = self.bump_size * self.base_params.spot
        
        # Bump spot up
        params_up = FXOptionParams(
            spot=self.base_params.spot + bump,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_up = GarmanKohlhagen(params_up).price()
        
        # Bump spot down
        params_down = FXOptionParams(
            spot=self.base_params.spot - bump,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_down = GarmanKohlhagen(params_down).price()
        
        # Bump spot up twice
        params_up2 = FXOptionParams(
            spot=self.base_params.spot + 2 * bump,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_up2 = GarmanKohlhagen(params_up2).price()
        
        # Second derivative: (f(x+h) - 2*f(x) + f(x-h)) / h^2
        return (premium_up - 2 * self.base_model.price() + premium_down) / (bump**2)
    
    def vega_fd(self) -> float:
        """Vega via central finite difference"""
        bump = self.bump_size * 0.01  # 1% vol bump
        
        # Bump vol up
        params_up = FXOptionParams(
            spot=self.base_params.spot,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility + bump,
            option_type=self.base_params.option_type
        )
        premium_up = GarmanKohlhagen(params_up).price()
        
        # Bump vol down
        params_down = FXOptionParams(
            spot=self.base_params.spot,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility - bump,
            option_type=self.base_params.option_type
        )
        premium_down = GarmanKohlhagen(params_down).price()
        
        return (premium_up - premium_down) / (2 * bump)
    
    def theta_fd(self) -> float:
        """Theta via central finite difference (per calendar day)"""
        bump = self.bump_size / 365.0  # 1 day bump
        
        # Bump time up
        params_up = FXOptionParams(
            spot=self.base_params.spot,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity + bump,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_up = GarmanKohlhagen(params_up).price()
        
        # Bump time down
        params_down = FXOptionParams(
            spot=self.base_params.spot,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity - bump,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_down = GarmanKohlhagen(params_down).price()
        
        # Negative because theta is -dPremium/dT
        return -(premium_up - premium_down) / (2 * bump)
    
    def rho_domestic_fd(self) -> float:
        """Rho (domestic rate) via finite difference"""
        bump = self.bump_size * 0.01  # 10bp bump
        
        params_up = FXOptionParams(
            spot=self.base_params.spot,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate + bump,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_up = GarmanKohlhagen(params_up).price()
        
        params_down = FXOptionParams(
            spot=self.base_params.spot,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity,
            domestic_rate=self.base_params.domestic_rate - bump,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility,
            option_type=self.base_params.option_type
        )
        premium_down = GarmanKohlhagen(params_down).price()
        
        return (premium_up - premium_down) / (2 * bump)


# ============================================================================
# Monte Carlo Greeks
# ============================================================================

class MonteCarloGreeks:
    """
    Calculate Greeks using Monte Carlo with pathwise estimators
    """
    
    def __init__(self, base_params: FXOptionParams, 
                 n_simulations: int = 100000, 
                 n_steps: int = 100,
                 seed: Optional[int] = None):
        self.base_params = base_params
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.seed = seed
        self.base_model = GarmanKohlhagen(base_params)
    
    def _generate_paths(self, spot: float, volatility: float, 
                       drift: float, T: float) -> np.ndarray:
        """Generate geometric Brownian motion paths"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        dt = T / self.n_steps
        paths = np.zeros((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = spot
        
        for t in range(1, self.n_steps + 1):
            z = np.random.standard_normal(self.n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt + 
                volatility * np.sqrt(dt) * z
            )
        
        return paths
    
    def _payoff(self, spot: float, strike: float, 
                option_type: OptionType) -> float:
        """Calculate option payoff at maturity"""
        if option_type == OptionType.CALL:
            return max(spot - strike, 0)
        else:
            return max(strike - spot, 0)
    
    def price_mc(self) -> Tuple[float, float]:
        """
        Monte Carlo price with confidence interval
        Returns: (price, std_error)
        """
        S = self.base_params.spot
        K = self.base_params.strike
        T = self.base_params.time_to_maturity
        r_d = self.base_params.domestic_rate
        r_f = self.base_params.foreign_rate
        sigma = self.base_params.volatility
        
        # Drift in risk-neutral measure for FX
        drift = r_d - r_f
        
        paths = self._generate_paths(S, sigma, drift, T)
        final_spots = paths[:, -1]
        
        payoffs = np.array([self._payoff(s, K, self.base_params.option_type) 
                           for s in final_spots])
        
        # Discount factor
        discount = np.exp(-r_d * T)
        
        # Present value
        pv = discount * payoffs
        
        price = np.mean(pv)
        std_error = np.std(pv) / np.sqrt(self.n_simulations)
        
        return price, std_error
    
    def delta_mc(self) -> Tuple[float, float]:
        """
        Delta via pathwise estimator (likelihood ratio method)
        Returns: (delta, std_error)
        """
        S = self.base_params.spot
        K = self.base_params.strike
        T = self.base_params.time_to_maturity
        r_d = self.base_params.domestic_rate
        r_f = self.base_params.foreign_rate
        sigma = self.base_params.volatility
        
        drift = r_d - r_f
        
        paths = self._generate_paths(S, sigma, drift, T)
        final_spots = paths[:, -1]
        
        payoffs = np.array([self._payoff(s, K, self.base_params.option_type) 
                           for s in final_spots])
        
        # For pathwise delta: dP/dS = exp(-r_d * T) * E[payoff * (W_T / (S * sigma * sqrt(T)))]
        # where W_T is the terminal Brownian motion
        
        # Generate the Brownian motion increments
        dt = T / self.n_steps
        W_T = np.sum(np.random.standard_normal((self.n_simulations, self.n_steps)) * 
                    np.sqrt(dt), axis=1)
        
        # Pathwise estimator for delta
        if self.base_params.option_type == OptionType.CALL:
            # For call: delta = exp(-r_f * T) * N(d1) approximately
            # Pathwise: delta_mc = exp(-r_d * T) * mean(payoff * W_T / (S * sigma * sqrt(T)))
            indicator = final_spots > K
        else:
            indicator = final_spots < K
        
        # Likelihood ratio estimator
        # delta = exp(-r_d * T) * cov(payoff, W_T) / (S * sigma * sqrt(T) * var(W_T))
        # But simpler: use the fact that dS/S = sigma * dW
        
        # Simplified pathwise for delta
        discount = np.exp(-r_d * T)
        
        # For call option: delta = exp(-r_f * T) * N(d1)
        # We'll use finite difference within MC for robustness
        # Pathwise requires careful implementation
        
        # Use likelihood ratio method
        # delta = E[ payoff * (W_T) / (S * sigma * sqrt(T)) ] * exp(-r_d * T)
        
        # W_T for each path (already generated in _generate_paths)
        # Re-generate with same seed for consistency
        if self.seed is not None:
            np.random.seed(self.seed)
        
        dt = T / self.n_steps
        W_increments = np.random.standard_normal((self.n_simulations, self.n_steps))
        W_T = np.sum(W_increments * np.sqrt(dt), axis=1)
        
        # Calculate payoffs again
        final_spots = S * np.exp(
            (drift - 0.5 * sigma**2) * T + sigma * W_T
        )
        payoffs = np.array([self._payoff(s, K, self.base_params.option_type) 
                           for s in final_spots])
        
        discount = np.exp(-r_d * T)
        
        if self.base_params.option_type == OptionType.CALL:
            # Pathwise delta for call
            delta_estimator = discount * np.mean(payoffs * W_T / (S * sigma * np.sqrt(T)))
        else:
            # For put, need to be careful with the sign
            delta_estimator = discount * np.mean(payoffs * W_T / (S * sigma * np.sqrt(T)))
        
        # Calculate standard error
        if self.base_params.option_type == OptionType.CALL:
            values = discount * payoffs * W_T / (S * sigma * np.sqrt(T))
        else:
            values = discount * payoffs * W_T / (S * sigma * np.sqrt(T))
        
        std_error = np.std(values) / np.sqrt(self.n_simulations)
        
        return delta_estimator, std_error
    
    def vega_mc(self) -> Tuple[float, float]:
        """
        Vega via likelihood ratio method
        """
        S = self.base_params.spot
        K = self.base_params.strike
        T = self.base_params.time_to_maturity
        r_d = self.base_params.domestic_rate
        r_f = self.base_params.foreign_rate
        sigma = self.base_params.volatility
        
        drift = r_d - r_f
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        dt = T / self.n_steps
        W_increments = np.random.standard_normal((self.n_simulations, self.n_steps))
        W_T = np.sum(W_increments * np.sqrt(dt), axis=1)
        
        final_spots = S * np.exp(
            (drift - 0.5 * sigma**2) * T + sigma * W_T
        )
        payoffs = np.array([self._payoff(s, K, self.base_params.option_type) 
                           for s in final_spots])
        
        discount = np.exp(-r_d * T)
        
        # Vega pathwise: E[ payoff * (W_T^2 - T) / sigma ] * exp(-r_d * T)
        # For Black-Scholes, vega = S * exp(-r_f * T) * N'(d1) * sqrt(T)
        
        vega_estimator = discount * np.mean(payoffs * (W_T**2 - T) / sigma)
        
        values = discount * payoffs * (W_T**2 - T) / sigma
        std_error = np.std(values) / np.sqrt(self.n_simulations)
        
        return vega_estimator, std_error


# ============================================================================
# P&L Attribution Module
# ============================================================================

@dataclass
class Scenario:
    """Market move scenario for P&L attribution"""
    spot_move: float       # Absolute change in spot
    vol_move: float        # Absolute change in volatility
    time_move: float = 0.0  # Change in time (days)


@dataclass
class PLAttribution:
    """P&L decomposition result"""
    total_pl: float
    delta_pl: float
    gamma_pl: float
    vega_pl: float
    theta_pl: float
    rho_pl: float
    residual: float


class PLAttributionCalculator:
    """
    P&L attribution using Taylor expansion
    Decomposes option P&L into Greeks contributions
    """
    
    def __init__(self, base_params: FXOptionParams):
        self.base_params = base_params
        self.base_model = GarmanKohlhagen(base_params)
        self.base_result = self.base_model.calculate_all()
    
    def calculate_pl(self, scenario: Scenario) -> PLAttribution:
        """
        Calculate P&L and decompose into Greeks contributions
        
        Taylor expansion:
        dP ≈ delta * dS + 0.5 * gamma * (dS)^2 + vega * dSigma + theta * dT
        """
        # Calculate new premium
        new_params = FXOptionParams(
            spot=self.base_params.spot + scenario.spot_move,
            strike=self.base_params.strike,
            time_to_maturity=self.base_params.time_to_maturity + scenario.time_move/365.0,
            domestic_rate=self.base_params.domestic_rate,
            foreign_rate=self.base_params.foreign_rate,
            volatility=self.base_params.volatility + scenario.vol_move,
            option_type=self.base_params.option_type
        )
        new_model = GarmanKohlhagen(new_params)
        new_premium = new_model.price()
        
        total_pl = new_premium - self.base_result.premium
        
        # Greeks contributions
        delta_pl = self.base_result.delta * scenario.spot_move
        gamma_pl = 0.5 * self.base_result.gamma * (scenario.spot_move**2)
        vega_pl = self.base_result.vega * scenario.vol_move
        theta_pl = self.base_result.theta * scenario.time_move
        
        # Rho contribution (assuming domestic rate unchanged)
        rho_pl = 0.0
        
        # Residual (higher order terms)
        residual = total_pl - (delta_pl + gamma_pl + vega_pl + theta_pl + rho_pl)
        
        return PLAttribution(
            total_pl=total_pl,
            delta_pl=delta_pl,
            gamma_pl=gamma_pl,
            vega_pl=vega_pl,
            theta_pl=theta_pl,
            rho_pl=rho_pl,
            residual=residual
        )
    
    def full_decomposition(self, scenario: Scenario) -> dict:
        """Full P&L attribution report"""
        pl = self.calculate_pl(scenario)
        
        return {
            "scenario": {
                "spot_move": scenario.spot_move,
                "vol_move": scenario.vol_move,
                "time_move_days": scenario.time_move
            },
            "pl_attribution": {
                "total_pl": pl.total_pl,
                "delta_contribution": pl.delta_pl,
                "gamma_contribution": pl.gamma_pl,
                "vega_contribution": pl.vega_pl,
                "theta_contribution": pl.theta_pl,
                "residual": pl.residual
            },
            "base_premium": self.base_result.premium,
            "new_premium": self.base_result.premium + pl.total_pl,
            "greeks": {
                "delta": self.base_result.delta,
                "gamma": self.base_result.gamma,
                "vega": self.base_result.vega,
                "theta": self.base_result.theta
            }
        }
