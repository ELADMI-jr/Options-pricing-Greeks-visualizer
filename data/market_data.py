"""
Market Data Module for FX Options
Pulls live spot rates, forward points, and constructs synthetic vol surfaces
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not available. Using synthetic data.")


class CurrencyPair(Enum):
    """Common FX currency pairs"""
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"
    USDJPY = "USDJPY"
    AUDUSD = "AUDUSD"
    USDCAD = "USDCAD"
    USDCHF = "USDCHF"
    NZDUSD = "NZDUSD"


@dataclass
class SpotRate:
    """Spot FX rate with timestamp"""
    pair: str
    rate: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class ForwardPoints:
    """Forward points for different tenors"""
    pair: str
    tenor: str  # e.g., "1M", "3M", "6M", "1Y"
    points: float  # Forward points (pips)
    timestamp: datetime


@dataclass
class InterestRate:
    """Interest rate data"""
    currency: str
    rate: float  # Annualized rate
    tenor: str
    timestamp: datetime


@dataclass
class MarketData:
    """Complete market data for FX options pricing"""
    spot: SpotRate
    domestic_rate: float  # Domestic risk-free rate
    foreign_rate: float   # Foreign risk-free rate
    forward_rates: Dict[str, float]  # Tenor -> forward rate
    vol_surface: Optional[Dict] = None  # Volatility surface data


class FXMarketData:
    """
    Fetch and manage FX market data
    Uses yfinance for spot rates, synthetic data for forward points and rates
    """
    
    # Typical forward point structure (synthetic data)
    # In practice, these would come from a data provider
    FORWARD_POINTS_TEMPLATE = {
        "EURUSD": {
            "1W": 2.5, "2W": 5.0, "1M": 10.0, "2M": 20.0,
            "3M": 30.0, "6M": 60.0, "9M": 90.0, "1Y": 120.0
        },
        "GBPUSD": {
            "1W": 3.0, "2W": 6.0, "1M": 12.0, "2M": 24.0,
            "3M": 36.0, "6M": 72.0, "9M": 108.0, "1Y": 144.0
        },
        "USDJPY": {
            "1W": -50.0, "2W": -100.0, "1M": -200.0, "2M": -400.0,
            "3M": -600.0, "6M": -1200.0, "9M": -1800.0, "1Y": -2400.0
        },
        "AUDUSD": {
            "1W": 1.5, "2W": 3.0, "1M": 6.0, "2M": 12.0,
            "3M": 18.0, "6M": 36.0, "9M": 54.0, "1Y": 72.0
        }
    }
    
    # Typical interest rate differentials (synthetic)
    RATE_DIFFERENTIALS = {
        "EURUSD": {"USD": 0.0525, "EUR": 0.0375},  # USD rate, EUR rate
        "GBPUSD": {"USD": 0.0525, "GBP": 0.0475},
        "USDJPY": {"USD": 0.0525, "JPY": 0.0025},
        "AUDUSD": {"USD": 0.0525, "AUD": 0.0425},
    }
    
    # Synthetic volatility surfaces (ATM + smile)
    # Format: {pair: {tenor: {delta: vol}}}
    VOL_SURFACE_TEMPLATE = {
        "EURUSD": {
            "1M": {0.10: 0.085, 0.25: 0.080, 0.50: 0.075, 0.75: 0.078, 0.90: 0.082},
            "3M": {0.10: 0.090, 0.25: 0.085, 0.50: 0.080, 0.75: 0.083, 0.90: 0.087},
            "6M": {0.10: 0.095, 0.25: 0.090, 0.50: 0.085, 0.75: 0.088, 0.90: 0.092},
            "1Y": {0.10: 0.100, 0.25: 0.095, 0.50: 0.090, 0.75: 0.093, 0.90: 0.097},
        },
        "GBPUSD": {
            "1M": {0.10: 0.095, 0.25: 0.090, 0.50: 0.085, 0.75: 0.088, 0.90: 0.092},
            "3M": {0.10: 0.100, 0.25: 0.095, 0.50: 0.090, 0.75: 0.093, 0.90: 0.097},
            "6M": {0.10: 0.105, 0.25: 0.100, 0.50: 0.095, 0.75: 0.098, 0.90: 0.102},
            "1Y": {0.10: 0.110, 0.25: 0.105, 0.50: 0.100, 0.75: 0.103, 0.90: 0.107},
        },
        "USDJPY": {
            "1M": {0.10: 0.120, 0.25: 0.115, 0.50: 0.110, 0.75: 0.113, 0.90: 0.117},
            "3M": {0.10: 0.125, 0.25: 0.120, 0.50: 0.115, 0.75: 0.118, 0.90: 0.122},
            "6M": {0.10: 0.130, 0.25: 0.125, 0.50: 0.120, 0.75: 0.123, 0.90: 0.127},
            "1Y": {0.10: 0.135, 0.25: 0.130, 0.50: 0.125, 0.75: 0.128, 0.90: 0.132},
        },
        "AUDUSD": {
            "1M": {0.10: 0.105, 0.25: 0.100, 0.50: 0.095, 0.75: 0.098, 0.90: 0.102},
            "3M": {0.10: 0.110, 0.25: 0.105, 0.50: 0.100, 0.75: 0.103, 0.90: 0.107},
            "6M": {0.10: 0.115, 0.25: 0.110, 0.50: 0.105, 0.75: 0.108, 0.90: 0.112},
            "1Y": {0.10: 0.120, 0.25: 0.115, 0.50: 0.110, 0.75: 0.113, 0.90: 0.117},
        }
    }
    
    def __init__(self, pair: str = "EURUSD"):
        self.pair = pair
        self.spot_cache: Optional[SpotRate] = None
        self.forward_cache: Dict[str, ForwardPoints] = {}
        self.rate_cache: Dict[str, InterestRate] = {}
    
    def get_spot(self, use_cache: bool = True) -> SpotRate:
        """
        Get current spot rate for the currency pair
        Uses yfinance if available, otherwise synthetic data
        """
        if use_cache and self.spot_cache is not None:
            # Cache valid for 5 minutes
            if (datetime.now() - self.spot_cache.timestamp).seconds < 300:
                return self.spot_cache
        
        if YFINANCE_AVAILABLE:
            try:
                # yfinance uses USD as base for most pairs
                # For EURUSD, we need EURUSD=X
                ticker = self._get_yfinance_ticker()
                data = yf.Ticker(ticker)
                hist = data.history(period="1d")
                
                if len(hist) > 0:
                    rate = float(hist['Close'].iloc[-1])
                    timestamp = datetime.now()
                    
                    # Get bid/ask if available
                    bid = float(hist['Low'].iloc[-1]) if 'Low' in hist.columns else None
                    ask = float(hist['High'].iloc[-1]) if 'High' in hist.columns else None
                    
                    self.spot_cache = SpotRate(
                        pair=self.pair,
                        rate=rate,
                        timestamp=timestamp,
                        bid=bid,
                        ask=ask
                    )
                    return self.spot_cache
            except Exception as e:
                print(f"Error fetching spot from yfinance: {e}")
        
        # Fallback to synthetic data
        synthetic_rates = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2750,
            "USDJPY": 149.50,
            "AUDUSD": 0.6550,
            "USDCAD": 1.3500,
            "USDCHF": 0.8850,
            "NZDUSD": 0.6150
        }
        
        rate = synthetic_rates.get(self.pair, 1.0)
        self.spot_cache = SpotRate(
            pair=self.pair,
            rate=rate,
            timestamp=datetime.now()
        )
        return self.spot_cache
    
    def _get_yfinance_ticker(self) -> str:
        """Convert currency pair to yfinance ticker format"""
        pair_map = {
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "AUDUSD": "AUDUSD=X",
            "USDCAD": "USDCAD=X",
            "USDCHF": "USDCHF=X",
            "NZDUSD": "NZDUSD=X"
        }
        return pair_map.get(self.pair, f"{self.pair}=X")
    
    def get_forward_points(self, tenor: str) -> ForwardPoints:
        """Get forward points for a specific tenor"""
        if tenor in self.forward_cache:
            return self.forward_cache[tenor]
        
        # Get from template
        points = self.FORWARD_POINTS_TEMPLATE.get(self.pair, {}).get(tenor, 0.0)
        
        fp = ForwardPoints(
            pair=self.pair,
            tenor=tenor,
            points=points,
            timestamp=datetime.now()
        )
        self.forward_cache[tenor] = fp
        return fp
    
    def get_interest_rates(self) -> Tuple[float, float]:
        """
        Get domestic and foreign interest rates
        Returns: (domestic_rate, foreign_rate)
        """
        rates = self.RATE_DIFFERENTIALS.get(self.pair, {})
        
        # Extract domestic and foreign currencies
        domestic, foreign = self._extract_currencies()
        
        domestic_rate = rates.get(domestic, 0.0525)
        foreign_rate = rates.get(foreign, 0.0375)
        
        return domestic_rate, foreign_rate
    
    def _extract_currencies(self) -> Tuple[str, str]:
        """Extract domestic and foreign currencies from pair"""
        # Convention: pair is in format BASE/QUOTE or BASEQUOTE
        # For FX options, domestic is typically the quote currency
        # But this depends on the desk's perspective
        
        # Simple approach: last 3 chars = domestic, first 3 = foreign
        if len(self.pair) == 6:
            foreign = self.pair[:3]
            domestic = self.pair[3:]
        else:
            # Handle pairs like EURUSD
            foreign = self.pair[:3]
            domestic = self.pair[3:]
        
        return domestic, foreign
    
    def get_forward_rate(self, tenor: str) -> float:
        """
        Calculate forward rate for a given tenor
        F = S + forward_points (in pips)
        """
        spot = self.get_spot().rate
        fp = self.get_forward_points(tenor)
        
        # Convert pips to rate units
        # For JPY pairs, 1 pip = 0.01, for others 1 pip = 0.0001
        if "JPY" in self.pair:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        forward_rate = spot + fp.points * pip_value
        return forward_rate
    
    def get_volatility_smile(self, tenor: str = "3M") -> Dict[float, float]:
        """
        Get volatility smile for a given tenor
        Returns: {delta: volatility}
        """
        vol_surface = self.VOL_SURFACE_TEMPLATE.get(self.pair, {})
        return vol_surface.get(tenor, {})
    
    def get_market_data(self, tenor: str = "3M") -> MarketData:
        """
        Get complete market data for pricing
        """
        spot = self.get_spot()
        domestic_rate, foreign_rate = self.get_interest_rates()
        
        # Calculate forward rates for different tenors
        forward_rates = {}
        for t in ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]:
            forward_rates[t] = self.get_forward_rate(t)
        
        # Get volatility surface
        vol_surface = self.VOL_SURFACE_TEMPLATE.get(self.pair, {})
        
        return MarketData(
            spot=spot,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            forward_rates=forward_rates,
            vol_surface=vol_surface
        )
    
    def get_atm_vol(self, tenor: str = "3M") -> float:
        """Get ATM volatility for a given tenor"""
        smile = self.get_volatility_smile(tenor)
        return smile.get(0.50, 0.08)  # Default to 8% if not found
    
    def get_risk_reversal(self, tenor: str = "3M", delta: float = 0.25) -> float:
        """Get risk reversal (call vol - put vol) at a given delta"""
        smile = self.get_volatility_smile(tenor)
        call_vol = smile.get(delta, 0.0)
        put_vol = smile.get(1.0 - delta, 0.0)
        return call_vol - put_vol
    
    def get_butterfly(self, tenor: str = "3M", delta: float = 0.25) -> float:
        """Get butterfly (ATM vol - average of call/put vol at delta)"""
        smile = self.get_volatility_smile(tenor)
        atm_vol = smile.get(0.50, 0.08)
        call_vol = smile.get(delta, 0.0)
        put_vol = smile.get(1.0 - delta, 0.0)
        return atm_vol - 0.5 * (call_vol + put_vol)


class SyntheticVolSurface:
    """
    Generate synthetic but realistic volatility surfaces
    Useful for testing and when live data is not available
    """
    
    @staticmethod
    def generate_smile(
        atm_vol: float = 0.08,
        skew: float = -0.02,
        smile_convexity: float = 0.05,
        deltas: List[float] = [0.10, 0.25, 0.50, 0.75, 0.90]
    ) -> Dict[float, float]:
        """
        Generate a volatility smile with skew and convexity
        
        Model: vol(delta) = ATM + skew * (delta - 0.5) + smile * (delta - 0.5)^2
        """
        smile = {}
        for delta in deltas:
            # Convert delta to signed delta (call delta positive, put delta negative)
            # For simplicity, we use absolute delta
            signed_delta = delta - 0.5
            vol = atm_vol + skew * signed_delta + smile_convexity * (signed_delta ** 2)
            smile[delta] = max(vol, 0.01)  # Ensure positive volatility
        
        return smile
    
    @staticmethod
    def generate_surface(
        tenors: List[str] = ["1M", "3M", "6M", "1Y"],
        atm_vols: Optional[Dict[str, float]] = None,
        skew: float = -0.02,
        smile_convexity: float = 0.05
    ) -> Dict[str, Dict[float, float]]:
        """
        Generate a full volatility surface across tenors
        
        Returns: {tenor: {delta: vol}}
        """
        if atm_vols is None:
            # Typical term structure: short-dated vol lower, long-dated higher
            atm_vols = {
                "1M": 0.075,
                "3M": 0.080,
                "6M": 0.085,
                "1Y": 0.090
            }
        
        surface = {}
        for tenor in tenors:
            atm = atm_vols.get(tenor, 0.08)
            surface[tenor] = SyntheticVolSurface.generate_smile(
                atm_vol=atm,
                skew=skew,
                smile_convexity=smile_convexity
            )
        
        return surface


class LiveDataFetcher:
    """
    Advanced live data fetcher using multiple sources
    """
    
    @staticmethod
    def get_fx_rates(pairs: List[str]) -> Dict[str, float]:
        """Get current FX rates for multiple pairs"""
        rates = {}
        
        if YFINANCE_AVAILABLE:
            for pair in pairs:
                try:
                    fetcher = FXMarketData(pair)
                    spot = fetcher.get_spot()
                    rates[pair] = spot.rate
                except Exception as e:
                    print(f"Error fetching {pair}: {e}")
                    # Use synthetic
                    synthetic = {
                        "EURUSD": 1.0850, "GBPUSD": 1.2750, "USDJPY": 149.50,
                        "AUDUSD": 0.6550, "USDCAD": 1.3500
                    }
                    rates[pair] = synthetic.get(pair, 1.0)
        else:
            synthetic = {
                "EURUSD": 1.0850, "GBPUSD": 1.2750, "USDJPY": 149.50,
                "AUDUSD": 0.6550, "USDCAD": 1.3500
            }
            for pair in pairs:
                rates[pair] = synthetic.get(pair, 1.0)
        
        return rates
    
    @staticmethod
    def get_historical_volatility(pair: str, tenor: str = "3M") -> float:
        """
        Estimate historical volatility from price history
        Uses yfinance to get historical data
        """
        if not YFINANCE_AVAILABLE:
            return 0.08  # Default
        
        try:
            ticker = FXMarketData(pair)._get_yfinance_ticker()
            data = yf.Ticker(ticker)
            
            # Get appropriate historical period
            period_map = {
                "1M": "30d",
                "3M": "90d",
                "6M": "180d",
                "1Y": "1y"
            }
            period = period_map.get(tenor, "90d")
            
            hist = data.history(period=period)
            
            if len(hist) < 10:
                return 0.08
            
            # Calculate daily log returns
            closes = hist['Close'].values
            log_returns = np.log(closes[1:] / closes[:-1])
            
            # Annualize volatility
            daily_vol = np.std(log_returns)
            annual_vol = daily_vol * np.sqrt(252)
            
            return max(annual_vol, 0.01)
            
        except Exception as e:
            print(f"Error calculating historical vol for {pair}: {e}")
            return 0.08


# ============================================================================
# Volatility Surface Interpolation
# ============================================================================

class VolSurfaceInterpolator:
    """
    Interpolate volatility surface for any strike/delta and tenor
    """
    
    def __init__(self, vol_surface: Dict[str, Dict[float, float]]):
        self.vol_surface = vol_surface
    
    def get_vol(self, tenor: str, delta: float) -> float:
        """
        Get volatility for a given tenor and delta
        Linear interpolation between available delta points
        """
        if tenor not in self.vol_surface:
            # Find closest tenor
            available_tenors = list(self.vol_surface.keys())
            if not available_tenors:
                return 0.08
            
            # Simple: use 3M as default
            tenor = "3M"
            if tenor not in self.vol_surface:
                tenor = available_tenors[0]
        
        smile = self.vol_surface[tenor]
        
        # Find closest deltas
        available_deltas = sorted(smile.keys())
        
        if delta <= available_deltas[0]:
            return smile[available_deltas[0]]
        elif delta >= available_deltas[-1]:
            return smile[available_deltas[-1]]
        
        # Linear interpolation
        for i in range(len(available_deltas) - 1):
            d1, d2 = available_deltas[i], available_deltas[i + 1]
            if d1 <= delta <= d2:
                v1, v2 = smile[d1], smile[d2]
                fraction = (delta - d1) / (d2 - d1)
                return v1 + fraction * (v2 - v1)
        
        return smile[available_deltas[0]]
    
    def get_vol_by_strike(self, tenor: str, strike: float, 
                         spot: float, time_to_maturity: float,
                         domestic_rate: float, foreign_rate: float,
                         volatility: float = 0.08) -> float:
        """
        Get volatility by strike (converts strike to delta first)
        """
        from models.garman_kohlhagen import StrikeFromDelta, OptionType, DeltaConvention
        
        # Calculate delta for ATM option at this strike
        # Use Garman-Kohlhagen to get delta
        from models.garman_kohlhagen import GarmanKohlhagen, FXOptionParams
        
        params_call = FXOptionParams(
            spot=spot,
            strike=strike,
            time_to_maturity=time_to_maturity,
            domestic_rate=domestic_rate,
            foreign_rate=foreign_rate,
            volatility=volatility,
            option_type=OptionType.CALL
        )
        
        model = GarmanKohlhagen(params_call)
        delta_call = model.delta()
        
        # Get vol at this delta
        return self.get_vol(tenor, delta_call)
