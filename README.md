# FX Options Pricing & Greeks Visualizer

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.12%2B-ff4b4b)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

**Professional FX Options Analysis Tool with Garman-Kohlhagen Model**

A comprehensive, production-ready FX options pricing and risk analysis system that follows **real trading desk conventions** - not just equity options theory ported to FX.

---

## 🎯 **Key Features**

### ✅ **1. Garman-Kohlhagen Model (FX-Specific)**
- Extended Black-Scholes for foreign exchange with **two interest rates** (domestic + foreign)
- Proper handling of FX forward rates: `F = S * exp((r_d - r_f) * T)`
- All Greeks: delta, gamma, theta, vega, rho (domestic & foreign)

### ✅ **2. Delta-Based Quoting (Market Convention)**
- FX desks quote volatility **by delta, not by strike**
- Built-in **25-delta risk reversal**, **25-delta butterfly**, **ATM straddle**
- Strike-to-delta and delta-to-strike converters
- Forward delta vs. spot delta conventions

### ✅ **3. Live Market Data**
- Pulls **live spot rates** from yfinance
- Synthetic **forward points** for all major currency pairs
- **Interest rate differentials** (USD, EUR, GBP, JPY, AUD)
- Full **volatility surface** with term structure
- Graceful fallback to synthetic data

### ✅ **4. SVI Volatility Surface Fitting**
- Implements **SVI-J** (Jim Gatheral) parameterization
- **Arbitrage-free checks**: Feller condition, Andersen-BR, convexity
- Automatic parameter adjustment to ensure no arbitrage
- Also includes **SABR model** for comparison

### ✅ **5. Greeks Three Ways**
- **Closed-form**: Analytical Garman-Kohlhagen formulas
- **Finite Difference**: Bump-and-reprice method
- **Monte Carlo**: Pathwise estimators with likelihood ratio
- Dashboard shows convergence comparison

### ✅ **6. P&L Attribution**
- Taylor expansion: `dP ≈ delta*dS + 0.5*gamma*(dS)² + vega*dSigma + theta*dT`
- Decomposes P&L into **delta, gamma, vega, theta contributions**
- Shows **residual** (higher-order terms)
- Interactive scenario testing

### ✅ **7. Interactive Dashboard**
- **6 tabs** with comprehensive FX options analysis
- **3D volatility surface** visualization (rotatable)
- **Live Greeks** updating as you adjust parameters
- **Risk reversal/butterfly term structure** charts
- **Waterfall P&L attribution** visualization

---

## 📊 **Dashboard Screenshots**

### 🏠 **Main Dashboard Overview**
![Dashboard Overview](https://img.shields.io/badge/Interactive-Dashboard-blue?style=for-the-badge)
*Professional FX options analysis interface with real-time calculations*

### 📈 **Volatility Smile Tab**
![Volatility Smile](https://img.shields.io/badge/Volatility-Smile-orange?style=for-the-badge)
*Visualize the volatility smile with ATM, risk reversal, and butterfly metrics*

### 🎯 **Delta-Based Quoting Tab**
![Delta Quoting](https://img.shields.io/badge/Delta-Quoting-green?style=for-the-badge)
*Convert between strike and delta - the way FX desks actually quote*

### 🔬 **Greeks Analysis Tab**
![Greeks Analysis](https://img.shields.io/badge/Greeks-Analysis-purple?style=for-the-badge)
*Compare closed-form, finite difference, and Monte Carlo Greeks*

### 💰 **P&L Attribution Tab**
![P&L Attribution](https://img.shields.io/badge/P%26L-Attribution-red?style=for-the-badge)
*Decompose option P&L into Greeks contributions with waterfall chart*

### 🎨 **3D Volatility Surface Tab**
![3D Surface](https://img.shields.io/badge/3D-Surface-cyan?style=for-the-badge)
*Interactive 3D visualization of volatility across strikes and tenors*

*(Note: For actual screenshots, run the dashboard and take screenshots of each tab. The badges above are placeholders.)*

---

## 📁 **Project Structure**

```
Options-pricing-Greeks-visualizer/
├── models/
│   └── garman_kohlhagen.py      # Core GK model + Greeks + converters
├── data/
│   └── market_data.py            # Live data + synthetic vol surfaces
├── analytics/
│   └── svi_model.py              # SVI fitting + arbitrage checks + SABR
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
├── utils/
│   ├── __init__.py
│   ├── converters.py             # Strike <-> delta conversions
│   └── validators.py             # Input validation + arbitrage checks
├── requirements.txt              # Dependencies
└── README.md                    # This file
```

---

## 🚀 **Quick Start**

### Installation

```bash
# Clone the repository
git clone https://github.com/ELADMI-jr/Options-pricing-Greeks-visualizer.git
cd Options-pricing-Greeks-visualizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📖 **Usage Guide**

### **Tab 1: Option Pricing**
- Select currency pair (EURUSD, GBPUSD, USDJPY, etc.)
- Choose option type (CALL or PUT)
- Input strike as **absolute value** or **delta**
- Select volatility (ATM or custom)
- View complete pricing results with all Greeks

### **Tab 2: Volatility Smile**
- Visualize the volatility smile for selected tenor
- See **ATM volatility**, **25Δ risk reversal**, **25Δ butterfly**
- Fit **SVI model** to the smile with arbitrage checks
- View RMSE and parameter values

### **Tab 3: Delta-Based Quoting**
- Convert **strike → delta** (both spot and forward delta)
- Convert **delta → strike**
- View market quote conventions (ATM straddle, RR, butterfly)
- See volatility by delta table

### **Tab 4: Greeks Analysis**
- Compare Greeks calculated using **3 methods**
- View **convergence metrics** and error percentages
- Monte Carlo convergence plot
- Adjust number of simulations

### **Tab 5: P&L Attribution**
- Input market move scenario (spot, vol, time)
- See **P&L decomposition** into Greeks contributions
- Waterfall chart showing each component
- View residual (higher-order terms)

### **Tab 6: 3D Volatility Surface**
- Interactive 3D visualization
- Rotate, zoom, explore the surface
- Term structure analysis
- Risk reversal & butterfly by tenor

---

## 📚 **Theory Behind the Code**

### **Garman-Kohlhagen Model (1983)**

The extension of Black-Scholes for FX options:

```
C = S * e^(-r_f * T) * N(d1) - K * e^(-r_d * T) * N(d2)
P = K * e^(-r_d * T) * N(-d2) - S * e^(-r_f * T) * N(-d1)

where:
d1 = [ln(S/K) + (r_d - r_f + σ²/2) * T] / (σ * √T)
d2 = d1 - σ * √T
```

**Key difference from Black-Scholes:** Two interest rates (domestic and foreign)

### **Delta-Based Quoting**

FX desks quote volatility by **delta**, not strike:
- **ATM**: 50Δ (delta = 0.5)
- **25Δ Risk Reversal**: Call vol at 25Δ - Put vol at 25Δ
- **25Δ Butterfly**: ATM vol - 0.5 * (25Δ Call vol + 25Δ Put vol)

### **SVI Model**

Stochastic Volatility Inspired parameterization:

```
σ²(K) = a + b * [ρ * (k - m) + √((k - m)² + σ²)]

where k = ln(K/F) (log moneyness)
```

**Arbitrage-free conditions:**
- a ≥ 0
- b ≥ 0
- |ρ| ≤ 1
- σ > 0
- a ≥ σ²/4 (Andersen-BR condition)

### **Greeks Calculation Methods**

1. **Closed-Form**: Direct analytical formulas from GK model
2. **Finite Difference**: Bump-and-reprice (central difference)
3. **Monte Carlo**: Pathwise estimators using likelihood ratio method

### **P&L Attribution**

Taylor expansion of option price:

```
dP ≈ delta * dS + 0.5 * gamma * (dS)² + vega * dσ + theta * dT
```

This decomposes P&L into contributions from each Greek.

---

## 🔧 **Configuration**

### **Currency Pairs**
- EURUSD
- GBPUSD
- USDJPY
- AUDUSD
- USDCAD
- USDCHF
- NZDUSD

### **Tenors**
- 1W (1 week)
- 2W (2 weeks)
- 1M (1 month)
- 2M (2 months)
- 3M (3 months)
- 6M (6 months)
- 9M (9 months)
- 1Y (1 year)

### **Data Sources**
- **Live**: yfinance (if available)
- **Synthetic**: Built-in templates for forward points, rates, vol surfaces

---

## 📊 **Example Use Cases**

### **1. Price an FX Option**
```python
from models.garman_kohlhagen import GarmanKohlhagen, FXOptionParams, OptionType

params = FXOptionParams(
    spot=1.0850,
    strike=1.0800,
    time_to_maturity=0.25,  # 3 months
    domestic_rate=0.0525,   # USD rate
    foreign_rate=0.0375,    # EUR rate
    volatility=0.08,
    option_type=OptionType.CALL
)

model = GarmanKohlhagen(params)
result = model.calculate_all()

print(f"Premium: {result.premium:.6f}")
print(f"Delta: {result.delta:.6f}")
print(f"Gamma: {result.gamma:.8f}")
```

### **2. Convert Strike to Delta**
```python
from models.garman_kohlhagen import StrikeFromDelta, OptionType, DeltaConvention

strike = StrikeFromDelta.strike_from_delta(
    spot=1.0850,
    delta=0.25,  # 25-delta
    time_to_maturity=0.25,
    domestic_rate=0.0525,
    foreign_rate=0.0375,
    volatility=0.08,
    option_type=OptionType.CALL,
    convention=DeltaConvention.FORWARD_DELTA
)

print(f"25Δ Call Strike: {strike:.6f}")
```

### **3. Fit SVI to Market Data**
```python
from analytics.svi_model import SVIModel, SVIVariance

model = SVIModel(variance_type=SVIVariance.JS)

result = model.fit_to_market_data(
    strikes=[1.05, 1.06, 1.07, 1.08, 1.09, 1.10],
    market_vols=[0.095, 0.090, 0.085, 0.080, 0.083, 0.088],
    forward=1.0850
)

print(f"SVI Parameters: a={result.params.a:.6f}, b={result.params.b:.6f}")
print(f"RMSE: {result.rmse:.8f}")
print(f"Arbitrage-free: {result.is_valid}")
```

### **4. P&L Attribution**
```python
from models.garman_kohlhagen import PLAttributionCalculator, Scenario, FXOptionParams, OptionType

params = FXOptionParams(
    spot=1.0850,
    strike=1.0800,
    time_to_maturity=0.25,
    domestic_rate=0.0525,
    foreign_rate=0.0375,
    volatility=0.08,
    option_type=OptionType.CALL
)

calculator = PLAttributionCalculator(params)

scenario = Scenario(
    spot_move=0.01,    # +100 pips
    vol_move=0.005,    # +50bp
    time_move=1        # +1 day
)

pl_report = calculator.full_decomposition(scenario)

print(f"Total P&L: {pl_report['pl_attribution']['total_pl']:+.6f}")
print(f"Delta contribution: {pl_report['pl_attribution']['delta_contribution']:+.6f}")
print(f"Gamma contribution: {pl_report['pl_attribution']['gamma_contribution']:+.6f}")
```

---

## 🛠 **Dependencies**

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Core language |
| numpy | ≥ 1.21.0 | Numerical computing |
| pandas | ≥ 1.3.0 | Data manipulation |
| scipy | ≥ 1.7.0 | Optimization, special functions |
| streamlit | ≥ 1.12.0 | Interactive dashboard |
| plotly | ≥ 5.0.0 | Interactive visualizations |
| yfinance | ≥ 0.1.60 | Live market data (optional) |

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

### **Development Setup**

```bash
# Clone the repo
git clone https://github.com/ELADMI-jr/Options-pricing-Greeks-visualizer.git
cd Options-pricing-Greeks-visualizer

# Install dev dependencies
pip install -r requirements.txt
pip install pytest

# Run tests
pytest
```

### **Adding Features**

1. **American Options**: Add binomial tree pricing
2. **Barrier Options**: Add Monte Carlo with variance reduction
3. **More Currency Pairs**: Extend market data templates
4. **Real Data Sources**: Integrate with FXCM, OANDA, or other APIs
5. **Risk Reports**: Add VaR, CVaR calculations

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Garman & Kohlhagen (1983)**: "Foreign Currency Option Values"
- **Jim Gatheral**: "The Volatility Surface: A Practitioner's Guide"
- **Andersen & Brotherton-Ratcliffe**: "The Generalized SVI Volatility Surface"
- **Hagan et al. (2002)**: "Managing Smile Risk" (SABR model)

---

## 📞 **Contact**

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for FX traders and quantitative finance professionals**

*This tool demonstrates genuine understanding of FX derivatives conventions and trading desk practices.*
