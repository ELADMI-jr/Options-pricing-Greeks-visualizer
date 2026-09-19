"""
Streamlit Dashboard for FX Options Pricing & Greeks Visualizer
Interactive web interface for FX options analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.garman_kohlhagen import (
    GarmanKohlhagen, FXOptionParams, FXOptionResult, OptionType,
    StrikeFromDelta, DeltaConvention, VolatilitySmile, MarketQuote
)
from data.market_data import FXMarketData, SyntheticVolSurface, VolSurfaceInterpolator
from analytics.svi_model import SVIModel, SVIParams, SVIResult, SVIFitter, SVIVariance
from models.garman_kohlhagen import FiniteDifferenceGreeks, MonteCarloGreeks
from models.garman_kohlhagen import PLAttributionCalculator, Scenario


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="FX Options Pricing & Greeks Visualizer",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

def format_currency(value: float, pair: str = "EURUSD") -> str:
    """Format currency value based on pair"""
    if "JPY" in pair:
        return f"¥{value:,.2f}"
    else:
        return f"${value:,.4f}"


def format_pips(value: float, pair: str = "EURUSD") -> str:
    """Format pips"""
    if "JPY" in pair:
        return f"{value:,.0f} pips"
    else:
        return f"{value:,.2f} pips"


def calculate_option_metrics(
    spot: float,
    strike: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    option_type: OptionType
) -> Tuple[FXOptionResult, Dict]:
    """Calculate all option metrics"""
    params = FXOptionParams(
        spot=spot,
        strike=strike,
        time_to_maturity=T,
        domestic_rate=r_d,
        foreign_rate=r_f,
        volatility=vol,
        option_type=option_type
    )
    
    model = GarmanKohlhagen(params)
    result = model.calculate_all()
    
    # Finite difference Greeks
    fd_greeks = FiniteDifferenceGreeks(params, bump_size=0.001)
    
    # Monte Carlo Greeks (quick calculation)
    mc_greeks = MonteCarloGreeks(params, n_simulations=10000, seed=42)
    
    greeks_comparison = {
        "delta": {
            "closed_form": result.delta,
            "finite_diff": fd_greeks.delta_fd(),
            "mc": mc_greeks.delta_mc()[0] if hasattr(mc_greeks, 'delta_mc') else None
        },
        "gamma": {
            "closed_form": result.gamma,
            "finite_diff": fd_greeks.gamma_fd(),
        },
        "vega": {
            "closed_form": result.vega,
            "finite_diff": fd_greeks.vega_fd(),
        },
        "theta": {
            "closed_form": result.theta,
            "finite_diff": fd_greeks.theta_fd(),
        }
    }
    
    return result, greeks_comparison


def create_volatility_smile_plot(
    strikes: List[float],
    vols: List[float],
    forward: float,
    atm_vol: float,
    rr: float,
    bf: float
) -> go.Figure:
    """Create volatility smile plot"""
    fig = go.Figure()
    
    # Add volatility curve
    fig.add_trace(go.Scatter(
        x=strikes,
        y=vols,
        mode='lines+markers',
        name='Market Volatility',
        line=dict(color='royalblue', width=3),
        marker=dict(size=8)
    ))
    
    # Add ATM marker
    fig.add_trace(go.Scatter(
        x=[forward],
        y=[atm_vol],
        mode='markers',
        name='ATM',
        marker=dict(size=12, color='green', symbol='star')
    ))
    
    # Add annotations
    fig.add_annotation(
        x=forward, y=atm_vol,
        text=f"ATM: {atm_vol:.2%}",
        showarrow=True,
        arrowhead=1,
        ax=20,
        ay=-30
    )
    
    # Add Risk Reversal and Butterfly annotations
    fig.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"Risk Reversal (25Δ): {rr:+.4f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.7)",
        borderpad=4
    )
    
    fig.add_annotation(
        x=0.05, y=0.90,
        xref="paper", yref="paper",
        text=f"Butterfly (25Δ): {bf:+.4f}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.7)",
        borderpad=4
    )
    
    fig.update_layout(
        title="Volatility Smile",
        xaxis_title="Strike",
        yaxis_title="Implied Volatility",
        height=500,
        hovermode='x unified'
    )
    
    # Add vertical line at forward
    fig.add_vline(x=forward, line_dash="dash", line_color="green", 
                  annotation_text="Forward")
    
    return fig


def create_greeks_comparison_plot(greeks_comparison: Dict) -> go.Figure:
    """Create Greeks comparison plot"""
    fig = go.Figure()
    
    methods = list(greeks_comparison["delta"].keys())
    greek_names = list(greeks_comparison.keys())
    
    for greek in greek_names:
        values = [greeks_comparison[greek][m] for m in methods if greeks_comparison[greek][m] is not None]
        
        if len(values) > 0:
            fig.add_trace(go.Bar(
                x=[greek.capitalize()],
                y=[np.mean(values)],
                name=greek.capitalize(),
                error_y=dict(type='data', array=[np.std(values) if len(values) > 1 else 0])
            ))
    
    fig.update_layout(
        title="Greeks Comparison: Closed-Form vs Finite Difference",
        xaxis_title="Greek",
        yaxis_title="Value",
        height=400,
        barmode='group'
    )
    
    return fig


def create_pl_attribution_plot(pl_report: Dict) -> go.Figure:
    """Create P&L attribution waterfall plot"""
    pl_data = pl_report["pl_attribution"]
    
    categories = ['Delta', 'Gamma', 'Vega', 'Theta', 'Residual']
    values = [
        pl_data["delta_contribution"],
        pl_data["gamma_contribution"],
        pl_data["vega_contribution"],
        pl_data["theta_contribution"],
        pl_data["residual"]
    ]
    
    # Colors based on sign
    colors = ['green' if v >= 0 else 'red' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Waterfall(
        orientation="v",
        measure=["relative"] * 5,
        x=categories,
        y=values,
        text=[f"{v:+.6f}" for v in values],
        textposition="outside",
        marker=dict(color=colors),
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="P&L Attribution Decomposition",
        xaxis_title="Component",
        yaxis_title="P&L Contribution",
        height=400,
        showlegend=False
    )
    
    return fig


def create_svi_fit_plot(
    strikes: List[float],
    market_vols: List[float],
    svi_result: SVIResult,
    forward: float
) -> go.Figure:
    """Create SVI fit comparison plot"""
    fig = go.Figure()
    
    # Market vols
    fig.add_trace(go.Scatter(
        x=strikes,
        y=market_vols,
        mode='markers',
        name='Market',
        marker=dict(size=10, color='blue')
    ))
    
    # SVI fit
    svi_vols = [svi_result.fitted_vols[s] for s in strikes]
    fig.add_trace(go.Scatter(
        x=strikes,
        y=svi_vols,
        mode='lines+markers',
        name='SVI Fit',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    # ATM
    atm_vol = svi_result.params.a**0.5 if svi_result.params.a > 0 else 0
    fig.add_vline(x=forward, line_dash="dash", line_color="green",
                  annotation_text="Forward")
    
    fig.update_layout(
        title=f"SVI Fit (RMSE: {svi_result.rmse:.6f})",
        xaxis_title="Strike",
        yaxis_title="Volatility",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_3d_vol_surface_plot(
    vol_surface: Dict[str, Dict[float, float]],
    pair: str
) -> go.Figure:
    """Create 3D volatility surface plot"""
    tenors = sorted(vol_surface.keys())
    deltas = sorted(list(set(d for d in vol_surface[t].keys() for t in tenors)))
    
    # Create meshgrid
    tenor_idx = {t: i for i, t in enumerate(tenors)}
    delta_idx = {d: i for i, d in enumerate(deltas)}
    
    X = []
    Y = []
    Z = []
    
    for t in tenors:
        for d in deltas:
            vol = vol_surface[t].get(d, 0.0)
            X.append(tenor_idx[t])
            Y.append(delta_idx[d])
            Z.append(vol)
    
    fig = go.Figure(data=[go.Surface(
        z=np.array(Z).reshape(len(tenors), len(deltas)),
        x=np.arange(len(tenors)),
        y=np.arange(len(deltas)),
        colorscale='Viridis',
        showscale=True
    )])
    
    fig.update_layout(
        title=f"3D Volatility Surface - {pair}",
        scene=dict(
            xaxis_title='Tenor',
            yaxis_title='Delta',
            zaxis_title='Volatility',
            xaxis=dict(ticktext=tenors, tickvals=list(range(len(tenors)))),
            yaxis=dict(ticktext=[f"{d:.0%}" for d in deltas], tickvals=list(range(len(deltas))))
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<div class="main-header">💱 FX Options Pricing & Greeks Visualizer</div>', 
                unsafe_allow_html=True)
    st.markdown("### Professional FX Options Analysis with Garman-Kohlhagen Model")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Currency pair selection
    pair = st.sidebar.selectbox(
        "Currency Pair",
        ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
        index=0
    )
    
    # Data source
    use_live_data = st.sidebar.checkbox("Use Live Data (yfinance)", value=False)
    
    # Initialize market data
    market_data = FXMarketData(pair)
    
    # Get spot rate
    spot_data = market_data.get_spot()
    spot = spot_data.rate
    
    # Get interest rates
    r_d, r_f = market_data.get_interest_rates()
    
    # Tenor selection
    tenor = st.sidebar.selectbox("Tenor", ["1M", "3M", "6M", "1Y"], index=1)
    
    # Get forward rate
    forward = market_data.get_forward_rate(tenor)
    
    # Time to maturity
    tenor_to_T = {"1M": 1/12, "3M": 3/12, "6M": 6/12, "1Y": 1.0}
    T = tenor_to_T[tenor]
    
    # Volatility surface
    vol_smile = market_data.get_volatility_smile(tenor)
    
    # ========================================================================
    # Tabs
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Option Pricing",
        "📈 Volatility Smile",
        "🎯 Delta-Based Quoting",
        "🔬 Greeks Analysis",
        "💰 P&L Attribution",
        "🎨 3D Vol Surface"
    ])
    
    # ========================================================================
    # Tab 1: Option Pricing
    # ========================================================================
    with tab1:
        st.header("Option Pricing")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            option_type = st.radio("Option Type", ["CALL", "PUT"], horizontal=True)
        
        with col2:
            # Strike input
            strike_input_method = st.radio("Strike Input", ["Absolute", "Delta"], horizontal=True)
        
        with col3:
            if strike_input_method == "Absolute":
                strike = st.number_input("Strike", value=forward, min_value=0.01, step=0.0001)
            else:
                delta_input = st.number_input("Delta", value=0.5, min_value=0.01, max_value=0.99, step=0.01)
                strike = StrikeFromDelta.strike_from_delta(
                    spot=spot,
                    delta=delta_input,
                    time_to_maturity=T,
                    domestic_rate=r_d,
                    foreign_rate=r_f,
                    volatility=vol_smile.get(0.50, 0.08),
                    option_type=OptionType.CALL if option_type == "CALL" else OptionType.PUT,
                    convention=DeltaConvention.FORWARD_DELTA
                )
                st.write(f"Equivalent Strike: {strike:.6f}")
        
        # Volatility selection
        col1, col2 = st.columns(2)
        
        with col1:
            vol_input_method = st.radio("Volatility Input", ["ATM", "Custom"], horizontal=True)
        
        with col2:
            if vol_input_method == "ATM":
                vol = vol_smile.get(0.50, 0.08)
                st.write(f"ATM Vol: {vol:.2%}")
            else:
                vol = st.number_input("Volatility", value=0.08, min_value=0.01, max_value=2.0, step=0.001)
        
        # Calculate option metrics
        option_type_enum = OptionType.CALL if option_type == "CALL" else OptionType.PUT
        result, greeks_comparison = calculate_option_metrics(
            spot=spot,
            strike=strike,
            T=T,
            r_d=r_d,
            r_f=r_f,
            vol=vol,
            option_type=option_type_enum
        )
        
        # Display results
        st.subheader("Pricing Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Premium", f"{result.premium:.6f}")
            st.metric("Forward", f"{forward:.6f}")
        
        with col2:
            st.metric("Delta", f"{result.delta:.6f}")
            st.metric("Gamma", f"{result.gamma:.8f}")
        
        with col3:
            st.metric("Vega", f"{result.vega:.6f}")
            st.metric("Theta", f"{result.theta:.6f}")
        
        with col4:
            st.metric("Rho (Dom)", f"{result.rho_domestic:.6f}")
            st.metric("Rho (For)", f"{result.rho_foreign:.6f}")
        
        # Intermediate parameters
        with st.expander("📋 Intermediate Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"d1: {result.d1:.6f}")
                st.write(f"d2: {result.d2:.6f}")
            with col2:
                st.write(f"K/F: {strike/forward:.6f}")
                st.write(f"ln(S/K): {np.log(spot/strike):.6f}")
        
        # Greeks comparison
        with st.expander("🔍 Greeks Comparison"):
            df_greeks = pd.DataFrame({
                "Method": ["Closed-Form", "Finite Diff"],
                "Delta": [greeks_comparison["delta"]["closed_form"], greeks_comparison["delta"]["finite_diff"]],
                "Gamma": [greeks_comparison["gamma"]["closed_form"], greeks_comparison["gamma"]["finite_diff"]],
                "Vega": [greeks_comparison["vega"]["closed_form"], greeks_comparison["vega"]["finite_diff"]],
                "Theta": [greeks_comparison["theta"]["closed_form"], greeks_comparison["theta"]["finite_diff"]]
            })
            st.dataframe(df_greeks.style.format("{:.8f}"), use_container_width=True)
    
    # ========================================================================
    # Tab 2: Volatility Smile
    # ========================================================================
    with tab2:
        st.header("Volatility Smile Analysis")
        
        # Get volatility smile data
        deltas = sorted(vol_smile.keys())
        vols = [vol_smile[d] for d in deltas]
        
        # Convert deltas to strikes for plotting
        strikes_plot = []
        for d in deltas:
            strike_call = StrikeFromDelta.strike_from_delta(
                spot=spot,
                delta=d,
                time_to_maturity=T,
                domestic_rate=r_d,
                foreign_rate=r_f,
                volatility=vol_smile.get(0.50, 0.08),
                option_type=OptionType.CALL,
                convention=DeltaConvention.FORWARD_DELTA
            )
            strikes_plot.append(strike_call)
        
        # Calculate metrics
        atm_vol = vol_smile.get(0.50, 0.08)
        rr_25d = market_data.get_risk_reversal(tenor, 0.25)
        bf_25d = market_data.get_butterfly(tenor, 0.25)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ATM Volatility", f"{atm_vol:.2%}")
        with col2:
            st.metric("25Δ Risk Reversal", f"{rr_25d:+.4f}", delta_color="normal")
        with col3:
            st.metric("25Δ Butterfly", f"{bf_25d:+.4f}", delta_color="normal")
        
        # Plot
        fig_smile = create_volatility_smile_plot(
            strikes=strikes_plot,
            vols=vols,
            forward=forward,
            atm_vol=atm_vol,
            rr=rr_25d,
            bf=bf_25d
        )
        st.plotly_chart(fig_smile, use_container_width=True)
        
        # SVI Fit
        with st.expander("🎯 Fit SVI Model to Smile"):
            if st.button("Fit SVI"):
                with st.spinner("Fitting SVI model..."):
                    # Prepare data for SVI fit
                    strikes_for_fit = np.linspace(0.8 * forward, 1.2 * forward, 20)
                    
                    # Interpolate market vols
                    market_vols_for_fit = []
                    for s in strikes_for_fit:
                        # Find closest delta
                        k = np.log(s / forward)
                        # Use linear interpolation
                        vol = np.interp(k, 
                                       [np.log(strikes_plot[i] / forward) for i in range(len(strikes_plot))],
                                       vols)
                        market_vols_for_fit.append(max(vol, 0.01))
                    
                    # Fit SVI
                    svi_model = SVIModel(variance_type=SVIVariance.JS)
                    svi_result = svi_model.fit_to_market_data(
                        strikes=list(strikes_for_fit),
                        market_vols=market_vols_for_fit,
                        forward=forward
                    )
                    
                    # Display results
                    st.subheader("SVI Fit Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Parameters:**")
                        st.write(f"a (level): {svi_result.params.a:.8f}")
                        st.write(f"b (slope): {svi_result.params.b:.8f}")
                        st.write(f"rho (corr): {svi_result.params.rho:.8f}")
                    with col2:
                        st.write(f"m (skew): {svi_result.params.m:.8f}")
                        st.write(f"sigma (convexity): {svi_result.params.sigma:.8f}")
                        st.write(f"RMSE: {svi_result.rmse:.8f}")
                    
                    # Arbitrage check
                    if svi_result.is_valid:
                        st.success("✅ No arbitrage violations detected")
                    else:
                        st.error("❌ Arbitrage violations found:")
                        for v in svi_result.arbitrage_violations:
                            st.write(f"- {v}")
                    
                    # Plot fit
                    fig_svi = create_svi_fit_plot(
                        strikes=list(strikes_for_fit),
                        market_vols=market_vols_for_fit,
                        svi_result=svi_result,
                        forward=forward
                    )
                    st.plotly_chart(fig_svi, use_container_width=True)
    
    # ========================================================================
    # Tab 3: Delta-Based Quoting
    # ========================================================================
    with tab3:
        st.header("Delta-Based Quoting (Market Convention)")
        
        st.markdown("""
        FX options desks quote volatility by delta, not by strike.
        This tab converts between strike and delta conventions.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strike → Delta")
            
            strike_input = st.number_input("Strike", value=forward, min_value=0.01, step=0.0001, key="strike_input")
            
            # Calculate delta
            delta_spot = StrikeFromDelta.delta_from_strike(
                spot=spot,
                strike=strike_input,
                time_to_maturity=T,
                domestic_rate=r_d,
                foreign_rate=r_f,
                volatility=vol_smile.get(0.50, 0.08),
                option_type=OptionType.CALL,
                convention=DeltaConvention.SPOT_DELTA
            )
            
            delta_forward = StrikeFromDelta.delta_from_strike(
                spot=spot,
                strike=strike_input,
                time_to_maturity=T,
                domestic_rate=r_d,
                foreign_rate=r_f,
                volatility=vol_smile.get(0.50, 0.08),
                option_type=OptionType.CALL,
                convention=DeltaConvention.FORWARD_DELTA
            )
            
            st.write(f"**Spot Delta:** {delta_spot:.6f}")
            st.write(f"**Forward Delta:** {delta_forward:.6f}")
            
            # Volatility at this delta
            vol_at_delta = vol_smile.get(max(0.01, min(0.99, delta_forward)), 0.08)
            st.write(f"**Market Vol at {delta_forward:.0%}Δ:** {vol_at_delta:.2%}")
        
        with col2:
            st.subheader("Delta → Strike")
            
            delta_input = st.number_input("Delta", value=0.25, min_value=0.01, max_value=0.99, step=0.01, key="delta_input")
            option_type_delta = st.radio("Option Type", ["CALL", "PUT"], horizontal=True, key="option_type_delta")
            
            # Calculate strike
            strike_from_delta = StrikeFromDelta.strike_from_delta(
                spot=spot,
                delta=delta_input,
                time_to_maturity=T,
                domestic_rate=r_d,
                foreign_rate=r_f,
                volatility=vol_smile.get(0.50, 0.08),
                option_type=OptionType.CALL if option_type_delta == "CALL" else OptionType.PUT,
                convention=DeltaConvention.FORWARD_DELTA
            )
            
            st.write(f"**Strike:** {strike_from_delta:.6f}")
            
            # Volatility at this delta
            vol_at_delta2 = vol_smile.get(delta_input, 0.08)
            st.write(f"**Market Vol at {delta_input:.0%}Δ:** {vol_at_delta2:.2%}")
        
        # Market quote display
        st.subheader("Market Quote Convention")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ATM Straddle", f"{vol_smile.get(0.50, 0.08):.2%}")
        with col2:
            st.metric("25Δ Risk Reversal", f"{rr_25d:+.4f}")
        with col3:
            st.metric("25Δ Butterfly", f"{bf_25d:+.4f}")
        
        # Volatility by delta table
        st.subheader("Volatility by Delta")
        
        delta_table = pd.DataFrame({
            "Delta": [f"{d:.0%}" for d in sorted(vol_smile.keys())],
            "Call Vol": [vol_smile.get(d, 0.0) for d in sorted(vol_smile.keys())],
            "Put Vol": [vol_smile.get(1.0 - d, 0.0) for d in sorted(vol_smile.keys())]
        })
        
        st.dataframe(delta_table.style.format({"Call Vol": "{:.2%}", "Put Vol": "{:.2%}"}), 
                     use_container_width=True)
    
    # ========================================================================
    # Tab 4: Greeks Analysis
    # ========================================================================
    with tab4:
        st.header("Greeks Analysis")
        
        st.markdown("""
        Compare Greeks calculated using three different methods:
        1. **Closed-Form**: Analytical formulas from Garman-Kohlhagen
        2. **Finite Difference**: Bump-and-reprice method
        3. **Monte Carlo**: Pathwise estimators with likelihood ratio method
        """)
        
        # Option parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            option_type_greeks = st.selectbox("Option Type", ["CALL", "PUT"], index=0, key="option_type_greeks")
        
        with col2:
            strike_greeks = st.number_input("Strike", value=forward, min_value=0.01, step=0.0001, key="strike_greeks")
        
        with col3:
            vol_greeks = st.number_input("Volatility", value=vol_smile.get(0.50, 0.08), min_value=0.01, step=0.001, key="vol_greeks")
        
        # Calculate all Greeks
        option_type_enum = OptionType.CALL if option_type_greeks == "CALL" else OptionType.PUT
        params = FXOptionParams(
            spot=spot,
            strike=strike_greeks,
            time_to_maturity=T,
            domestic_rate=r_d,
            foreign_rate=r_f,
            volatility=vol_greeks,
            option_type=option_type_enum
        )
        
        model = GarmanKohlhagen(params)
        result = model.calculate_all()
        
        # Finite difference
        fd = FiniteDifferenceGreeks(params, bump_size=0.001)
        
        # Monte Carlo
        n_simulations = st.slider("MC Simulations", 1000, 100000, 50000, 10000, key="mc_simulations")
        mc = MonteCarloGreeks(params, n_simulations=n_simulations, seed=42)
        
        # Calculate
        with st.spinner("Calculating Greeks..."):
            delta_fd = fd.delta_fd()
            gamma_fd = fd.gamma_fd()
            vega_fd = fd.vega_fd()
            theta_fd = fd.theta_fd()
            
            delta_mc, _ = mc.delta_mc()
            vega_mc, _ = mc.vega_mc()
            mc_price, mc_std = mc.price_mc()
        
        # Display results
        st.subheader("Greeks Comparison")
        
        # Create comparison table
        comparison_data = {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Price"],
            "Closed-Form": [
                result.delta,
                result.gamma,
                result.vega,
                result.theta,
                result.premium
            ],
            "Finite Diff": [
                delta_fd,
                gamma_fd,
                vega_fd,
                theta_fd,
                result.premium  # FD price same as closed-form
            ],
            "Monte Carlo": [
                delta_mc,
                "N/A",  # Gamma MC not implemented
                vega_mc,
                "N/A",  # Theta MC not implemented
                mc_price
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison.style.format("{:.8f}"), use_container_width=True)
        
        # Error metrics
        st.subheader("Convergence Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            delta_error = abs(result.delta - delta_fd) / abs(result.delta) * 100 if result.delta != 0 else 0
            vega_error = abs(result.vega - vega_fd) / abs(result.vega) * 100 if result.vega != 0 else 0
            
            st.metric("Delta FD Error", f"{delta_error:.4f}%")
            st.metric("Vega FD Error", f"{vega_error:.4f}%")
        
        with col2:
            delta_mc_error = abs(result.delta - delta_mc) / abs(result.delta) * 100 if result.delta != 0 else 0
            vega_mc_error = abs(result.vega - vega_mc) / abs(result.vega) * 100 if result.vega != 0 else 0
            price_mc_error = abs(result.premium - mc_price) / abs(result.premium) * 100 if result.premium != 0 else 0
            
            st.metric("Delta MC Error", f"{delta_mc_error:.4f}%")
            st.metric("Vega MC Error", f"{vega_mc_error:.4f}%")
            st.metric("Price MC Error", f"{price_mc_error:.4f}%")
        
        # Convergence plot
        st.subheader("Monte Carlo Convergence")
        
        # Run MC with different simulation counts
        n_simulations_list = [1000, 5000, 10000, 50000, 100000]
        mc_prices = []
        
        for n in n_simulations_list:
            mc_temp = MonteCarloGreeks(params, n_simulations=n, seed=42)
            price, _ = mc_temp.price_mc()
            mc_prices.append(price)
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=n_simulations_list,
            y=mc_prices,
            mode='lines+markers',
            name='MC Price'
        ))
        fig_conv.add_hline(y=result.premium, line_dash="dash", line_color="red",
                          annotation_text="Closed-Form Price")
        
        fig_conv.update_layout(
            title="Monte Carlo Price Convergence",
            xaxis_title="Number of Simulations",
            yaxis_title="Price",
            height=400
        )
        
        st.plotly_chart(fig_conv, use_container_width=True)
    
    # ========================================================================
    # Tab 5: P&L Attribution
    # ========================================================================
    with tab5:
        st.header("P&L Attribution")
        
        st.markdown("""
        Decompose option P&L into Greeks contributions using Taylor expansion.
        This shows how much of the P&L comes from delta, gamma, vega, theta.
        """)
        
        # Base option parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            option_type_pl = st.selectbox("Option Type", ["CALL", "PUT"], index=0, key="option_type_pl")
        
        with col2:
            strike_pl = st.number_input("Strike", value=forward, min_value=0.01, step=0.0001, key="strike_pl")
        
        with col3:
            vol_pl = st.number_input("Volatility", value=vol_smile.get(0.50, 0.08), min_value=0.01, step=0.001, key="vol_pl")
        
        # Calculate base Greeks
        option_type_enum = OptionType.CALL if option_type_pl == "CALL" else OptionType.PUT
        base_params = FXOptionParams(
            spot=spot,
            strike=strike_pl,
            time_to_maturity=T,
            domestic_rate=r_d,
            foreign_rate=r_f,
            volatility=vol_pl,
            option_type=option_type_enum
        )
        
        pl_calculator = PLAttributionCalculator(base_params)
        base_result = pl_calculator.base_model.calculate_all()
        
        # Display base Greeks
        st.subheader("Base Greeks")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Premium", f"{base_result.premium:.6f}")
            st.metric("Delta", f"{base_result.delta:.6f}")
        with col2:
            st.metric("Gamma", f"{base_result.gamma:.8f}")
            st.metric("Vega", f"{base_result.vega:.6f}")
        with col3:
            st.metric("Theta", f"{base_result.theta:.6f}")
        with col4:
            st.metric("Forward", f"{forward:.6f}")
        
        # Scenario inputs
        st.subheader("Market Move Scenario")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spot_move = st.number_input("Spot Move", value=0.01, step=0.0001, key="spot_move")
            spot_move_pct = st.number_input("Spot Move %", value=0.0, step=0.1, key="spot_move_pct")
            actual_spot_move = spot * spot_move_pct / 100 + spot_move
        
        with col2:
            vol_move = st.number_input("Vol Move (bp)", value=0, step=1, key="vol_move_bp")
            vol_move_abs = vol_move / 10000  # Convert bp to absolute
        
        with col3:
            time_move = st.number_input("Time Move (days)", value=0, step=1, key="time_move")
        
        # Calculate P&L
        scenario = Scenario(
            spot_move=actual_spot_move,
            vol_move=vol_move_abs,
            time_move=time_move
        )
        
        if st.button("Calculate P&L", key="calculate_pl"):
            with st.spinner("Calculating P&L..."):
                pl_report = pl_calculator.full_decomposition(scenario)
            
            # Display results
            st.subheader("P&L Attribution Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total P&L", f"{pl_report['pl_attribution']['total_pl']:+.6f}")
            with col2:
                st.metric("New Premium", f"{pl_report['new_premium']:.6f}")
            with col3:
                st.metric("Residual", f"{pl_report['pl_attribution']['residual']:+.6f}")
            
            # Detailed breakdown
            st.subheader("Contribution Breakdown")
            
            breakdown_data = {
                "Component": ["Delta", "Gamma", "Vega", "Theta", "Residual"],
                "Contribution": [
                    pl_report['pl_attribution']['delta_contribution'],
                    pl_report['pl_attribution']['gamma_contribution'],
                    pl_report['pl_attribution']['vega_contribution'],
                    pl_report['pl_attribution']['theta_contribution'],
                    pl_report['pl_attribution']['residual']
                ]
            }
            df_breakdown = pd.DataFrame(breakdown_data)
            st.dataframe(df_breakdown.style.format({"Contribution": "{:+.8f}"}), 
                         use_container_width=True)
            
            # Plot
            fig_pl = create_pl_attribution_plot(pl_report)
            st.plotly_chart(fig_pl, use_container_width=True)
            
            # Scenario summary
            st.subheader("Scenario Summary")
            st.write(f"**Spot:** {spot:.6f} → {spot + actual_spot_move:.6f} ({spot_move_pct:+.2f}%)")
            st.write(f"**Volatility:** {vol_pl:.2%} → {vol_pl + vol_move_abs:.2%} ({vol_move:+.0f}bp)")
            st.write(f"**Time:** {T:.2f} years → {T + time_move/365:.2f} years ({time_move:+.0f} days)")
    
    # ========================================================================
    # Tab 6: 3D Volatility Surface
    # ========================================================================
    with tab6:
        st.header("3D Volatility Surface")
        
        st.markdown("""
        Interactive 3D visualization of the volatility surface across strikes/deltas and tenors.
        Rotate and zoom to explore the smile and term structure.
        """)
        
        # Get full volatility surface
        vol_surface_full = market_data.VOL_SURFACE_TEMPLATE.get(pair, {})
        
        if vol_surface_full:
            # Create 3D plot
            fig_3d = create_3d_vol_surface_plot(vol_surface_full, pair)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("No volatility surface data available for this pair.")
        
        # Term structure analysis
        st.subheader("Term Structure Analysis")
        
        analysis = VolSurfaceAnalytics.term_structure_analysis(vol_surface_full)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Term Structure", analysis.get("term_structure", "N/A"))
            st.metric("Slope", f"{analysis.get('term_structure_slope', 0):.6f}")
        
        with col2:
            atm_vols_by_tenor = analysis.get("atm_vols_by_tenor", {})
            if atm_vols_by_tenor:
                st.write("**ATM Vol by Tenor:**")
                for tenor, vol in atm_vols_by_tenor.items():
                    st.write(f"{tenor}: {vol:.2%}")
        
        # Risk Reversal and Butterfly by tenor
        st.subheader("Risk Reversal & Butterfly by Tenor")
        
        rr_bf_data = []
        for tenor in sorted(vol_surface_full.keys()):
            rr = market_data.get_risk_reversal(tenor, 0.25)
            bf = market_data.get_butterfly(tenor, 0.25)
            rr_bf_data.append({
                "Tenor": tenor,
                "Risk Reversal (25Δ)": rr,
                "Butterfly (25Δ)": bf
            })
        
        df_rr_bf = pd.DataFrame(rr_bf_data)
        st.dataframe(df_rr_bf.style.format({"Risk Reversal (25Δ)": "{:+.4f}", "Butterfly (25Δ)": "{:+.4f}"}),
                     use_container_width=True)
        
        # Plot RR and Butterfly
        fig_rr_bf = go.Figure()
        
        fig_rr_bf.add_trace(go.Bar(
            x=[d["Tenor"] for d in rr_bf_data],
            y=[d["Risk Reversal (25Δ)"] for d in rr_bf_data],
            name="Risk Reversal",
            marker_color='lightblue'
        ))
        
        fig_rr_bf.add_trace(go.Bar(
            x=[d["Tenor"] for d in rr_bf_data],
            y=[d["Butterfly (25Δ)"] for d in rr_bf_data],
            name="Butterfly",
            marker_color='lightcoral'
        ))
        
        fig_rr_bf.update_layout(
            title="Risk Reversal & Butterfly by Tenor",
            xaxis_title="Tenor",
            yaxis_title="Value",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_rr_bf, use_container_width=True)
    
    # ========================================================================
    # Footer
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>FX Options Pricing & Greeks Visualizer | Garman-Kohlhagen Model</p>
        <p>Built with Streamlit • Python • Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
