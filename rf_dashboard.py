"""
rf_dashboard.py — Random Forest Direction + Range Dashboard
=============================================================
Standalone Streamlit tab for the Moses project.

Run standalone:  streamlit run rf_dashboard.py

Shows:
  • Direction signal (UP/DOWN + confidence %)
  • Expected range in points
  • Iron condor wing placement
  • Multi-horizon consensus table
  • Feature importance bars
  • Model health (accuracy per regime split)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# ── Standalone page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOSES · RF Signal",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Reuse existing JUDAH dark theme aesthetics ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
.stApp { font-family:'Inter',sans-serif !important; background:linear-gradient(180deg,#0a0c14,#06080e); color:#d1d5e0; }
.block-container { padding:1.5rem 2.5rem 2rem !important; max-width:1400px !important; }
.glass { background:rgba(15,18,34,0.7); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:22px; margin-bottom:16px; }
.glass-sm { background:rgba(20,24,40,0.5); border:1px solid rgba(255,255,255,0.05); border-radius:10px; padding:14px 16px; }
.sec-label { font-size:0.62rem; letter-spacing:0.2em; text-transform:uppercase; color:#4a5270; font-weight:600; margin-bottom:10px; }
.mono { font-family:'JetBrains Mono',monospace !important; }
.tag-up { background:rgba(34,197,94,0.1); color:#22c55e; border:1px solid rgba(34,197,94,0.25); border-radius:20px; padding:3px 10px; font-size:0.72rem; font-weight:700; }
.tag-dn { background:rgba(239,68,68,0.1); color:#ef4444; border:1px solid rgba(239,68,68,0.25); border-radius:20px; padding:3px 10px; font-size:0.72rem; font-weight:700; }
.tag-nt { background:rgba(234,179,8,0.1); color:#eab308; border:1px solid rgba(234,179,8,0.25); border-radius:20px; padding:3px 10px; font-size:0.72rem; font-weight:700; }
.ic-box { background:rgba(20,24,40,0.6); border:1px solid rgba(255,255,255,0.07); border-radius:9px; padding:12px 16px; text-align:center; }
.briefing-card { background:linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.8) 100%); border:1px solid rgba(148,163,184,0.1); border-radius:18px; padding:24px; position:relative; overflow:hidden; }
.briefing-card::before { content:''; position:absolute; top:0; left:0; width:4px; height:100%; background:#818cf8; }
.scenario-tag { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#818cf8; background:rgba(129,140,248,0.1); padding:4px 10px; border-radius:4px; border:1px solid rgba(129,140,248,0.2); }
</style>
""", unsafe_allow_html=True)


# ── LOADERS ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_rf_metrics():
    path = os.path.join(MODEL_DIR, "rf_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_resource
def load_models():
    # Load the primary 1d horizon model
    dir_path   = os.path.join(MODEL_DIR, "rf_dir_1d.pkl")
    range_path = os.path.join(MODEL_DIR, "rf_range.pkl")
    dir_model   = joblib.load(dir_path)   if os.path.exists(dir_path)   else None
    range_model = joblib.load(range_path) if os.path.exists(range_path) else None
    return dir_model, range_model

def load_importance(model_type="direction"):
    path = os.path.join(MODEL_DIR, f"rf_{model_type}_importance.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def load_nifty_latest():
    path = os.path.join(DATA_DIR, "nifty_daily.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if df.empty:
        return {}
    row = df.iloc[-1]
    return {
        "spot":  float(row.get("close", 22000)),
        "high":  float(row.get("high",  22200)),
        "low":   float(row.get("low",   21800)),
        "date":  str(row.get("date",    "")),
    }

def r50(val):
    """Round to nearest 50 for strike selection."""
    return round(val / 50) * 50



# ── UI HELPERS ────────────────────────────────────────────────────────────────
def render_tactical_briefing(pred):
    """Render the ELIn-style tactical summary."""
    scenario  = pred.get("scenario", "Standard Regime")
    rationale = pred.get("rationale", "No model rationale available. Run trainer.")
    strategy  = pred.get("recommended_strategy", "Neutral")
    strikes   = pred.get("recommended_strikes", "—")
    tag       = pred.get("strategy_tag", "Steady")
    
    st.markdown(f"""
    <div class="briefing-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
            <div>
                <div class="sec-label" style="margin-bottom:4px;">AI Tactical Briefing</div>
                <div style="font-size:1.4rem; font-weight:800; color:#f8fafc;">{scenario}</div>
            </div>
            <div class="scenario-tag">{tag}</div>
        </div>
        <div style="font-size:0.92rem; line-height:1.6; color:#cbd5e1; margin-bottom:20px;">
            {rationale}
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div class="glass-sm">
                <div class="sec-label" style="font-size:0.55rem;">Recommended Strategy</div>
                <div style="font-weight:700; color:#818cf8; font-size:1rem;">{strategy}</div>
            </div>
            <div class="glass-sm">
                <div class="sec-label" style="font-size:0.55rem;">Execution Strikes</div>
                <div class="mono" style="font-weight:700; color:#22c55e; font-size:0.9rem;">{strikes}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN DASHBOARD ────────────────────────────────────────────────────────────
def render_rf_dashboard():
    metrics   = load_rf_metrics()
    nifty     = load_nifty_latest()
    dir_imp   = load_importance("direction")
    range_imp = load_importance("range")

    # ── NO MODEL WARNING ──────────────────────────────────────────────────────
    if not metrics:
        st.warning("No RF model found. Running the trainer now...")
        st.info("Execute: `python rf_trainer.py` in your terminal.")
        return

    pred       = metrics.get("todays_prediction", {})
    intensity  = metrics.get("search_intensity", "Standard")
    rm         = metrics.get("range_model", {})
    h1d        = metrics.get("horizons", {}).get("1", {})
    
    # Title row with branding
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
        <div style="font-size:1.5rem; font-weight:800; color:#f0f2f8; letter-spacing:-0.02em;">
            🌲 MOSES <span style="color:#4a5270; font-weight:400; margin-left:8px;">RF ENGINE</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#4a5270; text-align:right;">
            SYSTEM STATUS: <span style="color:#22c55e;">● LIVE</span><br>
            <span style="font-size:0.6rem; color:#818cf8; opacity:0.6;">{intensity}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    direction      = pred.get("direction", "—")
    up_prob        = pred.get("up_prob", 0.5)
    down_prob      = pred.get("down_prob", 0.5)
    spot           = pred.get("spot", nifty.get("spot", 22000))
    vix            = pred.get("vix", 15)
    pcr            = pred.get("pcr", 1.0)
    inda_ret       = pred.get("inda_ret", 0)
    epi_ret        = pred.get("epi_ret", 0)
    usdinr_vel     = pred.get("usdinr_vel", 0)
    exp_range      = pred.get("expected_range_pts", 0)
    ic_upper       = pred.get("iron_condor_upper", r50(spot + exp_range * 0.6))
    ic_lower       = pred.get("iron_condor_lower", r50(spot - exp_range * 0.6))
    last_trained   = metrics.get("last_trained", "—")
    conviction     = max(up_prob, down_prob)
    dir_color      = "#22c55e" if direction == "UP" else "#ef4444"
    # ── ROW 1: HERO METRICS ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([1.4, 1.0, 1.0, 1.0, 1.0])

    with c1:
        arrow = "↑" if direction == "UP" else "↓"
        st.markdown(f"""
        <div class="glass" style="border-color:{dir_color}44; height:140px;">
            <div class="sec-label">RF Direction Signal</div>
            <div class="mono" style="font-size:2.8rem; font-weight:800; color:{dir_color}; line-height:1;">{arrow} {direction}</div>
            <div style="font-size:0.78rem; color:#4a5270; margin-top:6px;">
                UP: <span style="color:#e8ecf4; font-family:'JetBrains Mono',monospace">{up_prob:.1%}</span> &nbsp;·&nbsp;
                DOWN: <span style="color:#e8ecf4; font-family:'JetBrains Mono',monospace">{down_prob:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass" style="height:140px;">
            <div class="sec-label">Conviction</div>
            <div class="mono" style="font-size:2.2rem; font-weight:800; color:#e8ecf4;">{conviction:.0%}</div>
            <div style="font-size:0.72rem; color:#4a5270; margin-top:4px;">
                {'HIGH' if conviction >= 0.65 else 'MODERATE' if conviction >= 0.55 else 'LOW — NO TRADE'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass" style="height:140px;">
            <div class="sec-label">Expected Range</div>
            <div class="mono" style="font-size:2.2rem; font-weight:800; color:#e8ecf4;">±{exp_range:.0f}</div>
            <div style="font-size:0.72rem; color:#4a5270; margin-top:4px;">points · MAE ±{rm.get('last30d_mae_pts', 0):.0f} pts</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        acc = h1d.get('cv_accuracy', 0.5)
        acc_color = "#22c55e" if acc >= 0.54 else "#eab308" if acc >= 0.51 else "#ef4444"
        st.markdown(f"""
        <div class="glass" style="height:140px;">
            <div class="sec-label">Dir CV Accuracy (1d)</div>
            <div class="mono" style="font-size:2.2rem; font-weight:800; color:{acc_color};">{acc:.1%}</div>
            <div style="font-size:0.72rem; color:#4a5270; margin-top:4px;">Mosaic Search Edge</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        vix_color = "#ef4444" if vix > 20 else "#eab308" if vix > 16 else "#22c55e"
        st.markdown(f"""
        <div class="glass" style="height:140px;">
            <div class="sec-label">Institutional Proxy Pulse</div>
            <div class="mono" style="font-size:1.4rem; font-weight:800; color:{vix_color};">{'BULLISH' if inda_ret > 0 and epi_ret > 0 else 'BEARISH' if inda_ret < 0 and epi_ret < 0 else 'NEUTRAL'}</div>
            <div style="font-size:0.72rem; color:#4a5270; margin-top:4px;">
                INDA: {inda_ret:+.2%} &nbsp;·&nbsp; EPI: {epi_ret:+.2%}
            </div>
            <div style="font-size:0.6rem; color:#4a5270; border-top:1px solid rgba(255,255,255,0.05); padding-top:4px; margin-top:4px;">
                Rupee Vel: {usdinr_vel:+.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ROW 1.5: TECHNICAL PULSE ──────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    tp1, tp2, tp3, tp4 = st.columns(4)
    
    rsi = pred.get("rsi", 50)
    macd = pred.get("macd_hist", 0)
    ema_dist = pred.get("ema_20_dist", 0)
    
    with tp1:
        rsi_color = "#ef4444" if rsi > 70 else "#22c55e" if rsi < 30 else "#818cf8"
        st.markdown(f"""
        <div class="glass-sm">
            <div class="sec-label" style="font-size:0.55rem;">RSI (14)</div>
            <div class="mono" style="font-size:1.1rem; font-weight:700; color:{rsi_color};">{rsi:.1f}</div>
            <div style="font-size:0.6rem; color:#4a5270;">{'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with tp2:
        macd_color = "#22c55e" if macd > 0 else "#ef4444"
        st.markdown(f"""
        <div class="glass-sm">
            <div class="sec-label" style="font-size:0.55rem;">MACD Histogram</div>
            <div class="mono" style="font-size:1.1rem; font-weight:700; color:{macd_color};">{'▲' if macd > 0 else '▼'} {abs(macd):.2f}</div>
            <div style="font-size:0.6rem; color:#4a5270;">{'Bullish Momentum' if macd > 0 else 'Bearish Momentum'}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with tp3:
        ema_color = "#22c55e" if ema_dist > 0 else "#ef4444"
        st.markdown(f"""
        <div class="glass-sm">
            <div class="sec-label" style="font-size:0.55rem;">EMA (20) Distance</div>
            <div class="mono" style="font-size:1.1rem; font-weight:700; color:{ema_color};">{ema_dist:+.2f}%</div>
            <div style="font-size:0.6rem; color:#4a5270;">{'Extending' if abs(ema_dist) > 2 else 'Stable'}</div>
        </div>
        """, unsafe_allow_html=True)

    with tp4:
        st.markdown(f"""
        <div class="glass-sm">
            <div class="sec-label" style="font-size:0.55rem;">Technical Bias</div>
            <div class="mono" style="font-size:1.1rem; font-weight:700; color:#818cf8;">
                {'BULLISH' if rsi > 50 and macd > 0 else 'BEARISH' if rsi < 50 and macd < 0 else 'CONSOLIDATING'}
            </div>
            <div style="font-size:0.6rem; color:#4a5270;">Momentum Aggregator</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── ROW 2: TACTICAL BRIEFING + RANGE VIZ + MULTI-HORIZON ─────────────────
    col_brief, col_range, col_horiz = st.columns([1.3, 1.0, 1.0])

    with col_brief:
        render_tactical_briefing(pred)

    with col_range:
        st.markdown('<div class="sec-label">Range Forecast</div>', unsafe_allow_html=True)
        low_est  = spot - exp_range
        high_est = spot + exp_range

        fig = go.Figure()

        # Range band
        fig.add_shape(type="rect",
            x0=low_est, x1=high_est, y0=0, y1=1,
            fillcolor="rgba(96,165,250,0.08)",
            line=dict(color="rgba(96,165,250,0.35)", width=1),
            yref="paper"
        )
        # Spot
        fig.add_vline(x=spot, line_color="#f0f2f8", line_width=1.5, line_dash="solid")
        # Labels
        fig.add_annotation(x=low_est, y=0.95, yref="paper", text=f"{low_est:,.0f}", showarrow=False,
                           font=dict(color="#ef4444", size=11, family="JetBrains Mono"))
        fig.add_annotation(x=spot, y=0.5, yref="paper", text=f"◉ {spot:,.0f}", showarrow=False,
                           font=dict(color="#f0f2f8", size=11, family="JetBrains Mono"))
        fig.add_annotation(x=high_est, y=0.95, yref="paper", text=f"{high_est:,.0f}", showarrow=False,
                           font=dict(color="#22c55e", size=11, family="JetBrains Mono"))
        # IC wings
        fig.add_vline(x=ic_lower, line_color="#ef4444", line_width=1, line_dash="dot")
        fig.add_vline(x=ic_upper, line_color="#22c55e", line_width=1, line_dash="dot")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=180, margin=dict(l=10, r=10, t=10, b=30),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, color="#4a5270",
                       tickfont=dict(size=10, family="JetBrains Mono", color="#4a5270"),
                       range=[low_est - 200, high_est + 200]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # IC wing table
        ic_cols = st.columns(4)
        ic_data = [
            ("Buy PE", f"{r50(ic_lower - 100):,.0f}", "#60a5fa"),
            ("Sell PE", f"{ic_lower:,.0f}", "#ef4444"),
            ("Sell CE", f"{ic_upper:,.0f}", "#22c55e"),
            ("Buy CE", f"{r50(ic_upper + 100):,.0f}", "#60a5fa"),
        ]
        for col, (lbl, val, clr) in zip(ic_cols, ic_data):
            col.markdown(f"""
            <div class="ic-box">
                <div style="font-size:0.58rem; color:#4a5270; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:4px;">{lbl}</div>
                <div style="font-family:JetBrains Mono,monospace; font-size:0.85rem; font-weight:700; color:{clr};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_horiz:
        st.markdown('<div class="sec-label">Multi-Horizon Signals (Consensus)</div>', unsafe_allow_html=True)
        
        horizon_data = pred.get("horizons", {})
        rows = []
        
        # Sort horizons logically: 1, 3, 5, 7, 14
        h_keys = sorted(horizon_data.keys(), key=lambda x: int(x)) if horizon_data else []
        
        for h in h_keys:
            h_res = horizon_data[h]
            d = h_res.get("direction", "—")
            c = h_res.get("conviction", 0.5)
            
            # Simple strategy mapping for the table
            if d == "UP":
                strat = "Bull Call" if c > 0.6 else "Bull Put"
                icon  = "🟩 UP"
            else:
                strat = "Bear Put" if c > 0.6 else "Bear Call"
                icon  = "🟥 DOWN"
            
            rows.append({
                "Horizon": f"{h}d",
                "Direction": icon,
                "Conviction": f"{c:.1%}",
                "Strategy": strat
            })

        if not rows:
            st.info("No multi-horizon data. Run Hyper-Mosaic Trainer.")
        else:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                height=180,
            )

        st.markdown(f"""
        <div style="font-size:0.68rem; color:#4a5270; line-height:1.7; margin-top:8px;">
            <span style="color:#22c55e;">●</span> &gt;65% → Debit spread (high conviction) &nbsp;
            <span style="color:#60a5fa;">●</span> 55–65% → Credit spread &nbsp;
            <span style="color:#4a5270;">●</span> &lt;55% → No trade
        </div>
        """, unsafe_allow_html=True)

    # ── ROW 3: FEATURE IMPORTANCE + REGIME HEALTH ────────────────────────────
    col_dimp, col_rimp, col_health = st.columns(3)

    with col_dimp:
        st.markdown('<div class="sec-label">Direction Model — Feature Importance</div>', unsafe_allow_html=True)
        if not dir_imp.empty:
            top = dir_imp.head(8)
            fig = px.bar(
                top, x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale=[[0, "#1c2438"], [0.5, "#4a7fc1"], [1, "#818cf8"]],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=240, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                coloraxis_showscale=False,
                xaxis=dict(showgrid=False, zeroline=False, color="#4a5270",
                           tickfont=dict(size=9, family="JetBrains Mono", color="#4a5270")),
                yaxis=dict(showgrid=False, color="#4a5270", autorange="reversed",
                           tickfont=dict(size=10, family="JetBrains Mono", color="#7a82a0")),
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Run rf_trainer.py to generate importance data")

    with col_rimp:
        st.markdown('<div class="sec-label">Range Model — Feature Importance</div>', unsafe_allow_html=True)
        if not range_imp.empty:
            top = range_imp.head(8)
            fig = px.bar(
                top, x="importance", y="feature", orientation="h",
                color="importance",
                color_continuous_scale=[[0, "#1a2424"], [0.5, "#2a6b55"], [1, "#22c55e"]],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=240, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(showgrid=False, zeroline=False, color="#4a5270",
                           tickfont=dict(size=9, family="JetBrains Mono", color="#4a5270")),
                yaxis=dict(showgrid=False, color="#4a5270", autorange="reversed",
                           tickfont=dict(size=10, family="JetBrains Mono", color="#7a82a0")),
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.info("Run rf_trainer.py to generate importance data")

    with col_health:
        st.markdown('<div class="sec-label">Model Health — Regime Validation</div>', unsafe_allow_html=True)
        # Pull metrics from the 1d horizon which is primary
        h1d = metrics.get("horizons", {}).get("1", {})
        dir_reg = h1d.get("regime_scores", {})
        rng_reg = rm.get("regime_mae_pts", {})

        rows = []
        for split in dir_reg:
            acc   = dir_reg[split]
            mae   = rng_reg.get(split, 0)
            color = "🟢" if acc >= 0.56 else "🟡" if acc >= 0.52 else "🔴"
            rows.append({
                "Split": split,
                "1d Acc": f"{color} {acc:.1%}",
                "Range MAE": f"±{mae:.0f} pts",
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=200)

        st.markdown(f"""
        <div style="font-size:0.65rem; color:#4a5270; margin-top:8px; line-height:1.7;">
            Last trained: <span style="color:#7a8fbb; font-family:JetBrains Mono,monospace">{last_trained}</span>
            <br>1d Best Params: <span style="color:#818cf8">{h1d.get('best_params', '—')}</span> &nbsp;·&nbsp;
            Range Best: <span style="color:#22c55e">{rm.get('best_params', '—')}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── BOTTOM: HARD RULES ────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Hard Rules</div>', unsafe_allow_html=True)
    r1, r2, r3, r4, r5 = st.columns(5)
    rules = [
        ("🔴", "RED regime → no trade"),
        ("⚡", "VIX > 20 → credit spreads only"),
        ("🎯", "RF < 55% conviction → no directional trade"),
        ("✅", "Exit at 50% of max credit captured"),
        ("🚪", "Wing breach → exit stop-loss immediately"),
    ]
    for col, (icon, text) in zip([r1, r2, r3, r4, r5], rules):
        col.markdown(f"""
        <div class="glass-sm" style="text-align:center; font-size:0.72rem; color:#5a6280; line-height:1.6;">
            <div style="font-size:0.95rem; margin-bottom:4px;">{icon}</div>
            {text}
        </div>
        """, unsafe_allow_html=True)


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    render_rf_dashboard()
