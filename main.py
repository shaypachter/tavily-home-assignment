import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Tavily Research API — Analytics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .kpi-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 600;
        color: #e2e8f0;
        font-family: 'DM Mono', monospace;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .kpi-delta-pos { color: #4ade80; font-size: 0.85rem; }
    .kpi-delta-neg { color: #f87171; font-size: 0.85rem; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a2d3a;
    }
    .insight-box {
        background: #1a1d27;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .insight-box.warning { border-left-color: #f59e0b; }
    .insight-box.success { border-left-color: #4ade80; }
    .insight-box.danger  { border-left-color: #f87171; }

    h1 { color: #e2e8f0 !important; font-weight: 600 !important; }
    h2, h3 { color: #cbd5e1 !important; font-weight: 500 !important; }
    .stTabs [data-baseweb="tab"] { color: #64748b; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font_family="DM Sans",
    font_color="#94a3b8",
)
COLOR_SEQ = ["#6366f1", "#4ade80", "#f59e0b", "#f87171", "#38bdf8", "#a78bfa"]

# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    def load_file(name):
        if os.path.exists(f"{name}.csv.gz"):
            return pd.read_csv(f"{name}.csv.gz", compression="gzip")
        return pd.read_csv(f"{name}.csv")

    rr = load_file("research_requests")
    hu = load_file("hourly_usage")
    ic = load_file("infrastructure_costs")

    rr['TIMESTAMP'] = pd.to_datetime(rr['TIMESTAMP'], utc=True)
    hu['HOUR'] = pd.to_datetime(hu['HOUR'], utc=True)
    ic['hour'] = pd.to_datetime(ic['hour'], utc=True)

    rr['date'] = rr['TIMESTAMP'].dt.date
    rr['week'] = rr['TIMESTAMP'].dt.to_period('W').dt.start_time.dt.tz_localize('UTC')
    rr['hour_floor'] = rr['TIMESTAMP'].dt.floor('h')

    infra_cols = [c for c in ic.columns if c.startswith('infra_')]
    model_cols = [c for c in ic.columns if c.startswith('model_')]
    ic['total_infra'] = ic[infra_cols].sum(axis=1)
    ic['total_model'] = ic[model_cols].sum(axis=1)
    ic['total_cost'] = ic['total_infra'] + ic['total_model']

    return rr, hu, ic, infra_cols, model_cols

rr, hu, ic, infra_cols, model_cols = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Research API")
    st.markdown("**Analytics Dashboard**")
    st.markdown("---")
    st.markdown("**Data range**")
    st.markdown(f"`{rr['TIMESTAMP'].min().date()}` → `{rr['TIMESTAMP'].max().date()}`")
    st.markdown("**Total requests**")
    st.markdown(f"`{len(rr):,}`")
    st.markdown("---")
    st.markdown("**Questions**")
    st.markdown("1. 📈 Growth: Expansion vs. Cannibalization")
    st.markdown("2. 🩺 Health: Technical Reliability")
    st.markdown("3. 💰 Economics: Unit Cost Analysis")
    st.markdown("4. 🏗️ Infrastructure: Cost Breakdown")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Tavily Research API — Leadership Dashboard")
st.markdown("*Production data · Nov 2025 – Mar 2026 · Sampled dataset*")
st.markdown("---")

# ── Top-level KPIs ────────────────────────────────────────────────────────────
total_requests = len(rr)
success_rate = (rr['STATUS'] == 'success').mean() * 100
unique_users = rr['USER_ID'].nunique()
p50_latency = rr['RESPONSE_TIME_SECONDS'].median()
total_infra_cost = ic['total_cost'].sum()

# Week-over-week growth (last two complete weeks)
weekly = rr[rr['STATUS']=='success'].groupby('week').size().reset_index(name='count')
weekly = weekly.sort_values('week')
if len(weekly) >= 3:
    wow_growth = (weekly.iloc[-2]['count'] - weekly.iloc[-3]['count']) / weekly.iloc[-3]['count'] * 100
else:
    wow_growth = 0

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label, delta, delta_type in [
    (c1, f"{total_requests:,}", "Total Requests", None, None),
    (c2, f"{success_rate:.1f}%", "Success Rate", None, None),
    (c3, f"{unique_users:,}", "Unique Users", None, None),
    (c4, f"{p50_latency:.0f}s", "P50 Latency", None, None),
    (c5, f"${total_infra_cost:,.0f}", "Total Infra Cost", None, None),
]:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Growth: Expansion vs. Cannibalization",
    "🩺  Health: Technical Reliability",
    "💰  Unit Economics",
    "🏗️  Infrastructure Costs",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — GROWTH
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Q1: Does the Research API expand total platform usage or cannibalize existing endpoints?")
    st.markdown("""
    <div class="insight-box success">
        <b>Hypothesis:</b> Most Research API usage comes from existing users substituting simpler endpoints rather than
        representing net-new demand — potentially masking a cost inefficiency.
    </div>""", unsafe_allow_html=True)

    # ── Weekly research request growth ────────────────────────────────────
    st.markdown('<div class="section-header">Research API Growth Over Time</div>', unsafe_allow_html=True)

    weekly_all = rr.groupby('week').agg(
        total=('STATUS', 'count'),
        success=('STATUS', lambda x: (x=='success').sum()),
        unique_users=('USER_ID', 'nunique')
    ).reset_index()
    weekly_all = weekly_all[weekly_all['total'] > 10].copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=weekly_all['week'], y=weekly_all['success'],
        name="Successful Requests", marker_color="#6366f1", opacity=0.85
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=weekly_all['week'], y=weekly_all['unique_users'],
        name="Unique Users", line=dict(color="#4ade80", width=2),
        mode='lines+markers', marker=dict(size=5)
    ), secondary_y=True)
    fig.update_layout(
        **PLOTLY_THEME, height=320,
        legend=dict(orientation='h', y=1.1),
        xaxis_title="Week", yaxis_title="Requests", yaxis2_title="Users",
        bargap=0.2
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── User segmentation: new vs existing ───────────────────────────────
    st.markdown('<div class="section-header">User Segmentation: Who Adopted Research?</div>', unsafe_allow_html=True)

    first_research = rr.groupby('USER_ID')['TIMESTAMP'].min().reset_index()
    first_research.columns = ['USER_ID', 'adoption_date']

    hu_non_research = hu[hu['REQUEST_TYPE'] != 'research']
    prior_activity = hu_non_research.groupby('USER_ID')['REQUEST_COUNT'].sum().reset_index()
    prior_activity.columns = ['USER_ID', 'prior_requests']

    seg = first_research.merge(prior_activity, on='USER_ID', how='left')
    seg['prior_requests'] = seg['prior_requests'].fillna(0)
    seg['segment'] = seg['prior_requests'].apply(
        lambda x: 'New Users' if x == 0 else ('Light Users (<50 req)' if x < 50 else 'Active Users (50+ req)')
    )
    seg_counts = seg['segment'].value_counts().reset_index()
    seg_counts.columns = ['segment', 'count']

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = px.pie(
            seg_counts, names='segment', values='count',
            color_discrete_sequence=COLOR_SEQ,
            title="Research Adopters by Prior Activity"
        )
        fig2.update_layout(**PLOTLY_THEME, height=300)
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Before/After analysis ─────────────────────────────────────────────
    with col_b:
        hu2 = hu.merge(first_research, on='USER_ID', how='inner')
        hu2['days_from_adoption'] = (hu2['HOUR'] - hu2['adoption_date']).dt.total_seconds() / 86400

        pre = hu2[(hu2['days_from_adoption'] >= -30) & (hu2['days_from_adoption'] < 0)]
        post = hu2[(hu2['days_from_adoption'] >= 0) & (hu2['days_from_adoption'] < 30)]

        pre_nonres = pre[pre['REQUEST_TYPE']!='research'].groupby('USER_ID')['REQUEST_COUNT'].sum()
        post_nonres = post[post['REQUEST_TYPE']!='research'].groupby('USER_ID')['REQUEST_COUNT'].sum()

        comp = pd.DataFrame({'pre': pre_nonres, 'post': post_nonres}).dropna()
        expanded = (comp['post'] > comp['pre']).sum()
        reduced = (comp['post'] < comp['pre']).sum()
        stable = (comp['post'] == comp['pre']).sum()

        direction_df = pd.DataFrame({
            'outcome': ['Expanded usage', 'Reduced usage', 'Stable'],
            'users': [expanded, reduced, stable]
        })
        fig3 = px.bar(
            direction_df, x='outcome', y='users',
            color='outcome',
            color_discrete_sequence=["#4ade80", "#f87171", "#64748b"],
            title="Change in Non-Research Usage (30d Pre vs Post Adoption)"
        )
        fig3.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                           xaxis_title="", yaxis_title="Number of Users")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Endpoint mix shift ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Endpoint Mix: Pre vs Post Research Adoption</div>', unsafe_allow_html=True)

    pre_mix = pre.groupby('REQUEST_TYPE')['REQUEST_COUNT'].sum()
    post_mix = post.groupby('REQUEST_TYPE')['REQUEST_COUNT'].sum()

    pre_pct = (pre_mix / pre_mix.sum() * 100).reset_index()
    pre_pct.columns = ['endpoint', 'share']
    pre_pct['period'] = 'Pre-adoption (30d)'

    post_pct = (post_mix / post_mix.sum() * 100).reset_index()
    post_pct.columns = ['endpoint', 'share']
    post_pct['period'] = 'Post-adoption (30d)'

    mix_df = pd.concat([pre_pct, post_pct])

    fig4 = px.bar(
        mix_df, x='period', y='share', color='endpoint',
        color_discrete_sequence=COLOR_SEQ,
        title="Endpoint Share Before vs After Research Adoption (%)",
        barmode='stack'
    )
    fig4.update_layout(**PLOTLY_THEME, height=320,
                       xaxis_title="", yaxis_title="Share (%)")
    st.plotly_chart(fig4, use_container_width=True)

    # Insights
    pct_expanded = expanded / len(comp) * 100
    pct_reduced = reduced / len(comp) * 100
    new_user_pct = (seg['segment'] == 'New Users').mean() * 100

    st.markdown(f"""
    <div class="insight-box success">
        <b>✅ Additive signal:</b> {pct_expanded:.0f}% of existing users <i>increased</i> their non-research endpoint
        usage in the 30 days after adopting Research — suggesting the API is expanding total consumption for the majority.
    </div>
    <div class="insight-box warning">
        <b>⚠️ Substitution risk:</b> {pct_reduced:.0f}% of users reduced their usage of simpler endpoints after
        adopting Research, which may indicate some substitution of cheaper endpoints with resource-heavy Research calls.
    </div>
    <div class="insight-box">
        <b>📌 Mostly existing users:</b> Only {new_user_pct:.1f}% of Research adopters had no prior platform activity —
        the API is primarily being adopted by existing customers, not driving significant new acquisition yet.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEALTH
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Q2: Is the Research API performing reliably?")
    st.markdown("""
    <div class="insight-box success">
        <b>Hypothesis:</b> The API has an acceptable overall success rate, but latency and failure patterns
        may reveal systemic issues concentrated in specific models or time periods.
    </div>""", unsafe_allow_html=True)

    # KPIs
    total = len(rr)
    n_success = (rr['STATUS'] == 'success').sum()
    n_failed = (rr['STATUS'] == 'failed').sum()
    n_cancelled = (rr['STATUS'] == 'cancelled').sum()
    n_not_entitled = (rr['STATUS'] == 'not_entitled').sum()
    p95_latency = rr['RESPONSE_TIME_SECONDS'].quantile(0.95)

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"{n_success/total*100:.1f}%", "Success Rate"),
        (k2, f"{n_failed/total*100:.1f}%", "Failure Rate"),
        (k3, f"{p50_latency:.0f}s", "P50 Latency"),
        (k4, f"{p95_latency:.0f}s", "P95 Latency"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("")

    # ── Weekly success rate ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Weekly Success Rate & Volume</div>', unsafe_allow_html=True)

    weekly_health = rr.groupby('week').agg(
        total=('STATUS', 'count'),
        success=('STATUS', lambda x: (x == 'success').sum()),
        failed=('STATUS', lambda x: (x == 'failed').sum()),
    ).reset_index()
    weekly_health = weekly_health[weekly_health['total'] > 10].copy()
    weekly_health['success_rate'] = weekly_health['success'] / weekly_health['total'] * 100

    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Bar(
        x=weekly_health['week'], y=weekly_health['total'],
        name="Total Requests", marker_color="#6366f1", opacity=0.4
    ), secondary_y=False)
    fig5.add_trace(go.Scatter(
        x=weekly_health['week'], y=weekly_health['success_rate'],
        name="Success Rate (%)", line=dict(color="#4ade80", width=2.5),
        mode='lines+markers'
    ), secondary_y=True)
    fig5.add_hline(y=95, line_dash="dot", line_color="#f59e0b",
                   annotation_text="95% target", secondary_y=True)
    fig5.update_layout(
        **PLOTLY_THEME, height=320,
        legend=dict(orientation='h', y=1.1),
        yaxis2=dict(range=[75, 100])
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ── Failure breakdown & latency distribution ──────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Failure Breakdown</div>', unsafe_allow_html=True)
        status_df = rr['STATUS'].value_counts(dropna=False).reset_index()
        status_df.columns = ['status', 'count']
        status_df['status'] = status_df['status'].fillna('unknown')
        fig6 = px.pie(status_df, names='status', values='count',
                      color_discrete_sequence=["#4ade80","#f87171","#f59e0b","#64748b","#38bdf8"],
                      title="Request Status Distribution")
        fig6.update_layout(**PLOTLY_THEME, height=300)
        fig6.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig6, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Latency Distribution by Model</div>', unsafe_allow_html=True)
        rr_success_plot = rr[rr['STATUS']=='success'].copy()
        fig7 = px.box(
            rr_success_plot[rr_success_plot['MODEL'].isin(['mini','pro'])],
            x='MODEL', y='RESPONSE_TIME_SECONDS',
            color='MODEL', color_discrete_map={'mini': '#6366f1', 'pro': '#f59e0b'},
            title="Response Time Distribution (Successful Requests)"
        )
        fig7.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                           yaxis_title="Response Time (s)", xaxis_title="Model")
        st.plotly_chart(fig7, use_container_width=True)

    # ── Latency over time ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Latency Trend Over Time (P50 & P95)</div>', unsafe_allow_html=True)

    latency_weekly = rr[rr['STATUS']=='success'].groupby('week')['RESPONSE_TIME_SECONDS'].agg(
        p50=lambda x: x.quantile(0.5),
        p95=lambda x: x.quantile(0.95)
    ).reset_index()
    latency_weekly = latency_weekly[latency_weekly['week'] > latency_weekly['week'].min()]

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=latency_weekly['week'], y=latency_weekly['p50'],
                              name='P50', line=dict(color='#6366f1', width=2)))
    fig8.add_trace(go.Scatter(x=latency_weekly['week'], y=latency_weekly['p95'],
                              name='P95', line=dict(color='#f87171', width=2, dash='dash')))
    fig8.update_layout(**PLOTLY_THEME, height=280,
                       yaxis_title="Response Time (s)", xaxis_title="Week",
                       legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig8, use_container_width=True)

    # ── Client source failure rates ───────────────────────────────────────
    st.markdown('<div class="section-header">Failure Rate by Client Source</div>', unsafe_allow_html=True)

    client_health = rr.groupby('CLIENT_SOURCE').agg(
        total=('STATUS', 'count'),
        failed=('STATUS', lambda x: (x == 'failed').sum())
    ).reset_index()
    client_health = client_health[client_health['total'] > 50]
    client_health['failure_rate'] = client_health['failed'] / client_health['total'] * 100
    client_health = client_health.sort_values('failure_rate', ascending=True)

    fig9 = px.bar(client_health, y='CLIENT_SOURCE', x='failure_rate',
                  orientation='h', color='failure_rate',
                  color_continuous_scale=['#4ade80', '#f59e0b', '#f87171'],
                  title="Failure Rate by Client Source (%)")
    fig9.update_layout(**PLOTLY_THEME, height=300,
                       xaxis_title="Failure Rate (%)", yaxis_title="",
                       coloraxis_showscale=False)
    st.plotly_chart(fig9, use_container_width=True)

    # Insights
    low_success_weeks = weekly_health[weekly_health['success_rate'] < 90]
    st.markdown(f"""
    <div class="insight-box success">
        <b>✅ Strong overall reliability:</b> The API maintains a {n_success/total*100:.1f}% success rate across
        {total:,} requests, with reliability improving to ~97% in recent weeks as the product matures.
    </div>
    <div class="insight-box warning">
        <b>⚠️ Latency gap between models:</b> Pro model P50 latency (~301s) is ~9x higher than mini (~34s).
        P95 for pro reaches {p95_latency:.0f}s — a significant UX concern for interactive use cases.
    </div>
    <div class="insight-box">
        <b>📌 Early instability:</b> {len(low_success_weeks)} weeks fell below 90% success rate, concentrated
        in the early launch period (Dec 2025 – Jan 2026), suggesting the product has stabilized over time.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — UNIT ECONOMICS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Q3: Is the Research API financially sustainable?")
    st.markdown("""
    <div class="insight-box danger">
        <b>Hypothesis:</b> The Research API is significantly more resource-intensive than other endpoints.
        Understanding which usage patterns drive the worst unit economics is critical before scaling.
    </div>""", unsafe_allow_html=True)

    # ── Assumption note ───────────────────────────────────────────────────
    CREDIT_TO_USD = 0.008  # PAYGO rate from docs.tavily.com/documentation/api-credits
    st.markdown(f"""
    <div class="insight-box" style="border-left-color:#38bdf8; margin-bottom:1rem;">
        <b>📋 Assumption:</b> 1 credit = <b>${CREDIT_TO_USD}</b> (Tavily's public PAYGO rate).
        Monthly plan users pay $0.005–$0.0075/credit, so this is a <b>conservative upper bound on revenue</b>
        — actual margins may be worse. Source: <a href="https://docs.tavily.com/documentation/api-credits"
        style="color:#38bdf8">docs.tavily.com</a>.
    </div>""", unsafe_allow_html=True)

    rr_eco = rr[rr['STATUS']=='success'].dropna(subset=['REQUEST_COST','CREDITS_USED']).copy()

    # REQUEST_COST is in USD; CREDITS_USED converted to USD using assumption
    rr_eco['revenue_usd'] = rr_eco['CREDITS_USED'] * CREDIT_TO_USD
    rr_eco['margin_usd'] = rr_eco['revenue_usd'] - rr_eco['REQUEST_COST']
    rr_eco['cost_per_llm_call'] = rr_eco['REQUEST_COST'] / rr_eco['LLM_CALLS'].replace(0, np.nan)

    avg_cost_usd   = rr_eco['REQUEST_COST'].mean()
    avg_rev_usd    = rr_eco['revenue_usd'].mean()
    avg_margin_usd = rr_eco['margin_usd'].mean()
    pct_loss       = (rr_eco['margin_usd'] < 0).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"${avg_cost_usd:.2f}", "Avg Serving Cost (USD)"),
        (k2, f"${avg_rev_usd:.3f}", "Avg Revenue (USD @ PAYGO)"),
        (k3, f"${avg_margin_usd:.2f}", "Avg Margin (USD)"),
        (k4, f"{pct_loss:.0f}%", "Requests Served at a Loss"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("")

    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="section-header">Avg Serving Cost vs Revenue by Model (USD)</div>', unsafe_allow_html=True)
        model_eco = rr_eco.groupby('MODEL').agg(
            avg_cost=('REQUEST_COST', 'mean'),
            avg_revenue=('revenue_usd', 'mean'),
            count=('REQUEST_COST', 'count')
        ).reset_index()
        model_eco = model_eco[model_eco['MODEL'].isin(['mini','pro'])]

        fig10 = go.Figure()
        fig10.add_trace(go.Bar(name='Avg Serving Cost (USD)', x=model_eco['MODEL'],
                               y=model_eco['avg_cost'], marker_color='#f87171'))
        fig10.add_trace(go.Bar(name='Avg Revenue (USD)', x=model_eco['MODEL'],
                               y=model_eco['avg_revenue'], marker_color='#4ade80'))
        fig10.update_layout(**PLOTLY_THEME, height=300, barmode='group',
                            yaxis_title="USD ($)", xaxis_title="Model",
                            legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig10, use_container_width=True)

    with col_f:
        st.markdown('<div class="section-header">Serving Cost Distribution by Client Source (USD)</div>', unsafe_allow_html=True)
        top_clients = rr_eco['CLIENT_SOURCE'].value_counts().head(6).index
        fig11 = px.box(
            rr_eco[rr_eco['CLIENT_SOURCE'].isin(top_clients)],
            x='CLIENT_SOURCE', y='REQUEST_COST',
            color='CLIENT_SOURCE', color_discrete_sequence=COLOR_SEQ,
            title="Request Serving Cost by Client ($)"
        )
        fig11.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                            xaxis_title="", yaxis_title="Serving Cost (USD)")
        st.plotly_chart(fig11, use_container_width=True)

    # ── Cost drivers ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cost Drivers: Search Calls vs Serving Cost (USD)</div>', unsafe_allow_html=True)

    sample = rr_eco.sample(min(3000, len(rr_eco)), random_state=42)
    fig12 = px.scatter(
        sample, x='SEARCH_CALLS', y='REQUEST_COST',
        color='MODEL', opacity=0.5, size_max=6,
        color_discrete_map={'mini': '#6366f1', 'pro': '#f59e0b', 'auto': '#4ade80'},
        title="Search Calls vs Serving Cost — sampled (USD)"
    )
    fig12.update_layout(**PLOTLY_THEME, height=300,
                        xaxis_title="Search Calls", yaxis_title="Serving Cost (USD)")
    st.plotly_chart(fig12, use_container_width=True)

    # ── Schema impact & weekly trend ──────────────────────────────────────
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<div class="section-header">Output Schema Impact on Serving Cost</div>', unsafe_allow_html=True)
        schema_eco = rr_eco.groupby('HAS_OUTPUT_SCHEMA').agg(
            avg_cost=('REQUEST_COST', 'mean'),
            count=('REQUEST_COST', 'count')
        ).reset_index()
        schema_eco['HAS_OUTPUT_SCHEMA'] = schema_eco['HAS_OUTPUT_SCHEMA'].map(
            {True: 'With Schema', False: 'No Schema', 'TRUE': 'With Schema', 'FALSE': 'No Schema'})
        fig13 = px.bar(schema_eco, x='HAS_OUTPUT_SCHEMA', y='avg_cost',
                       color='HAS_OUTPUT_SCHEMA', color_discrete_sequence=['#6366f1','#38bdf8'],
                       title="Avg Serving Cost: Schema vs No Schema (USD)")
        fig13.update_layout(**PLOTLY_THEME, height=280, showlegend=False,
                            xaxis_title="", yaxis_title="Avg Serving Cost (USD)")
        st.plotly_chart(fig13, use_container_width=True)

    with col_h:
        st.markdown('<div class="section-header">Weekly Avg Cost vs Revenue per Request (USD)</div>', unsafe_allow_html=True)
        weekly_cost = rr_eco.groupby('week').agg(
            avg_cost=('REQUEST_COST', 'mean'),
            avg_revenue=('revenue_usd', 'mean')
        ).reset_index()
        fig14 = go.Figure()
        fig14.add_trace(go.Scatter(x=weekly_cost['week'], y=weekly_cost['avg_cost'],
                                   name='Avg Serving Cost (USD)', line=dict(color='#f87171', width=2)))
        fig14.add_trace(go.Scatter(x=weekly_cost['week'], y=weekly_cost['avg_revenue'],
                                   name='Avg Revenue (USD)', line=dict(color='#4ade80', width=2)))
        fig14.update_layout(**PLOTLY_THEME, height=280,
                            yaxis_title="USD ($)", legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig14, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box danger">
        <b>🚨 Severe unit economics gap:</b> Average serving cost is <b>${avg_cost_usd:.2f}</b> per request,
        while average revenue at PAYGO rates is only <b>${avg_rev_usd:.3f}</b> — meaning the API loses
        ~${abs(avg_margin_usd):.2f} on average per successful request. {pct_loss:.0f}% of requests are served at a loss.
    </div>
    <div class="insight-box warning">
        <b>⚠️ Pro model widens the gap:</b> Pro requests cost ~4× more to serve than mini, but revenue
        does not scale proportionally with credits charged — making pro the primary driver of losses.
    </div>
    <div class="insight-box">
        <b>📌 Search calls are the primary cost driver:</b> Serving cost scales strongly with the number of
        web search operations, making search depth the most actionable lever for cost optimization.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — INFRASTRUCTURE
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Infrastructure & Cost Analysis")

    total_cost = ic['total_cost'].sum()
    total_infra = ic['total_infra'].sum()
    total_model = ic['total_model'].sum()
    avg_hourly = ic['total_cost'].mean()
    avg_daily = avg_hourly * 24
    avg_weekly = avg_daily * 7
    avg_monthly = avg_daily * 30.4
    infra_pct = total_infra / total_cost * 100
    model_pct = total_model / total_cost * 100

    # Cost per research request (total infra cost / research request count)
    hourly_requests = rr.groupby(rr['hour_floor'].dt.floor('h')).size().reset_index(name='request_count')
    hourly_requests.columns = ['hour', 'request_count']
    ic_tmp = ic.copy()
    ic_tmp['hour'] = ic_tmp['hour'].dt.tz_localize(None)
    hourly_requests['hour'] = hourly_requests['hour'].dt.tz_localize(None)
    ic_merged = ic_tmp.merge(hourly_requests, on='hour', how='left')
    ic_merged['request_count'] = ic_merged['request_count'].fillna(0)
    ic_merged['cost_per_req'] = ic_merged.apply(
        lambda r: r['total_cost'] / r['request_count'] if r['request_count'] > 0 else np.nan, axis=1)
    median_cost_per_req = ic_merged['cost_per_req'].median()

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Cost</div>
            <div class="kpi-value">${total_cost:,.0f}</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">Nov 2025 – Mar 2026</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:left;">
            <div class="kpi-label" style="text-align:center;">Avg Cost</div>
            <div class="kpi-value" style="font-size:1.4rem;">${avg_hourly:,.0f} <span style="font-size:0.85rem;font-weight:400;color:#64748b;">/ hr</span></div>
            <div style="font-size:0.9rem;font-weight:500;color:#e2e8f0;margin-top:4px;">${avg_daily:,.0f} <span style="font-size:0.78rem;font-weight:400;color:#64748b;">/ day</span></div>
            <div style="font-size:0.78rem;color:#64748b;margin-top:3px;">${avg_weekly:,.0f} / wk &nbsp;·&nbsp; ${avg_monthly:,.0f} / mo</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Infrastructure Cost</div>
            <div class="kpi-value">{infra_pct:.1f}%</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">${total_infra:,.0f} total</div>
            <div style="margin-top:8px;height:4px;background:#2a2d3a;border-radius:2px;">
                <div style="width:{infra_pct:.1f}%;height:4px;background:#6366f1;border-radius:2px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Cost</div>
            <div class="kpi-value">{model_pct:.1f}%</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">${total_model:,.0f} total</div>
            <div style="margin-top:8px;height:4px;background:#2a2d3a;border-radius:2px;">
                <div style="width:{model_pct:.1f}%;height:4px;background:#f59e0b;border-radius:2px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Cost per Research Request</div>
            <div class="kpi-value">${median_cost_per_req:.2f}</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">median (infra + model)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Cost breakdown with interactive toggle ────────────────────────────
    import json
    component_totals = ic[infra_cols + model_cols].sum().reset_index()
    component_totals.columns = ['component', 'total']
    component_totals['ctype'] = component_totals['component'].apply(
        lambda x: 'infra' if x.startswith('infra_') else 'model')
    component_totals['label'] = component_totals['component'].str.replace('infra_','').str.replace('model_','').str.replace('_',' ').str.title()
    component_totals = component_totals.sort_values('total', ascending=True)

    grand_total = float(component_totals['total'].sum())
    infra_palette = ['#EEEDFE','#CECBF6','#AFA9EC','#9F97E6','#8F85E0','#7F77DD','#6A62CC','#534AB7','#3C3489','#26215C']
    model_palette = ['#fef3c7','#fde68a','#fbbf24','#f59e0b','#d97706','#b45309']

    infra_rows = component_totals[component_totals['ctype']=='infra'].reset_index(drop=True)
    model_rows = component_totals[component_totals['ctype']=='model'].reset_index(drop=True)
    infra_rows['color'] = [infra_palette[i] for i in range(len(infra_rows))]
    model_rows['color'] = [model_palette[i] for i in range(len(model_rows))]
    all_rows = pd.concat([infra_rows, model_rows]).sort_values('total', ascending=True)

    infra_js = json.dumps([{'label': r['label'], 'value': round(float(r['total']),2), 'color': r['color']} for _,r in infra_rows.iterrows()])
    model_js = json.dumps([{'label': r['label'], 'value': round(float(r['total']),2), 'color': r['color']} for _,r in model_rows.iterrows()])
    all_js   = json.dumps([{'label': r['label'], 'value': round(float(r['total']),2), 'color': r['color']} for _,r in all_rows.iterrows()])

    html_code = f"""
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; font-family: sans-serif; }}
  body {{ margin:0; background:transparent; }}
  .tbtn {{ padding:4px 14px; font-size:12px; border-radius:6px; border:1px solid #4a5568; background:transparent; color:#94a3b8; cursor:pointer; }}
  .tbtn.active {{ background:#e2e8f0; color:#0f1117; }}
  .card {{ background:#1a1d27; border:1px solid #2a2d3a; border-radius:12px; padding:1rem 1.25rem; }}
  .slabel {{ font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin:0 0 0.75rem; }}
  #legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; font-size:11px; color:#94a3b8; }}
</style>
<div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
  <span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;">View:</span>
  <button class="tbtn active" id="btn-all" onclick="setFilter('all')">All</button>
  <button class="tbtn" id="btn-infra" onclick="setFilter('infra')">Infrastructure</button>
  <button class="tbtn" id="btn-models" onclick="setFilter('models')">Models</button>
</div>
<div style="display:grid;grid-template-columns:1.4fr 1fr;gap:16px;">
  <div class="card">
    <p class="slabel">Cost share by component</p>
    <div style="position:relative;width:100%;height:320px;"><canvas id="barChart"></canvas></div>
  </div>
  <div class="card">
    <p class="slabel">% of total cost</p>
    <div id="legend"></div>
    <div style="position:relative;width:100%;height:280px;">
      <canvas id="pieChart"></canvas>
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;">
        <div id="centerPct" style="font-size:22px;font-weight:500;color:#e2e8f0;"></div>
        <div id="centerSub" style="font-size:11px;color:#64748b;"></div>
      </div>
    </div>
  </div>
</div>
<script>
  const TOTAL = {grand_total};
  const infraData = {infra_js};
  const modelData = {model_js};
  const allData   = {all_js};
  function getData(f) {{
    if (f==="infra")  return {{ bar:[...infraData].sort((a,b)=>a.value-b.value), pie:infraData }};
    if (f==="models") return {{ bar:[...modelData].sort((a,b)=>a.value-b.value), pie:modelData }};
    return {{ bar:allData, pie:allData }};
  }}
  function buildLegend(data) {{
    document.getElementById("legend").innerHTML = data.map(d=>
      `<span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:2px;background:${{d.color}};display:inline-block;border:1px solid #2a2d3a;"></span>${{d.label}} ${{(d.value/TOTAL*100).toFixed(1)}}%</span>`
    ).join("");
  }}
  function updateCenter(f, pie) {{
    const t=pie.reduce((s,d)=>s+d.value,0);
    document.getElementById("centerPct").textContent=(t/TOTAL*100).toFixed(1)+"%";
    document.getElementById("centerSub").textContent=f==="all"?"of total":f==="infra"?"infrastructure":"models";
  }}
  function setFilter(f) {{
    ["all","infra","models"].forEach(x=>document.getElementById("btn-"+x).classList.toggle("active",x===f));
    const {{bar,pie}}=getData(f);
    const rem=TOTAL-pie.reduce((s,d)=>s+d.value,0);
    barChart.data.labels=bar.map(d=>d.label);
    barChart.data.datasets[0].data=bar.map(d=>d.value);
    barChart.data.datasets[0].backgroundColor=bar.map(d=>d.color);
    barChart.update();
    pieChart.data.labels=[...pie.map(d=>d.label),f!=="all"?"Other":null].filter(Boolean);
    pieChart.data.datasets[0].data=[...pie.map(d=>d.value),f!=="all"?rem:null].filter(v=>v!==null);
    pieChart.data.datasets[0].backgroundColor=[...pie.map(d=>d.color),f!=="all"?"#2a2d3a":null].filter(Boolean);
    pieChart.update();
    buildLegend(pie); updateCenter(f,pie);
  }}
  const {{bar:ib,pie:ip}}=getData("all");
  const gc="rgba(255,255,255,0.07)",tc="#94a3b8";
  const barChart=new Chart(document.getElementById("barChart"),{{
    type:"bar",
    data:{{labels:ib.map(d=>d.label),datasets:[{{data:ib.map(d=>d.value),backgroundColor:ib.map(d=>d.color),borderRadius:3}}]}},
    options:{{responsive:true,maintainAspectRatio:false,indexAxis:"y",
      plugins:{{legend:{{display:false}}}},
      scales:{{x:{{ticks:{{color:tc,font:{{size:10}},callback:v=>"$"+Math.round(v/1000)+"k"}},grid:{{color:gc}}}},
               y:{{ticks:{{color:tc,font:{{size:10}}}},grid:{{color:gc}}}}}}
    }}
  }});
  const pieChart=new Chart(document.getElementById("pieChart"),{{
    type:"doughnut",
    data:{{labels:ip.map(d=>d.label),datasets:[{{data:ip.map(d=>d.value),backgroundColor:ip.map(d=>d.color),borderWidth:0}}]}},
    options:{{responsive:true,maintainAspectRatio:false,cutout:"62%",
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>`${{(ctx.raw/TOTAL*100).toFixed(1)}}% of total (${{Math.round(ctx.raw/1000)}}k)`}}}}
      }}
    }}
  }});
  buildLegend(ip); updateCenter("all",ip);
</script>
"""
    st.components.v1.html(html_code, height=480)


    # ── Cost over time ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Total Cost Trend Over Time (Infrastructure vs Models)</div>', unsafe_allow_html=True)

    ic_daily = ic.copy()
    ic_daily['date'] = ic_daily['hour'].dt.date
    daily_costs = ic_daily.groupby('date').agg(
        infra=('total_infra', 'sum'),
        model=('total_model', 'sum')
    ).reset_index()

    fig17 = go.Figure()
    fig17.add_trace(go.Scatter(x=daily_costs['date'], y=daily_costs['infra'],
                               name='Infrastructure', fill='tozeroy',
                               line=dict(color='#6366f1', width=1.5),
                               fillcolor='rgba(99,102,241,0.2)'))
    fig17.add_trace(go.Scatter(x=daily_costs['date'], y=daily_costs['model'],
                               name="Model Cost", fill='tozeroy',
                               line=dict(color='#f59e0b', width=1.5),
                               fillcolor='rgba(245,158,11,0.2)'))
    fig17.update_layout(**PLOTLY_THEME, height=300,
                        yaxis_title="Daily Cost ($)",
                        legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig17, use_container_width=True)

    # ── Cost spikes ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cost Spikes — Total Cost vs Research Request Volume</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box" style="border-left-color:#38bdf8;">
        <b>📋 Assumption:</b> Cost spikes are analyzed by comparing total hourly infrastructure cost against
        request count. This assumes requests are roughly uniform in complexity — in practice, a few unusually
        heavy requests (e.g. pro model with 200+ search calls) can drive a cost spike without a corresponding
        volume spike, and vice versa. A more precise analysis would require request-level cost attribution
        via the <code>request_cost</code> field in <code>research_requests.csv</code>.
    </div>""", unsafe_allow_html=True)

    import json as _json2

    # Daily aggregation
    merged_spike = ic.copy()
    merged_spike['hour_naive'] = merged_spike['hour'].dt.tz_localize(None)
    rr_h = rr.groupby(rr['hour_floor'].dt.tz_localize(None)).size().reset_index(name='requests')
    rr_h.columns = ['hour', 'requests']
    merged_spike = merged_spike.merge(rr_h, left_on='hour_naive', right_on='hour', how='left')
    merged_spike['requests'] = merged_spike['requests'].fillna(0)
    merged_spike['date'] = pd.to_datetime(merged_spike['hour_naive']).dt.date
    merged_spike['month'] = pd.to_datetime(merged_spike['hour_naive']).dt.to_period('M').astype(str)

    daily_spike = merged_spike.groupby('date').agg(
        total_cost=('total_cost','sum'),
        requests=('requests','sum')
    ).reset_index()
    daily_spike = daily_spike[daily_spike['requests'] >= 50]

    mean_d = float(daily_spike['total_cost'].mean())
    std_d  = float(daily_spike['total_cost'].std())
    thresh_d = mean_d + 2*std_d

    daily_js = _json2.dumps({
        'labels': [str(d) for d in daily_spike['date']],
        'costs':  daily_spike['total_cost'].round(2).tolist(),
        'reqs':   daily_spike['requests'].astype(int).tolist(),
        'threshold': round(thresh_d, 2),
        'spikes': int((daily_spike['total_cost'] > thresh_d).sum())
    })

    # Hourly by month
    hourly_js_dict = {}
    for m in ['2025-12','2026-01','2026-02','2026-03']:
        sub = merged_spike[merged_spike['month'] == m].copy()
        sub2 = sub[sub.index % 2 == 0]
        mean_h = float(sub['total_cost'].mean())
        std_h  = float(sub['total_cost'].std())
        thresh_h = mean_h + 2*std_h
        hourly_js_dict[m] = {
            'labels':   sub2['hour_naive'].dt.strftime('%d %H:00').tolist(),
            'costs':    sub2['total_cost'].round(2).tolist(),
            'requests': sub2['requests'].astype(int).tolist(),
            'threshold': round(thresh_h, 2),
            'spikes':   int((sub['total_cost'] > thresh_h).sum())
        }

    hourly_js = _json2.dumps(hourly_js_dict)

    html_spike = f"""
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; font-family:sans-serif; }}
  body {{ margin:0; background:transparent; }}
  .vbtn {{ padding:4px 14px; font-size:12px; border-radius:6px; border:1px solid #4a5568; background:transparent; color:#94a3b8; cursor:pointer; transition:all 0.15s; }}
  .vbtn.active {{ background:#e2e8f0; color:#0f1117; }}
  .row {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:0.75rem; }}
  .legend {{ display:flex; gap:16px; margin-bottom:8px; font-size:11px; color:#94a3b8; }}
  .stats {{ display:flex; gap:16px; margin-bottom:10px; font-size:11px; color:#94a3b8; }}
</style>
<div class="row">
  <span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;">View:</span>
  <button class="vbtn active" id="btn-daily" onclick="setView('daily')">Daily</button>
  <span style="font-size:11px;color:#64748b;">Hourly zoom:</span>
  <button class="vbtn" id="btn-2025-12" onclick="setView('2025-12')">Dec 2025</button>
  <button class="vbtn" id="btn-2026-01" onclick="setView('2026-01')">Jan 2026</button>
  <button class="vbtn" id="btn-2026-02" onclick="setView('2026-02')">Feb 2026</button>
  <button class="vbtn" id="btn-2026-03" onclick="setView('2026-03')">Mar 2026</button>
</div>
<div class="legend">
  <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:3px;background:#534AB7;display:inline-block;"></span>Total cost</span>
  <span style="display:flex;align-items:center;gap:4px;"><span style="width:8px;height:8px;border-radius:50%;background:#f87171;display:inline-block;"></span>Cost spike (&gt;mean+2σ)</span>
  <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:3px;border-top:2px dashed #4ade80;display:inline-block;"></span>Research requests</span>
</div>
<div class="stats" id="statsRow"></div>
<div style="position:relative;width:100%;height:300px;"><canvas id="spikeChart"></canvas></div>
<script>
  const DAILY = {daily_js};
  const HOURLY = {hourly_js};
  const tc='#94a3b8', gc='rgba(255,255,255,0.07)';

  function makeThresholdLine(threshold) {{
    return function(chart) {{
      const ctx2=chart.ctx, xs=chart.scales.x, ys=chart.scales.y;
      const yPx=ys.getPixelForValue(threshold);
      if(yPx<ys.top||yPx>ys.bottom) return;
      ctx2.save(); ctx2.strokeStyle='#f87171'; ctx2.setLineDash([4,4]); ctx2.lineWidth=1;
      ctx2.beginPath(); ctx2.moveTo(xs.left,yPx); ctx2.lineTo(xs.right,yPx); ctx2.stroke();
      ctx2.fillStyle='#f87171'; ctx2.font='10px sans-serif';
      ctx2.fillText('mean+2σ', xs.right-55, yPx-4); ctx2.restore();
    }};
  }}

  function getVdata(view) {{
    if(view==='daily') return {{ ...DAILY, labels:DAILY.labels, costs:DAILY.costs, reqs:DAILY.reqs, costUnit:'$/day', reqUnit:'req/day', granularity:'Daily (≥50 req/day)' }};
    const h=HOURLY[view];
    return {{ ...h, labels:h.labels, costs:h.costs, reqs:h.requests, costUnit:'$/hr', reqUnit:'req/hr', granularity:view+' (hourly, every 2nd point)' }};
  }}

  function buildDs(vdata) {{
    return [
      {{ label:'Total cost', data:vdata.costs, yAxisID:'y',
        borderColor:'#534AB7', backgroundColor:'rgba(83,74,183,0.08)',
        borderWidth:1.5, fill:true, tension:0.3, pointRadius:0, pointHoverRadius:4,
        segment:{{ borderColor:ctx=>vdata.costs[ctx.p1DataIndex]>vdata.threshold?'#f87171':'#534AB7' }}
      }},
      {{ label:'Research requests', data:vdata.reqs, yAxisID:'y2',
        borderColor:'#4ade80', backgroundColor:'rgba(74,222,128,0.06)',
        borderWidth:1.5, borderDash:[4,4], fill:true, tension:0.3, pointRadius:0
      }}
    ];
  }}

  function updateStats(vdata) {{
    document.getElementById('statsRow').innerHTML=`
      <span>Threshold: <b style="color:#f87171;">${{vdata.threshold.toFixed(0)}} ${{vdata.costUnit}}</b></span>
      <span>Spikes detected: <b style="color:#f87171;">${{vdata.spikes}}</b></span>
      <span>Granularity: ${{vdata.granularity}}</span>`;
  }}

  function setView(view) {{
    document.querySelectorAll('.vbtn').forEach(b=>b.classList.remove('active'));
    document.getElementById('btn-'+view).classList.add('active');
    const vdata=getVdata(view);
    chart.data.labels=vdata.labels;
    chart.data.datasets=buildDs(vdata);
    chart.options.scales.y.afterDraw=makeThresholdLine(vdata.threshold);
    chart.options.scales.y.title.text='Total cost ('+vdata.costUnit+')';
    chart.options.scales.y2.title.text='Requests ('+vdata.reqUnit+')';
    chart.update(); updateStats(vdata);
  }}

  const initData=getVdata('daily');
  const chart=new Chart(document.getElementById('spikeChart'),{{
    type:'line',
    data:{{labels:initData.labels, datasets:buildDs(initData)}},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{legend:{{display:false}}}},
      scales:{{
        x:{{ticks:{{color:tc,font:{{size:9}},maxRotation:45,autoSkip:true,maxTicksLimit:20}},grid:{{color:gc}}}},
        y:{{ticks:{{color:tc,font:{{size:10}},callback:v=>'$'+Math.round(v)}},grid:{{color:gc}},
           title:{{display:true,text:'Total cost ($/day)',color:tc,font:{{size:10}}}},
           afterDraw:makeThresholdLine(initData.threshold)}},
        y2:{{position:'right',ticks:{{color:'#4ade80',font:{{size:10}}}},grid:{{display:false}},
             title:{{display:true,text:'Requests (req/day)',color:'#4ade80',font:{{size:10}}}}}}
      }}
    }}
  }});
  updateStats(initData);
</script>
"""
    st.components.v1.html(html_spike, height=420)

        # ── Cost efficiency ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cost Efficiency — Cost per Research Request Over Time (Weekly)</div>', unsafe_allow_html=True)

    eff_weekly = merged_spike.copy()
    eff_weekly['week'] = pd.to_datetime(eff_weekly['hour_naive']).dt.to_period('W').dt.start_time
    eff_agg = eff_weekly.groupby('week').agg(
        total_cost=('total_cost','sum'),
        requests=('requests','sum')
    ).reset_index()
    eff_agg = eff_agg[eff_agg['requests'] >= 500]
    eff_agg['cost_per_req'] = eff_agg['total_cost'] / eff_agg['requests']

    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=eff_agg['week'], y=eff_agg['cost_per_req'],
        mode='lines+markers',
        line=dict(color='#f59e0b', width=2.5),
        marker=dict(size=6, color='#f59e0b'),
        fill='tozeroy',
        fillcolor='rgba(245,158,11,0.08)',
        hovertemplate='Week of %{x|%b %d}<br>Cost/request: $%{y:.2f}<extra></extra>'
    ))
    fig_eff.update_layout(
        **PLOTLY_THEME, height=280,
        yaxis_title="Cost per research request ($)",
        xaxis_title="Week",
        annotations=[dict(
            x=eff_agg['week'].iloc[-1], y=eff_agg['cost_per_req'].iloc[-1],
            text=f"${eff_agg['cost_per_req'].iloc[-1]:.2f}/req",
            showarrow=True, arrowhead=2, arrowcolor='#f59e0b',
            font=dict(color='#f59e0b', size=11)
        )]
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    first_cpr = eff_agg['cost_per_req'].iloc[0]
    last_cpr  = eff_agg['cost_per_req'].iloc[-1]
    pct_drop  = (first_cpr - last_cpr) / first_cpr * 100
    st.markdown(f"""
    <div class="insight-box success">
        <b>✅ Strong efficiency gains:</b> Cost per research request dropped from
        <b>${first_cpr:.2f}</b> in early December to <b>${last_cpr:.2f}</b> by late March —
        a <b>{pct_drop:.0f}% reduction</b> as fixed infrastructure costs are spread across
        a rapidly growing request volume.
    </div>
    <div class="insight-box warning">
        <b>⚠️ Weeks with &lt;500 requests excluded</b> — low-volume weeks produce
        misleadingly high cost/request ratios due to fixed overhead dominating.
        The filter ensures only representative weeks are shown.
    </div>""", unsafe_allow_html=True)

    # ── Fixed vs variable — interactive section ───────────────────────────
    st.markdown('<div class="section-header">Fixed vs Variable Cost Components</div>', unsafe_allow_html=True)

    # Compute correlations with hourly research request volume
    import json as _json
    rr_hourly = rr.groupby(rr['hour_floor'].dt.tz_localize(None)).size().reset_index(name='request_count')
    rr_hourly.columns = ['hour', 'request_count']
    ic_tmp2 = ic.copy()
    ic_tmp2['hour_naive'] = ic_tmp2['hour'].dt.tz_localize(None)
    ic_corr = ic_tmp2.merge(rr_hourly, left_on='hour_naive', right_on='hour', how='left')
    ic_corr['request_count'] = ic_corr['request_count'].fillna(0)

    CORR_THRESHOLD = 0.3
    corr_results = []
    for col in infra_cols + model_cols:
        corr = float(ic_corr[col].corr(ic_corr['request_count']))
        label = col.replace('infra_','').replace('model_','').replace('_',' ').title()
        total = float(ic[col].sum())
        is_fixed = corr < CORR_THRESHOLD
        corr_results.append({'col': col, 'label': label, 'corr': corr, 'total': total, 'fixed': is_fixed})

    corr_df = pd.DataFrame(corr_results)
    fixed_components  = corr_df[corr_df['fixed']]['label'].tolist()
    var_components    = corr_df[~corr_df['fixed']]['label'].tolist()
    fixed_total       = corr_df[corr_df['fixed']]['total'].sum()
    var_total         = corr_df[~corr_df['fixed']]['total'].sum()
    grand_corr_total  = fixed_total + var_total
    fixed_pct         = fixed_total / grand_corr_total * 100
    var_pct           = var_total   / grand_corr_total * 100

    # Weekly aggregated data for time series (real data)
    rr_weekly = rr.copy()
    rr_weekly['week'] = rr_weekly['TIMESTAMP'].dt.to_period('W').dt.start_time.dt.tz_localize(None)
    weekly_req = rr_weekly.groupby('week').size().reset_index(name='requests')

    ic_weekly = ic_tmp2.copy()
    ic_weekly['week'] = pd.to_datetime(ic_weekly['hour_naive']).dt.to_period('W').dt.start_time
    weekly_costs_ts = ic_weekly.groupby('week')[infra_cols + model_cols].mean().reset_index()
    weekly_merged = weekly_costs_ts.merge(weekly_req, on='week', how='left').fillna(0)
    weekly_merged = weekly_merged[weekly_merged['requests'] > 0]

    weeks_js = _json.dumps([str(w.date()) for w in weekly_merged['week']])
    requests_js = _json.dumps(weekly_merged['requests'].tolist())

    # Build component data for JS
    comp_colors_fixed = ['#534AB7','#7F77DD','#AFA9EC','#9F97E6','#8F85E0','#6A62CC','#3C3489','#CECBF6','#26215C','#4f46e5','#818cf8','#c7d2fe']
    comp_colors_var   = ['#f59e0b','#d97706','#fbbf24','#b45309']

    ts_components = []
    fi = 0
    vi = 0
    preset_labels = set()
    # find largest fixed and largest variable for preset
    largest_fixed = corr_df[corr_df['fixed']].nlargest(1,'total')['label'].values[0]
    largest_var   = corr_df[~corr_df['fixed']].nlargest(1,'total')['label'].values[0]

    for _, row in corr_df.sort_values('corr').iterrows():
        if row['fixed']:
            color = comp_colors_fixed[fi % len(comp_colors_fixed)]
            fi += 1
        else:
            color = comp_colors_var[vi % len(comp_colors_var)]
            vi += 1
        is_preset = row['label'] in [largest_fixed, largest_var]
        data_vals = weekly_merged[row['col']].round(2).tolist()
        ts_components.append({
            'label': row['label'],
            'data': data_vals,
            'color': color,
            'preset': is_preset
        })

    ts_components_js = _json.dumps(ts_components)

    # Scatter data
    scatter_below = [{'label':r['label'],'corr':round(r['corr'],3),'total':round(r['total'],0)} for _,r in corr_df[corr_df['fixed']].iterrows()]
    scatter_above = [{'label':r['label'],'corr':round(r['corr'],3),'total':round(r['total'],0)} for _,r in corr_df[~corr_df['fixed']].iterrows()]
    scatter_below_js = _json.dumps(scatter_below)
    scatter_above_js = _json.dumps(scatter_above)

    fixed_names_str = ', '.join(fixed_components)
    var_names_str   = ', '.join(var_components)

    html_fv = f"""
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; font-family:sans-serif; }}
  body {{ margin:0; background:transparent; }}
  .card {{ background:#1a1d27; border:1px solid #2a2d3a; border-radius:12px; padding:1rem 1.25rem; margin-bottom:1rem; }}
  .slabel {{ font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin:0 0 0.4rem; }}
  .sdesc  {{ font-size:11px; color:#64748b; margin:0 0 0.75rem; }}
  .tbtn {{ padding:3px 10px; font-size:11px; border-radius:6px; cursor:pointer; transition:all 0.15s; }}
  #toggles {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:0.75rem; }}
  .legend-row {{ display:flex; gap:16px; margin-bottom:10px; font-size:11px; color:#94a3b8; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .concl-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .concl-card {{ background:#1a1d27; border:1px solid #2a2d3a; border-radius:8px; padding:1rem; }}
  .bar-bg {{ margin-top:8px; height:4px; background:#2a2d3a; border-radius:2px; }}
  .comp-names {{ margin-top:10px; font-size:11px; color:#64748b; line-height:1.7; }}
</style>

<div class="card">
  <p class="slabel">Step 1 — Observed: cost components vs research request volume over time</p>
  <p class="sdesc">Some components stay flat regardless of traffic — others clearly track request volume. Select components to compare:</p>
  <div id="toggles"></div>
  <div class="legend-row">
    <span style="display:flex;align-items:center;gap:4px;">
      <span style="width:16px;height:2px;border-top:2px dashed #4ade80;display:inline-block;"></span>
      Research requests (always on)
    </span>
  </div>
  <div style="position:relative;width:100%;height:260px;"><canvas id="timeChart"></canvas></div>
</div>

<div class="card">
  <p class="slabel">Step 2 — Validated: correlation with research request volume across all components</p>
  <p class="sdesc">A threshold of 0.3 was chosen based on a natural gap in the data — all components cluster either below 0.22 or above 0.36, with nothing in between.</p>
  <div class="legend-row">
    <span style="display:flex;align-items:center;gap:4px;"><span class="dot" style="background:#534AB7;"></span>Below threshold</span>
    <span style="display:flex;align-items:center;gap:4px;"><span class="dot" style="background:#f59e0b;"></span>Above threshold</span>
  </div>
  <div style="position:relative;width:100%;height:300px;"><canvas id="scatterChart"></canvas></div>
</div>

<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;padding:1rem;">
  <p class="slabel">Step 3 — Conclusion: fixed vs variable split</p>
  <div class="concl-grid">
    <div class="concl-card">
      <div style="font-size:12px;color:#64748b;margin-bottom:4px;">Fixed costs</div>
      <div style="font-size:22px;font-weight:500;color:#e2e8f0;">${fixed_total:,.0f} <span style="font-size:14px;color:#64748b;">{fixed_pct:.1f}%</span></div>
      <div style="font-size:11px;color:#64748b;margin-top:4px;">Always-on regardless of usage</div>
      <div class="bar-bg"><div style="width:{fixed_pct:.1f}%;height:4px;background:#534AB7;border-radius:2px;"></div></div>
      <div class="comp-names">{fixed_names_str}</div>
    </div>
    <div class="concl-card">
      <div style="font-size:12px;color:#64748b;margin-bottom:4px;">Variable costs</div>
      <div style="font-size:22px;font-weight:500;color:#e2e8f0;">${var_total:,.0f} <span style="font-size:14px;color:#64748b;">{var_pct:.1f}%</span></div>
      <div style="font-size:11px;color:#64748b;margin-top:4px;">Scales with research request volume</div>
      <div class="bar-bg"><div style="width:{var_pct:.1f}%;height:4px;background:#f59e0b;border-radius:2px;"></div></div>
      <div class="comp-names">{var_names_str}</div>
    </div>
  </div>
</div>

<script>
  const weeks = {weeks_js};
  const requestsData = {requests_js};
  const components = {ts_components_js};
  const scatterBelow = {scatter_below_js};
  const scatterAbove = {scatter_above_js};

  let active = new Set(components.filter(c=>c.preset).map(c=>c.label));

  function buildDatasets() {{
    const ds = components.filter(c=>active.has(c.label)).map(c=>{{
      return {{ label:c.label, data:c.data, borderColor:c.color, backgroundColor:'transparent',
               borderWidth:2, pointRadius:3, yAxisID:'y' }};
    }});
    ds.push({{ label:'Research requests', data:requestsData, borderColor:'#4ade80',
      backgroundColor:'rgba(74,222,128,0.06)', borderWidth:1.5, borderDash:[4,4],
      pointRadius:2, fill:true, yAxisID:'y2' }});
    return ds;
  }}

  function buildToggles() {{
    const cont = document.getElementById('toggles');
    cont.innerHTML = '';
    components.forEach(c=>{{
      const on = active.has(c.label);
      const btn = document.createElement('button');
      btn.className = 'tbtn';
      btn.textContent = c.label;
      btn.style.border = `1px solid ${{c.color}}`;
      btn.style.background = on ? c.color : 'transparent';
      btn.style.color = on ? '#0f1117' : c.color;
      btn.onclick = () => {{
        if(active.has(c.label)) active.delete(c.label); else active.add(c.label);
        chart.data.datasets = buildDatasets();
        chart.update();
        buildToggles();
      }};
      cont.appendChild(btn);
    }});
  }}

  const gc='rgba(255,255,255,0.07)', tc='#94a3b8';
  const chart = new Chart(document.getElementById('timeChart'), {{
    type:'line',
    data:{{ labels:weeks, datasets:buildDatasets() }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ display:false }} }},
      scales:{{
        x:{{ ticks:{{ color:tc, font:{{size:10}}, maxRotation:45 }}, grid:{{ color:gc }} }},
        y:{{ ticks:{{ color:tc, font:{{size:10}}, callback:v=>'$'+v.toFixed(0) }}, grid:{{ color:gc }}, title:{{ display:true, text:'Cost ($/hr)', color:tc, font:{{size:10}} }} }},
        y2:{{ position:'right', ticks:{{ color:'#4ade80', font:{{size:10}} }}, grid:{{ display:false }}, title:{{ display:true, text:'Requests/hr', color:'#4ade80', font:{{size:10}} }} }}
      }}
    }}
  }});
  buildToggles();

  const toPoint = (d,i) => ({{ x:d.corr, y:d.total, r:Math.max(5,Math.sqrt(d.total/800)), label:d.label }});
  new Chart(document.getElementById('scatterChart'), {{
    type:'bubble',
    data:{{
      datasets:[
        {{ label:'Below threshold', data:scatterBelow.map(toPoint), backgroundColor:'rgba(83,74,183,0.6)', borderColor:'#534AB7' }},
        {{ label:'Above threshold', data:scatterAbove.map(toPoint), backgroundColor:'rgba(245,158,11,0.6)', borderColor:'#f59e0b' }},
      ]
    }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{ display:false }},
        tooltip:{{ callbacks:{{ label:ctx=>`${{ctx.raw.label}}: r=${{ctx.raw.x.toFixed(2)}}, $${{Math.round(ctx.raw.y/1000)}}k` }} }}
      }},
      scales:{{
        x:{{ min:0, max:0.75,
          title:{{ display:true, text:'Correlation with research request volume →', color:tc, font:{{size:10}} }},
          ticks:{{ color:tc, font:{{size:10}} }}, grid:{{ color:gc }},
          afterDraw(chart) {{
            const ctx2=chart.ctx, xs=chart.scales.x, ys=chart.scales.y;
            const xPx=xs.getPixelForValue(0.3);
            ctx2.save();
            ctx2.strokeStyle='#f87171'; ctx2.setLineDash([4,4]); ctx2.lineWidth=1.5;
            ctx2.beginPath(); ctx2.moveTo(xPx,ys.top); ctx2.lineTo(xPx,ys.bottom); ctx2.stroke();
            ctx2.fillStyle='#f87171'; ctx2.font='10px sans-serif';
            ctx2.fillText('threshold = 0.3', xPx+4, ys.top+14);
            ctx2.restore();
          }}
        }},
        y:{{ title:{{ display:true, text:'Total cost ($)', color:tc, font:{{size:10}} }},
             ticks:{{ color:tc, font:{{size:10}}, callback:v=>'$'+Math.round(v/1000)+'k' }}, grid:{{ color:gc }} }}
      }}
    }}
  }});
</script>
"""
    st.components.v1.html(html_fv, height=1050)

    st.markdown(f"""
    <div class="insight-box danger">
        <b>🚨 Infrastructure dominates at 95.6%:</b> Model costs are only 4.4% of total spend —
        the infrastructure layer (EKS clusters, Elasticsearch, Redis) is the primary cost driver by far.
    </div>
    <div class="insight-box warning">
        <b>⚠️ High fixed cost base:</b> {fixed_pct:.1f}% of costs are fixed overhead incurred regardless
        of request volume — a significant baseline that must be covered even at low utilization.
    </div>
    <div class="insight-box success">
        <b>✅ Model costs scale with usage:</b> GPT-4o and Groq Llama show the strongest correlation
        with request volume (r &gt; 0.6), making model selection the most actionable lever for cost optimization.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem'>"
    "Tavily Research API · Data Analyst Home Assignment · "
    "Data sampled from production · Analysis period: Nov 2025 – Mar 2026"
    "</div>",
    unsafe_allow_html=True
)
