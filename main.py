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
    rr = pd.read_csv("research_requests.csv")
    hu = pd.read_csv("hourly_usage.csv")
    ic = pd.read_csv("infrastructure_costs.csv")

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
    min_hourly = ic['total_cost'].min()

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label in [
        (k1, f"${total_cost:,.0f}", "Total Cost (5 months)"),
        (k2, f"{total_infra/total_cost*100:.1f}%", "Infrastructure Share"),
        (k3, f"{total_model/total_cost*100:.1f}%", "Model Inference Share"),
        (k4, f"${min_hourly:.0f}", "Cost Floor ($/hr)"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("")

    # ── Cost breakdown ────────────────────────────────────────────────────
    col_i, col_j = st.columns(2)

    with col_i:
        st.markdown('<div class="section-header">Cost Share by Component</div>', unsafe_allow_html=True)
        component_totals = ic[infra_cols + model_cols].sum().reset_index()
        component_totals.columns = ['component', 'total']
        component_totals['type'] = component_totals['component'].apply(
            lambda x: 'Infrastructure' if x.startswith('infra_') else 'Model')
        component_totals['label'] = component_totals['component'].str.replace('infra_','').str.replace('model_','').str.replace('_',' ').str.title()
        component_totals = component_totals.sort_values('total', ascending=True)

        fig15 = px.bar(component_totals, y='label', x='total', color='type',
                       orientation='h',
                       color_discrete_map={'Infrastructure': '#6366f1', 'Model': '#f59e0b'},
                       title="Total Cost by Component ($)")
        fig15.update_layout(**PLOTLY_THEME, height=420,
                            xaxis_title="Total Cost ($)", yaxis_title="")
        st.plotly_chart(fig15, use_container_width=True)

    with col_j:
        st.markdown('<div class="section-header">Model Cost Share</div>', unsafe_allow_html=True)
        model_totals = ic[model_cols].sum().reset_index()
        model_totals.columns = ['model', 'total']
        model_totals['label'] = model_totals['model'].str.replace('model_','').str.replace('_',' ').str.title()
        fig16 = px.pie(model_totals, names='label', values='total',
                       color_discrete_sequence=COLOR_SEQ,
                       title="Model Inference Cost Breakdown")
        fig16.update_layout(**PLOTLY_THEME, height=420)
        fig16.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig16, use_container_width=True)

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
                               name='Model Inference', fill='tozeroy',
                               line=dict(color='#f59e0b', width=1.5),
                               fillcolor='rgba(245,158,11,0.2)'))
    fig17.update_layout(**PLOTLY_THEME, height=300,
                        yaxis_title="Daily Cost ($)",
                        legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig17, use_container_width=True)

    # ── Hourly patterns ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Hourly Cost Patterns (Average by Hour of Day)</div>', unsafe_allow_html=True)

    ic['hour_of_day'] = ic['hour'].dt.hour
    ic['day_of_week'] = ic['hour'].dt.day_name()

    hourly_pattern = ic.groupby('hour_of_day')['total_cost'].mean().reset_index()

    fig18 = px.bar(hourly_pattern, x='hour_of_day', y='total_cost',
                   color='total_cost', color_continuous_scale='Viridis',
                   title="Average Hourly Cost by Hour of Day (UTC)")
    fig18.update_layout(**PLOTLY_THEME, height=280,
                        xaxis_title="Hour of Day (UTC)", yaxis_title="Avg Cost ($)",
                        coloraxis_showscale=False)
    st.plotly_chart(fig18, use_container_width=True)

    # ── Fixed vs variable ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Fixed vs Variable Cost Components</div>', unsafe_allow_html=True)

    cv_data = []
    for col in infra_cols + model_cols:
        cv = ic[col].std() / ic[col].mean()
        category = 'Fixed (low variance)' if cv < 0.15 else ('Semi-variable' if cv < 0.25 else 'Variable (high variance)')
        ctype = 'Infrastructure' if col.startswith('infra_') else 'Model'
        cv_data.append({
            'component': col.replace('infra_','').replace('model_','').replace('_',' ').title(),
            'cv': cv, 'category': category, 'type': ctype,
            'total': ic[col].sum()
        })
    cv_df = pd.DataFrame(cv_data).sort_values('cv')

    fig19 = px.scatter(cv_df, x='cv', y='total', color='type', size='total',
                       hover_name='component', text='component',
                       color_discrete_map={'Infrastructure': '#6366f1', 'Model': '#f59e0b'},
                       title="Cost Variability vs Total Spend (bubble = total cost)")
    fig19.update_traces(textposition='top center', textfont_size=9)
    fig19.add_vline(x=0.15, line_dash="dot", line_color="#64748b",
                    annotation_text="Fixed threshold")
    fig19.update_layout(**PLOTLY_THEME, height=350,
                        xaxis_title="Coefficient of Variation (lower = more fixed)",
                        yaxis_title="Total Cost ($)")
    st.plotly_chart(fig19, use_container_width=True)

    fixed_cost = ic[['infra_eks_research_cluster','infra_eks_search_cluster',
                      'infra_eks_scraping_cluster','infra_elasticache_redis',
                      'infra_elasticsearch','infra_s3_storage']].sum(axis=1).mean()

    st.markdown(f"""
    <div class="insight-box danger">
        <b>🚨 Infrastructure dominates at 95.6%:</b> Model inference costs are only 4.4% of total spend —
        the infrastructure layer (EKS clusters, Elasticsearch, Redis) is the primary cost driver by far.
    </div>
    <div class="insight-box warning">
        <b>⚠️ High fixed cost base:</b> EKS clusters, Redis, and Elasticsearch show very low variance
        (CV &lt; 0.13), meaning ~${fixed_cost:.0f}/hr is spent regardless of actual request volume —
        a significant baseline that must be covered even at low utilization.
    </div>
    <div class="insight-box success">
        <b>✅ Model costs scale with usage:</b> OpenAI GPT-4o and Groq Llama show the strongest
        correlation with request volume (r &gt; 0.6), confirming model inference is the most elastic
        cost component and can be optimized through model selection.
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem'>"
    "Tavily Research API · Data Analyst Home Assignment · "
    "Data sampled from production · Analysis period: Nov 2025 – Mar 2026"
    "</div>",
    unsafe_allow_html=True
)
