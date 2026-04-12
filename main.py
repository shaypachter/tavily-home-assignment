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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .kpi-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
    .kpi-value { font-size: 2rem; font-weight: 600; color: #0f172a; font-family: 'DM Mono', monospace; line-height: 1.2; }
    .kpi-label { font-size: 0.75rem; color: #0f172a; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }
    .kpi-delta-pos { color: #4ade80; font-size: 0.85rem; }
    .kpi-delta-neg { color: #f87171; font-size: 0.85rem; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #0f172a; text-transform: uppercase; letter-spacing: 0.1em; margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #e2e8f0; }
    .insight-box { background: #f8fafc; border-left: 3px solid #6366f1; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.5rem 0; font-size: 0.88rem; color: #0f172a; }
    .insight-box.warning { border-left-color: #f59e0b; }
    .insight-box.success { border-left-color: #4ade80; }
    .insight-box.danger  { border-left-color: #f87171; }
    h1 { color: #0f172a !important; font-weight: 600 !important; }
    h2, h3 { color: #0f172a !important; font-weight: 500 !important; }
    .stTabs [data-baseweb="tab"] { color: #64748b; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    font_family="DM Sans", font_color="#0f172a",
)
AXIS_STYLE = dict(tickfont=dict(color='#0f172a'), title_font=dict(color='#0f172a'), gridcolor='#f1f5f9')
COLOR_SEQ = ["#6366f1", "#4ade80", "#f59e0b", "#f87171", "#38bdf8", "#a78bfa"]

COLORS = {"blue": "#378ADD", "green": "#1D9E75", "red": "#E24B4A", "gray": "#B4B2A9", "purple": "#7F77DD"}

@st.cache_data
def load_data():
    def load_file(name):
        if os.path.exists(f"{name}.csv.gz"):
            return pd.read_csv(f"{name}.csv.gz", compression="gzip")
        return pd.read_csv(f"{name}.csv")

    rr = load_file("research_requests")
    hu = load_file("hourly_usage")
    ic = load_file("infrastructure_costs")
    users_df = load_file("users")

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

    # Q1/Q2/Q3 pre-computations
    rr['week_p'] = rr['TIMESTAMP'].dt.to_period('W')
    complete_weeks = sorted(rr['week_p'].unique())[1:-1]
    rr_clean = rr[rr['week_p'].isin(complete_weeks)].copy()

    first_week = rr_clean.groupby('USER_ID')['week_p'].min().rename('cohort_week')
    rr_u = rr_clean.join(first_week, on='USER_ID')
    user_weeks = rr_clean.groupby('USER_ID')['week_p'].nunique().rename('active_weeks')
    rr_u = rr_u.join(user_weeks, on='USER_ID')
    rr_u['retained'] = (rr_u['active_weeks'] > 1).astype(int)
    rr_u['week_num'] = (rr_u['week_p'] - rr_u['cohort_week']).apply(lambda x: x.n)

    cohort_sizes = first_week.value_counts().rename('cohort_size')
    retention = rr_u.groupby(['cohort_week', 'week_num'])['USER_ID'].nunique().reset_index()
    retention.columns = ['cohort_week', 'week_num', 'active_users']
    retention = retention.join(cohort_sizes, on='cohort_week')
    retention['retention_rate'] = retention['active_users'] / retention['cohort_size']
    avg_ret = retention[retention['week_num'] <= 8].groupby('week_num')['retention_rate'].mean().reset_index()

    w0 = rr_u[rr_u['week_p'] == rr_u['cohort_week']]
    w0_agg = w0.groupby('USER_ID').agg(
        retained=('retained', 'first'),
        primary_client=('CLIENT_SOURCE', lambda x: x.mode()[0]),
        w0_requests=('REQUEST_ID', 'count'),
    ).reset_index()
    w0_agg = w0_agg.merge(users_df[['USER_ID', 'PLAN', 'HAS_PAYGO']], on='USER_ID', how='left')

    import pandas as _pd
    _bins = [0, 1, 2, 5, 10, 50, 999]
    _labels = ['1', '2', '3-5', '6-10', '11-50', '50+']
    w0_agg['req_bucket'] = _pd.cut(w0_agg['w0_requests'], bins=_bins, labels=_labels)
    w0_requests_ret = (
        w0_agg.groupby('req_bucket', observed=True)['retained']
        .agg(['mean', 'count']).reset_index()
        .rename(columns={'req_bucket': 'bucket', 'mean': 'retention_rate', 'count': 'user_count'})
    )

    plan_ret = (
        w0_agg.groupby('PLAN')['retained'].agg(['mean', 'count']).reset_index()
        .rename(columns={'PLAN': 'plan', 'mean': 'retention_rate', 'count': 'user_count'})
        .sort_values('retention_rate', ascending=False)
    )

    success = rr_clean[rr_clean['STATUS'] == 'success']
    plan_stats = success.merge(users_df[['USER_ID', 'PLAN']], on='USER_ID', how='left')
    plan_charge = plan_stats.groupby('PLAN').apply(lambda x: (x['CREDITS_USED'] > 0).mean()).reset_index()
    plan_charge.columns = ['plan', 'charge_rate']
    plan_charge['plan'] = plan_charge['plan'].astype(str)
    plan_charge = plan_charge.sort_values('charge_rate', ascending=False)

    weekly_status = (
        rr_clean.groupby('week_p')['STATUS'].value_counts(normalize=True)
        .unstack(fill_value=0).reset_index()
    )
    weekly_status['week_str'] = weekly_status['week_p'].astype(str).str[:10]

    success_only = rr_clean[rr_clean['STATUS'] == 'success']
    rt_weekly = (
        success_only.groupby('week_p')['RESPONSE_TIME_SECONDS']
        .agg(p50=lambda x: x.quantile(0.5), p95=lambda x: x.quantile(0.95)).reset_index()
    )
    rt_weekly['week_str'] = rt_weekly['week_p'].astype(str).str[:10]

    model_latency = (
        success_only.groupby('MODEL')['RESPONSE_TIME_SECONDS']
        .agg(p50=lambda x: x.quantile(0.5), p95=lambda x: x.quantile(0.95)).reset_index()
    )

    total_cost_rr = rr_clean['REQUEST_COST'].sum()
    uncharged_cost = success[success['CREDITS_USED'] == 0]['REQUEST_COST'].sum()
    charged_cost = success[success['CREDITS_USED'] > 0]['REQUEST_COST'].sum()
    fc_cost = rr_clean[rr_clean['STATUS'].isin(['failed', 'cancelled'])]['REQUEST_COST'].sum()

    user_weeks_ser = rr_u.groupby('USER_ID')['active_weeks'].first()
    total_users = rr_u['USER_ID'].nunique()
    one_done = (user_weeks_ser == 1).sum()
    power = (user_weeks_ser >= 4).sum()

    return dict(
        rr=rr, hu=hu, ic=ic,
        infra_cols=infra_cols, model_cols=model_cols,
        avg_ret=avg_ret, plan_ret=plan_ret,
        plan_charge=plan_charge, weekly_status=weekly_status,
        rt_weekly=rt_weekly, model_latency=model_latency,
        total_cost_rr=total_cost_rr, uncharged_cost=uncharged_cost,
        charged_cost=charged_cost, fc_cost=fc_cost,
        total_users=total_users, one_done=one_done, power=power,
        success_count=len(success),
        uncharged_count=(success['CREDITS_USED'] == 0).sum(),
        w0_requests_ret=w0_requests_ret,
        rr_clean=rr_clean,
    )

data = load_data()
rr = data['rr']
ic = data['ic']
infra_cols = data['infra_cols']
model_cols = data['model_cols']

# Sidebar
with st.sidebar:
    st.markdown("## 🔬 Research API")
    st.markdown("**Analytics Dashboard**")
    st.markdown("---")
    st.markdown("**Data range**")
    st.markdown(f"`{rr['TIMESTAMP'].min().date()}` → `{rr['TIMESTAMP'].max().date()}`")
    st.markdown("**Total requests**")
    st.markdown(f"`{len(rr):,}`")
    st.markdown("**Unique users**")
    st.markdown(f"`{rr['USER_ID'].nunique():,}`")
    st.markdown("---")
    st.markdown("**Questions**")
    st.markdown("1. 📈 Retention: Do users come back?")
    st.markdown("2. 💰 Profitability: Money on the floor?")
    st.markdown("3. 🩺 Technical health")
    st.markdown("4. 🏗️ Infrastructure costs")

# Header
st.markdown("# Tavily Research API — Leadership Dashboard")
st.markdown("*Production data · Nov 2025 – Mar 2026 · Sampled dataset · Complete weeks only*")
st.markdown("---")



tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Retention",
    "💰  Profitability",
    "🩺  Technical Health",
    "🏗️  Infrastructure Costs",
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — RETENTION
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Q1: Does the Research API retain users after their first week?")
    st.markdown('<div class="insight-box success"><b>Hypothesis:</b> Most users treat the Research API as a one-time experiment rather than integrating it into a recurring workflow. Users who arrive with a real integration (not just exploration) - signaled by their client source, usage depth, and plan type - will retain at significantly higher rates.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, val, label, note in [
        (c1, f"{data['one_done'] / data['total_users']:.0%}", "One-time users", "never return after week 0"),
        (c2, "22%", "Week-1 retention", "avg across cohorts"),
        (c3, f"{data['power'] / data['total_users']:.1%}", "Power users (4+ weeks)", "active 4+ weeks · generate 79% of credits charged"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div><div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{note}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Avg retention curve</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['avg_ret']['week_num'], y=data['avg_ret']['retention_rate'],
            mode='lines+markers', line=dict(color=COLORS['blue'], width=2),
            fill='tozeroy', fillcolor='rgba(55,138,221,0.08)', marker=dict(size=5),
            hovertemplate='Week %{x}: %{y:.0%}<extra></extra>',
        ))
        fig.update_layout(**PLOTLY_THEME, height=260, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(showgrid=False, dtick=1, title='Weeks since first use'),
                          yaxis=dict(tickformat='.0%', gridcolor='#f1f5f9', title='Retention rate'))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Retention rate by plan</div>', unsafe_allow_html=True)
        plan_ret = data['plan_ret']

        def plan_color(p, r):
            if p == 'enterprise': return '#1D9E75'
            if p == 'researcher': return '#E24B4A'
            return '#378ADD'

        rows_html = ""
        for _, row in plan_ret.iterrows():
            color = plan_color(row['plan'], row['retention_rate'])
            pct = row['retention_rate'] * 100
            count = f"{int(row['user_count']):,}"
            rows_html += f"""
            <tr>
                <td style="padding:9px 12px;font-size:13px;font-weight:500;color:#0f172a;">{row['plan']}</td>
                <td style="padding:9px 12px;font-size:12px;color:#64748b;">{count}</td>
                <td style="padding:9px 12px;font-size:13px;font-variant-numeric:tabular-nums;color:#0f172a;">{pct:.0f}%</td>
                <td style="padding:9px 12px;width:40%;">
                    <div style="position:relative;height:10px;border-radius:3px;background:{color};width:{pct:.0f}%;">
                        <span style="position:absolute;left:calc(100% + 6px);top:50%;transform:translateY(-50%);font-size:11px;color:#64748b;white-space:nowrap;">{pct:.0f}%</span>
                    </div>
                </td>
            </tr>"""

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:0.5px solid #e2e8f0;">
                    <th style="padding:6px 12px;font-size:11px;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;text-align:left;">Plan</th>
                    <th style="padding:6px 12px;font-size:11px;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;text-align:left;">Users</th>
                    <th colspan="2" style="padding:6px 12px;font-size:11px;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;text-align:left;">Retention</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>"""
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Retention rate by week-0 behavior</div>', unsafe_allow_html=True)
    seg_labels = ['MCP users', 'Non-MCP users', 'PayGo enabled', 'No PayGo', 'Non-streaming', 'Streaming']
    seg_values = [0.322, 0.187, 0.524, 0.238, 0.283, 0.168]
    seg_colors = [COLORS['blue'], COLORS['gray'], COLORS['blue'], COLORS['gray'], COLORS['blue'], COLORS['gray']]
    fig = go.Figure(go.Bar(
        x=seg_labels, y=seg_values, marker_color=seg_colors,
        text=[f"{v:.0%}" for v in seg_values], textposition='outside',
        customdata=seg_labels, hovertemplate='%{customdata}: %{y:.0%}<extra></extra>',
    ))
    fig.update_layout(**PLOTLY_THEME, height=260, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(tickformat='.0%', range=[0, 0.65], gridcolor='#f1f5f9', title='Retention rate'),
                      xaxis=dict(showgrid=False, title='Segment'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Retention rate by week-0 request volume</div>', unsafe_allow_html=True)
    w0r = data['w0_requests_ret']
    fig = go.Figure(go.Bar(
        x=w0r['bucket'],
        y=w0r['retention_rate'],
        marker_color='#0C447C',
        text=[f"{v:.0%}" for v in w0r['retention_rate']],
        textposition='outside',
        customdata=w0r['user_count'],
        hovertemplate='%{x} requests: %{y:.0%} retained (%{customdata:,} users)<extra></extra>',
    ))
    fig.update_layout(**PLOTLY_THEME, height=260, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(tickformat='.0%', range=[0, 0.5], gridcolor='#f1f5f9', title='Retention rate'),
                      xaxis=dict(showgrid=False, title='Requests made in week 0'))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Findings & recommendation"):
        st.markdown("""
**Findings**

Approximately 74% of users are one-time users, while average week-1 retention across cohorts is around 22%, and only 4.8% qualify as power users (active for four or more weeks).

The strongest predictors of retention from first-time-use behavior:
- Plan type - enterprise retains 100%, bootstrap/growth/project retain 50-76%, but researcher (96% of users) retains only 25%. The product is largely being tested by low-tier users who don't convert.
- Pay As You Go enabled = double retention rate - users with pay-as-you-go retain at 52% (vs 24% without PayGo). These users have a vested interest and are more likely to develop production-grade workflows.
- MCP integration = 1.7x retention - MCP users retain at 32% vs 19% (for non-MCP). MCP users have already embedded the API into a tool, meaning they're builders, not explorers.
- More requests in week 0 = higher retention. Going from 1 to 6-10 requests in the first week lifts retention from 21% to 35%. Early depth predicts stickiness.
- Streaming users churn more - streaming retention is 17% vs 28% for non-streaming. Streaming is likely a quick-test behavior, not a production integration pattern.

**Recommendation**

The users who integrate properly (MCP, PayGo, higher plans) retain well. The issue is that most users never reach the point where the product becomes useful in their workflow. Focus onboarding on driving users toward a real integration - specifically push toward MCP setup and structured output usage, and identify researcher-plan users who look like builders (high week-0 volume) for upgrade prompts. Those users seem to be on the wrong plan.
        """)

# ═══════════════════════════════════════════════════════════
# TAB 2 — PROFITABILITY
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Q2: Is the Research API profitable? Are there cases of 'money on the floor'?")
    st.markdown('<div class="insight-box warning"><b>Hypothesis:</b> When requests fail or are cancelled mid-run, the system has already consumed compute, LLM calls, and search operations but likely charges nothing. This partial work represents unrecovered cost and a structural profitability leak. Finding the reasons for cancellations (long running times, lack of credits) or failures (technical issues) might help improve profitability.</div>', unsafe_allow_html=True)

    recovery_rate = data['charged_cost'] / data['total_cost_rr']
    uncharged_pct = data['uncharged_cost'] / data['total_cost_rr']
    uncharged_req_pct = data['uncharged_count'] / data['success_count']

    rr_clean = data['rr_clean']
    n_failed = int((rr_clean['STATUS'] == 'failed').sum())
    n_cancelled = int((rr_clean['STATUS'] == 'cancelled').sum())
    n_total = len(rr_clean)
    fc_pct_cost = data['fc_cost'] / data['total_cost_rr']

    k1, k2, k3, k4 = st.columns(4)
    for col, val, label, note in [
        (k1, f"{n_failed:,}", "Failed requests", f"{n_failed/n_total:.1%} of all requests"),
        (k2, f"{n_cancelled:,}", "Cancelled requests", f"{n_cancelled/n_total:.1%} of all requests"),
        (k3, f"{fc_pct_cost:.2%}", "Their share of total cost", "almost nothing"),
        (k4, "3,081", "Credits recovered on cancelled", "billing partially works here"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div><div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{note}</div></div>', unsafe_allow_html=True)

    insight_html = '<div class="insight-box success" style="margin-top:0.75rem;"><b>Hypothesis disproved.</b> Failed and cancelled requests are a non-issue financially - despite averaging 78-134s of runtime and consuming real LLM and search calls.</div>'
    st.markdown(insight_html, unsafe_allow_html=True)


    c1, c2, c3 = st.columns(3)
    uncharged_count = data['uncharged_count']
    uncharged_cost_abs = data['uncharged_cost']
    for col, val, label, note in [
        (c1, f"{recovery_rate:.0%}", "Cost recovery rate", "credits charged / total system cost"),
        (c2, f"{uncharged_req_pct:.0%}  ({uncharged_count:,} requests)", "Successful but uncharged", "successful requests with 0 credits"),
        (c3, f"{uncharged_pct:.0%}", "Cost delivered free", f"~{uncharged_cost_abs:,.0f} cost units"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div><div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{note}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-header">Where the cost goes</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Recovered', 'Uncharged success', 'Failed/cancelled'],
            values=[data['charged_cost'], data['uncharged_cost'], data['fc_cost']],
            hole=0.65, marker_colors=[COLORS['green'], COLORS['red'], COLORS['gray']],
            textinfo='label+percent', hovertemplate='%{label}: %{percent}<extra></extra>',
        ))
        fig.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                          margin=dict(l=10, r=10, t=10, b=10),
                          annotations=[dict(text='cost split', x=0.5, y=0.5, font_size=12, showarrow=False, font_color='#888')])
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Charge rate by plan — % of successful requests billed</div>', unsafe_allow_html=True)
        pc = data['plan_charge']
        bar_colors = [COLORS['green'] if r >= 0.8 else COLORS['red'] if r == 0 else COLORS['blue'] for r in pc['charge_rate']]
        fig = go.Figure(go.Bar(
            x=pc['charge_rate'], y=pc['plan'], orientation='h', marker_color=bar_colors,
            text=[f"{v:.0%}" for v in pc['charge_rate']], textposition='outside',
            hovertemplate='%{y}: %{x:.0%}<extra></extra>',
        ))
        fig.update_layout(**PLOTLY_THEME, height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(tickformat='.0%', range=[0, 1.15], gridcolor='#f1f5f9', title='% of successful requests charged'),
                          yaxis=dict(showgrid=False, title='Plan'))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Findings & recommendation"):
        st.markdown("""
**Findings**

The hypothesis is disproved, but revealed a different and bigger issue.

Failed and cancelled requests are a non-issue financially. Despite averaging 78-134 seconds of runtime and consuming LLM and search calls, they account for only 0.13% of total system cost - and the billing system even partially recovers some of that (3,081 credits charged on cancelled jobs).

The leak is hiding in successful requests, which I didn't expect to see as non-paid. Almost 99% of uncharged requests have status = success. 69% of all successful requests - where the system fully completed the job and delivered value - were charged 0 credits. That's 76,818 completed requests representing 72% of total system cost delivered for free.

The growth plan is the only one that charges credits (94% of requests). Other plans receive the product largely for free.

**Recommendation**

Audit the billing trigger logic per plan. Identify whether uncharged requests reflect intentional free-tier allowances or a billing bug. Finding the reasons for cancellations (long running times, lack of credits) or failures (technical issues) might also help improve profitability.
        """)

# ═══════════════════════════════════════════════════════════
# TAB 3 — TECHNICAL HEALTH
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Q3: Is the Research API reliable and consistent enough for production use?")
    st.markdown('<div class="insight-box success"><b>Hypothesis:</b> Rapid volume growth puts stress on infrastructure. As request volume grew through early 2026, failure rates likely increased and latency increased - making the product less reliable for production builders at a critical stage of adoption.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, val, label, note in [
        (c1, "97–98%", "Current success rate", "up from 84% in Dec"),
        (c2, "446s", "p95 response time", "1 in 20 requests > 7 min"),
        (c3, "52s – 251s", "Weekly p50 range", "volatile, no clear trend"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div><div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{note}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Weekly success rate</div>', unsafe_allow_html=True)
    ws = data['weekly_status']
    fig = go.Figure(go.Scatter(
        x=ws['week_str'], y=ws['success'], mode='lines+markers',
        line=dict(color=COLORS['green'], width=2), fill='tozeroy',
        fillcolor='rgba(29,158,117,0.07)', marker=dict(size=4),
        hovertemplate='%{x}: %{y:.1%}<extra></extra>',
    ))
    fig.add_hline(y=0.95, line_dash='dot', line_color='#f59e0b',
                  annotation_text='95% target', annotation_position='bottom right')
    fig.update_layout(**PLOTLY_THEME, height=220, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(tickformat='.0%', range=[0.75, 1.02], gridcolor='#f1f5f9', title='Success rate'),
                      xaxis=dict(showgrid=False, tickangle=45, title='Week'))
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Weekly latency — p50 vs p95 (seconds)</div>', unsafe_allow_html=True)
        rt = data['rt_weekly']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rt['week_str'], y=rt['p50'], name='p50', mode='lines+markers',
                                 line=dict(color=COLORS['blue'], width=2), marker=dict(size=4),
                                 hovertemplate='%{x} p50: %{y:.0f}s<extra></extra>'))
        fig.add_trace(go.Scatter(x=rt['week_str'], y=rt['p95'], name='p95', mode='lines+markers',
                                 line=dict(color=COLORS['red'], width=2, dash='dash'), marker=dict(size=4),
                                 hovertemplate='%{x} p95: %{y:.0f}s<extra></extra>'))
        fig.update_layout(**PLOTLY_THEME, height=280, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(title='Response time (seconds)', gridcolor='#f1f5f9'),
                          xaxis=dict(showgrid=False, tickangle=45, title='Week'),
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Mini vs pro — latency profile</div>', unsafe_allow_html=True)
        ml = data['model_latency']
        ml = ml[ml['MODEL'].isin(['mini', 'pro'])]
        fig = go.Figure()
        for model, color in [('mini', COLORS['blue']), ('pro', COLORS['purple'])]:
            row = ml[ml['MODEL'] == model].iloc[0]
            fig.add_trace(go.Bar(
                name=model, x=['p50', 'p95'], y=[row['p50'], row['p95']], marker_color=color,
                text=[f"{row['p50']:.0f}s", f"{row['p95']:.0f}s"], textposition='outside',
                hovertemplate=f"{model} %{{x}}: %{{y:.0f}}s<extra></extra>",
            ))
        fig.update_layout(**PLOTLY_THEME, height=280, barmode='group',
                          margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(title='Response time (seconds)', gridcolor='#f1f5f9', range=[0, 620]),
                          xaxis=dict(showgrid=False, title='Latency percentile'),
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Findings & recommendation"):
        st.markdown("""
**Findings**

The hypothesis is disproved - reliability actually improved over time. Success rates started volatile (84-95% in Dec/Jan) and stabilized at 97-98% by March. Yet - keep working on the 2-3% failure rate, at current volume that's still 200-300 real users failing per week.

Latency tells a different story. The p50 response time is wildly inconsistent week to week - bouncing between 52s and 251s with no clear trend. The p95 sits at 446 seconds, meaning 1 in 20 requests takes over 7 minutes. For a developer setting API timeouts, that might be a hard limitation - worth further investigation.

The top 5% of successful requests consume about 35 LLM calls and run 2.3x slower (374s vs 160s) at 2.7x the cost ($204 vs $77) compared to typical requests. Either those are genuinely harder queries, or the pipeline is looping unnecessarily.

Model profiles are very different and that matters for user expectations - document mini vs pro difference explicitly. Mini demonstrates high reliability and low latency (96.2% success rate; p50: 34s, p95: 75s) with relatively few LLM calls (7 on average). In contrast, Pro has a slightly lower success rate (94.9%), substantially higher latency (p50: 306s, p95: 512s), and requires more LLM calls (25 on average), resulting in a slower experience with a long-tail distribution.

**Recommendation**

Cap LLM calls per request to contain the tail. Document mini vs pro differences explicitly - users need to know before they pick. Investigate the p50 volatility by correlating with infrastructure cost data. Keep pushing on the 2-3% failure rate.
        """)

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
            <div class="kpi-value" style="font-size:1.4rem;">${avg_hourly:,.0f} <span style="font-size:0.85rem;font-weight:400;color:#0f172a;">/ hr</span></div>
            <div style="font-size:0.9rem;font-weight:500;color:#0f172a;margin-top:4px;">${avg_daily:,.0f} <span style="font-size:0.78rem;font-weight:400;color:#0f172a;">/ day</span></div>
            <div style="font-size:0.78rem;color:#0f172a;margin-top:3px;">${avg_weekly:,.0f} / wk &nbsp;·&nbsp; ${avg_monthly:,.0f} / mo</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Infrastructure Cost</div>
            <div class="kpi-value">{infra_pct:.1f}%</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">${total_infra:,.0f} total</div>
            <div style="margin-top:8px;height:4px;background:#e2e8f0;border-radius:2px;">
                <div style="width:{infra_pct:.1f}%;height:4px;background:#6366f1;border-radius:2px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Cost</div>
            <div class="kpi-value">{model_pct:.1f}%</div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">${total_model:,.0f} total</div>
            <div style="margin-top:8px;height:4px;background:#e2e8f0;border-radius:2px;">
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
  .tbtn {{ padding:4px 14px; font-size:12px; border-radius:6px; border:1px solid #cbd5e1; background:transparent; color:#64748b; cursor:pointer; }}
  .tbtn.active {{ background:#1e293b; color:#ffffff; }}
  .card {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1rem 1.25rem; }}
  .slabel {{ font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#0f172a; margin:0 0 0.75rem; }}
  #legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; font-size:11px; color:#0f172a; }}
</style>
<div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
  <span style="font-size:11px;color:#0f172a;text-transform:uppercase;letter-spacing:0.06em;">View:</span>
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
        <div id="centerPct" style="font-size:22px;font-weight:500;color:#0f172a;"></div>
        <div id="centerSub" style="font-size:11px;color:#0f172a;"></div>
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
      `<span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:2px;background:${{d.color}};display:inline-block;border:1px solid #e2e8f0;"></span>${{d.label}} ${{(d.value/TOTAL*100).toFixed(1)}}%</span>`
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
    pieChart.data.datasets[0].backgroundColor=[...pie.map(d=>d.color),f!=="all"?"#e2e8f0":null].filter(Boolean);
    pieChart.update();
    buildLegend(pie); updateCenter(f,pie);
  }}
  const {{bar:ib,pie:ip}}=getData("all");
  const gc="rgba(0,0,0,0.07)",tc="#0f172a";
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

    # Fixed vs variable
    st.markdown('<div class="section-header">Fixed vs Variable Cost Components</div>', unsafe_allow_html=True)

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

    comp_colors_fixed = ['#534AB7','#7F77DD','#AFA9EC','#9F97E6','#8F85E0','#6A62CC','#3C3489','#CECBF6','#26215C','#4f46e5','#818cf8','#c7d2fe']
    comp_colors_var   = ['#f59e0b','#d97706','#fbbf24','#b45309']

    ts_components = []
    fi = vi = 0
    largest_fixed = corr_df[corr_df['fixed']].nlargest(1,'total')['label'].values[0]
    largest_var   = corr_df[~corr_df['fixed']].nlargest(1,'total')['label'].values[0]

    for _, row in corr_df.sort_values('corr').iterrows():
        if row['fixed']:
            color = comp_colors_fixed[fi % len(comp_colors_fixed)]; fi += 1
        else:
            color = comp_colors_var[vi % len(comp_colors_var)]; vi += 1
        is_preset = row['label'] in [largest_fixed, largest_var]
        data_vals = weekly_merged[row['col']].round(2).tolist()
        ts_components.append({'label': row['label'], 'data': data_vals, 'color': color, 'preset': is_preset})

    ts_components_js = _json.dumps(ts_components)
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
  .card {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1rem 1.25rem; margin-bottom:1rem; }}
  .slabel {{ font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#0f172a; margin:0 0 0.4rem; }}
  .sdesc  {{ font-size:11px; color:#0f172a; margin:0 0 0.75rem; }}
  .tbtn {{ padding:3px 10px; font-size:11px; border-radius:6px; cursor:pointer; transition:all 150ms; }}
  #toggles {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:0.75rem; }}
  .legend-row {{ display:flex; gap:16px; margin-bottom:10px; font-size:11px; color:#64748b; }}
  .dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .concl-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .concl-card {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:1rem; }}
  .bar-bg {{ margin-top:8px; height:4px; background:#e2e8f0; border-radius:2px; }}
  .comp-names {{ margin-top:10px; font-size:11px; color:#0f172a; line-height:1.7; }}
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
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;">
  <p class="slabel">Step 3 — Conclusion: fixed vs variable split</p>
  <div class="concl-grid">
    <div class="concl-card">
      <div style="font-size:12px;color:#0f172a;margin-bottom:4px;">Fixed costs</div>
      <div style="font-size:22px;font-weight:500;color:#0f172a;">${fixed_total:,.0f} <span style="font-size:14px;color:#64748b;">{fixed_pct:.1f}%</span></div>
      <div style="font-size:11px;color:#0f172a;margin-top:4px;">Always-on regardless of usage</div>
      <div class="bar-bg"><div style="width:{fixed_pct:.1f}%;height:4px;background:#534AB7;border-radius:2px;"></div></div>
      <div class="comp-names">{fixed_names_str}</div>
    </div>
    <div class="concl-card">
      <div style="font-size:12px;color:#0f172a;margin-bottom:4px;">Variable costs</div>
      <div style="font-size:22px;font-weight:500;color:#0f172a;">${var_total:,.0f} <span style="font-size:14px;color:#64748b;">{var_pct:.1f}%</span></div>
      <div style="font-size:11px;color:#0f172a;margin-top:4px;">Scales with research request volume</div>
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
      btn.style.border = `1px solid ${{c.color}}`;btn.style.fontWeight='500';
      btn.style.background = on ? c.color : 'transparent';
      btn.style.color = on ? '#ffffff' : c.color;
      btn.onclick = () => {{
        if(active.has(c.label)) active.delete(c.label); else active.add(c.label);
        chart.data.datasets = buildDatasets();
        chart.update();
        buildToggles();
      }};
      cont.appendChild(btn);
    }});
  }}
  const gc='rgba(0,0,0,0.07)', tc='#0f172a';
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

    # Cost spikes
    st.markdown('<div class="section-header">Cost Spikes — Total Cost vs Research Request Volume</div>', unsafe_allow_html=True)

    import json as _json2
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
  .vbtn {{ padding:4px 14px; font-size:12px; border-radius:6px; border:1px solid #cbd5e1; background:transparent; color:#64748b; cursor:pointer; transition:all 150ms; }}
  .vbtn.active {{ background:#1e293b; color:#ffffff; }}
  .row {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:0.75rem; }}
  .legend {{ display:flex; gap:16px; margin-bottom:8px; font-size:11px; color:#0f172a; }}
  .stats {{ display:flex; gap:16px; margin-bottom:10px; font-size:11px; color:#0f172a; }}
</style>
<div class="row">
  <span style="font-size:11px;color:#0f172a;text-transform:uppercase;letter-spacing:0.06em;">View:</span>
  <button class="vbtn active" id="btn-daily" onclick="setView('daily')">Daily</button>
  <span style="font-size:11px;color:#0f172a;">Hourly zoom:</span>
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
  const tc='#0f172a', gc='rgba(0,0,0,0.07)';
  function makeThresholdLine(threshold) {{
    return function(chart) {{
      const ctx2=chart.ctx, xs=chart.scales.x, ys=chart.scales.y;
      const yPx=ys.getPixelForValue(threshold);
      if(yPx<ys.top||yPx>ys.bottom) return;
      ctx2.save();
      ctx2.strokeStyle='#f87171'; ctx2.setLineDash([4,4]); ctx2.lineWidth=1;
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
      <span style="color:#0f172a;">Threshold: <b style="color:#dc2626;">${{vdata.threshold.toFixed(0)}} ${{vdata.costUnit}}</b></span>
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

    # Cost efficiency
    st.markdown('<div class="section-header">Cost Efficiency — Cost per Research Request Over Time (Weekly)</div>', unsafe_allow_html=True)

    eff_weekly = merged_spike.copy()
    eff_weekly['week'] = pd.to_datetime(eff_weekly['hour_naive']).dt.to_period('W').dt.start_time
    eff_agg = eff_weekly.groupby('week').agg(
        total_cost=('total_cost','sum'),
        requests=('requests','sum')
    ).reset_index()
    eff_agg = eff_agg[eff_agg['requests'] > 0]
    eff_agg['cost_per_req'] = eff_agg['total_cost'] / eff_agg['requests']
    eff_agg['is_outlier'] = eff_agg['requests'] < 500

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
        yaxis_type="log",
        annotations=[dict(
            x=eff_agg['week'].iloc[-1], y=eff_agg['cost_per_req'].iloc[-1],
            text=f"${eff_agg['cost_per_req'].iloc[-1]:.2f}/req",
            showarrow=True, arrowhead=2, arrowcolor='#f59e0b',
            font=dict(color='#f59e0b', size=11)
        )]
    )
    st.plotly_chart(fig_eff, use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:0.8rem'>"
    "Tavily Research API · Data Analyst Home Assignment · "
    "Data sampled from production · Analysis period: Nov 2025 – Mar 2026"
    "</div>",
    unsafe_allow_html=True
)
