import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import json
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Tavily Research API — Analytics Dashboard",
    page_icon="https://tavily.com/favicon.ico",
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
    .kpi-card-alert { background: #fff8f8; border: 1px solid #fca5a5; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
    .kpi-value { font-size: 2rem; font-weight: 600; color: #0f172a; font-family: 'DM Mono', monospace; line-height: 1.2; }
    .kpi-label { font-size: 0.75rem; color: #0f172a; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }
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
        used_stream=('STREAM', lambda x: bool(x.any())),
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
    CREDIT_TO_USD = 0.008
    total_revenue_usd = rr_clean['CREDITS_USED'].sum() * CREDIT_TO_USD
    recovery_rate_usd = total_revenue_usd / total_cost_rr

    status_stats = []
    for _s in ['success', 'failed', 'cancelled', 'not_entitled']:
        _sub = rr_clean[rr_clean['STATUS'] == _s]
        _total = len(_sub)
        _charged = int((_sub['CREDITS_USED'] > 0).sum())
        _cost = float(_sub['REQUEST_COST'].sum())
        _recovered = float(_sub['CREDITS_USED'].sum()) * CREDIT_TO_USD
        status_stats.append({
            'status': _s,
            'total_req': _total,
            'charged_req': _charged,
            'uncharged_req': _total - _charged,
            'pct_charged': _charged / _total if _total > 0 else 0,
            'total_cost': _cost,
            'recovered': _recovered,
            'unrecovered': _cost - _recovered,
        })
    status_stats_df = pd.DataFrame(status_stats)

    user_weeks_ser = rr_u.groupby('USER_ID')['active_weeks'].first()
    total_users = rr_u['USER_ID'].nunique()
    one_done = (user_weeks_ser == 1).sum()
    power = (user_weeks_ser >= 4).sum()

    # Segment retention for tab1 bar chart
    w0_seg = w0_agg.copy()
    w0_seg['is_mcp'] = w0_seg['primary_client'] == 'mcp'
    seg_values_dynamic = [
        w0_seg[w0_seg['is_mcp']]['retained'].mean(),
        w0_seg[~w0_seg['is_mcp']]['retained'].mean(),
        w0_seg[w0_seg['HAS_PAYGO'] == True]['retained'].mean(),
        w0_seg[w0_seg['HAS_PAYGO'] == False]['retained'].mean(),
        w0_seg[~w0_seg['used_stream']]['retained'].mean(),
        w0_seg[w0_seg['used_stream']]['retained'].mean(),
    ]

    # Power user credit share
    _power_ids = user_weeks_ser[user_weeks_ser >= 4].index
    _power_credits = rr_clean[rr_clean['USER_ID'].isin(_power_ids)]['CREDITS_USED'].sum()
    _total_credits = rr_clean['CREDITS_USED'].sum()
    power_credit_share = _power_credits / _total_credits if _total_credits > 0 else 0

    # Failed users per week
    _weekly_failed_u = rr_clean[rr_clean['STATUS']=='failed'].groupby('week_p')['USER_ID'].nunique()
    _weekly_cancel_u = rr_clean[rr_clean['STATUS']=='cancelled'].groupby('week_p')['USER_ID'].nunique()
    _weekly_bad_u = (_weekly_failed_u.add(_weekly_cancel_u, fill_value=0))
    avg_failed_users_per_week = _weekly_bad_u.mean()

    # Top 5% vs typical 95% by LLM calls
    _llm_thresh = success_only['LLM_CALLS'].quantile(0.95)
    _top5 = success_only[success_only['LLM_CALLS'] >= _llm_thresh]
    _typ  = success_only[success_only['LLM_CALLS'] < _llm_thresh]
    pipeline_comparison = {
        'top5_llm':  _top5['LLM_CALLS'].mean(),
        'top5_time': _top5['RESPONSE_TIME_SECONDS'].mean(),
        'top5_cost': _top5['REQUEST_COST'].mean(),
        'typ_llm':   _typ['LLM_CALLS'].mean(),
        'typ_time':  _typ['RESPONSE_TIME_SECONDS'].mean(),
        'typ_cost':  _typ['REQUEST_COST'].mean(),
        'llm_thresh': _llm_thresh,
    }

    # Mini vs pro comparison table
    _model_stats = {}
    for _m in ['mini', 'pro']:
        _sub = success_only[success_only['MODEL'] == _m]
        _all_m = rr_clean[rr_clean['MODEL'] == _m]
        _model_stats[_m] = {
            'sr': len(_sub) / len(_all_m) if len(_all_m) > 0 else 0,
            'p50': _sub['RESPONSE_TIME_SECONDS'].quantile(0.5),
            'p95': _sub['RESPONSE_TIME_SECONDS'].quantile(0.95),
            'llm': _sub['LLM_CALLS'].mean(),
        }

    # Q3 enhanced KPI card data
    _rt_p95_w = success_only.groupby('week_p')['RESPONSE_TIME_SECONDS'].quantile(0.95)
    q3_early_p95 = _rt_p95_w.iloc[:3].mean()
    q3_recent_p95 = _rt_p95_w.iloc[-3:].mean()

    _weekly_active = rr_clean.groupby('week_p')['USER_ID'].nunique()
    _weekly_bad_u2 = rr_clean[rr_clean['STATUS'].isin(['failed','cancelled'])].groupby('week_p')['USER_ID'].nunique()
    _weekly_pct_bad = (_weekly_bad_u2 / _weekly_active * 100).fillna(0)
    q3_early_pct_bad = _weekly_pct_bad.iloc[:3].mean()
    q3_recent_pct_bad = _weekly_pct_bad.iloc[-3:].mean()
    q3_pct_bad_trend = _weekly_pct_bad.tolist()
    _weekly_sr = rr_clean.groupby('week_p')['STATUS'].apply(lambda x: (x=='success').mean())
    q3_early_sr = _weekly_sr.iloc[:3].mean()
    q3_recent_sr = _weekly_sr.iloc[-3:].mean()
    q3_n_weeks = len(complete_weeks)

    _rt_p95_weekly = success_only.groupby('week_p')['RESPONSE_TIME_SECONDS'].quantile(0.95)
    q3_p95_weekly_vals = _rt_p95_weekly.tolist()

    _wfu = rr_clean[rr_clean['STATUS']=='failed'].groupby('week_p')['USER_ID'].nunique()
    _wcu = rr_clean[rr_clean['STATUS']=='cancelled'].groupby('week_p')['USER_ID'].nunique()
    _wbu = _wfu.add(_wcu, fill_value=0)
    q3_early_bad = _wbu.iloc[:3].mean()
    q3_recent_bad = _wbu.iloc[-3:].mean()
    q3_bad_trend = _wbu.tolist()

    # Dynamic KPI values
    weekly_sr = rr_clean.groupby('week_p')['STATUS'].apply(lambda x: (x=='success').mean())
    last3_sr = weekly_sr.iloc[-3:]
    current_sr_str = f"{last3_sr.min():.0%} - {last3_sr.max():.0%}"
    early_sr = weekly_sr.iloc[:4].mean()
    early_sr_str = f"{early_sr:.0%}"

    p95_overall = success_only['RESPONSE_TIME_SECONDS'].quantile(0.95)
    rt_weekly_p50 = success_only.groupby('week_p')['RESPONSE_TIME_SECONDS'].quantile(0.5)
    p50_min = rt_weekly_p50.min()
    p50_max = rt_weekly_p50.max()
    p50_range_str = f"{p50_min:.0f}s - {p50_max:.0f}s"

    w1_retention = retention[retention['week_num']==1]['retention_rate'].mean()

    credits_cancelled = float(rr_clean[rr_clean['STATUS']=='cancelled']['CREDITS_USED'].sum())

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
        recovery_rate_usd=recovery_rate_usd,
        total_revenue_usd=total_revenue_usd,
        status_stats_df=status_stats_df,
        current_sr_str=current_sr_str,
        early_sr_str=early_sr_str,
        p95_overall=p95_overall,
        p50_range_str=p50_range_str,
        w1_retention=w1_retention,
        credits_cancelled=credits_cancelled,
        seg_values_dynamic=seg_values_dynamic,
        power_credit_share=power_credit_share,
        avg_failed_users_per_week=avg_failed_users_per_week,
        pipeline_comparison=pipeline_comparison,
        q3_early_sr=q3_early_sr,
        q3_recent_sr=q3_recent_sr,
        q3_n_weeks=q3_n_weeks,
        q3_p95_weekly_vals=q3_p95_weekly_vals,
        q3_early_bad=q3_early_bad,
        q3_recent_bad=q3_recent_bad,
        q3_bad_trend=q3_bad_trend,
        q3_early_p95=q3_early_p95,
        q3_recent_p95=q3_recent_p95,
        q3_early_pct_bad=q3_early_pct_bad,
        q3_recent_pct_bad=q3_recent_pct_bad,
        q3_pct_bad_trend=q3_pct_bad_trend,
        model_stats=_model_stats,
    )


data = load_data()
rr = data['rr']
ic = data['ic']
infra_cols = data['infra_cols']
model_cols = data['model_cols']

with st.sidebar:
    st.markdown("## Research API")
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
    st.markdown("1. Retention: Do users come back?")
    st.markdown("2. Profitability: Money on the floor?")
    st.markdown("3. Technical health")
    st.markdown("4. Infrastructure costs")

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
    st.markdown("### Q1: Does the Research API retain users after their first week,<br>and what behaviors in the first session predict whether a user will come back?", unsafe_allow_html=True)
    st.markdown('<div class="insight-box success"><b>Hypothesis:</b> Most users treat the Research API as a one-time experiment rather than integrating it into a recurring workflow. Users who arrive with a real integration (not just exploration) - signaled by their client source, usage depth, and plan type - will retain at significantly higher rates.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, val, label, note in [
        (c1, f"{data['one_done'] / data['total_users']:.0%}", "One-time users", "never return after week 0"),
        (c2, f"{data['w1_retention']:.0%}", "Week-1 retention", "avg across cohorts"),
        (c3, f"{data['power'] / data['total_users']:.1%}", "Power users (4+ weeks)", f"active 4+ weeks · generate {data['power_credit_share']:.0%} of credits charged"),
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
        _w1_val = data['avg_ret'][data['avg_ret']['week_num']==1]['retention_rate'].values
        _w1 = float(_w1_val[0]) if len(_w1_val) > 0 else 0.22
        fig.add_annotation(
            x=1, y=_w1,
            text=f"<b>{_w1:.0%} came back after first session</b>",
            showarrow=True, arrowhead=2, arrowcolor=COLORS['red'],
            font=dict(color=COLORS['red'], size=11),
            ax=40, ay=-30, bgcolor='white', bordercolor=COLORS['red'], borderwidth=1,
        )
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
    w0r = data['w0_requests_ret']
    _r1 = float(w0r[w0r['bucket']=='1']['retention_rate'].values[0])
    _r610 = float(w0r[w0r['bucket']=='6-10']['retention_rate'].values[0])
    seg_labels = ['MCP users', 'Non-MCP users', 'PayGo enabled', 'No PayGo', 'Non-streaming', 'Streaming', '1 request', '6-10 requests']
    seg_vals_dyn = list(data['seg_values_dynamic']) + [_r1, _r610]
    seg_colors = [COLORS['blue'], COLORS['gray'], COLORS['blue'], COLORS['gray'], COLORS['blue'], COLORS['gray'], COLORS['gray'], COLORS['blue']]
    fig = go.Figure(go.Bar(
        x=seg_labels, y=seg_vals_dyn, marker_color=seg_colors,
        text=[f"{v:.0%}" for v in seg_vals_dyn], textposition='outside',
        customdata=seg_labels, hovertemplate='%{customdata}: %{y:.0%}<extra></extra>',
    ))
    fig.add_vline(x=5.5, line_dash='dot', line_color='#e2e8f0', line_width=1)
    fig.add_annotation(x=6.5, y=0.62, text="Week-0 volume", showarrow=False,
                       font=dict(size=10, color='#94a3b8'))
    fig.update_layout(**PLOTLY_THEME, height=300, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(tickformat='.0%', range=[0, 0.68], gridcolor='#f1f5f9', title='Retention rate'),
                      xaxis=dict(showgrid=False, title='Segment'))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Findings & recommendation"):
        st.markdown("""
**Findings**

74% of users are one-time users. Average week-1 retention is 22%, and only 4.8% qualify as power users (active 4+ weeks) — yet that 4.8% generates 79% of all credits charged.

The strongest predictors of retention from first-week behavior:
- Plan type — enterprise retains 100%, bootstrap 76%, growth 65%, project 55%, but researcher (96% of all users) retains only 25%. The product is largely being tested by low-tier users who don't convert.
- PayGo enabled = 2x retention — 52% vs 24% without PayGo. These users have skin in the game and are more likely building real workflows.
- MCP integration = 1.7x retention — MCP users retain at 32% vs 19%. MCP users have already embedded the API into a tool, meaning they're builders, not explorers.
- More requests in week 0 = higher retention. Going from 1 request (21%) to 6-10 requests (35%) in the first week meaningfully lifts the chance of coming back.
- Streaming users churn more — 17% vs 28% for non-streaming. Streaming looks like a quick-test behavior, not a production integration pattern.

**Recommendation**

Users who integrate properly (MCP, PayGo, higher plans) retain well. The problem is most users never reach that point. Focus onboarding on driving users toward a real integration — push toward MCP setup, identify researcher-plan users with high week-0 volume as upgrade candidates. Those users are on the wrong plan.
        """)

# ═══════════════════════════════════════════════════════════
# TAB 2 — PROFITABILITY
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Q2: Is the Research API profitable? Are there cases of 'money on the floor'?")
    st.markdown('<div class="insight-box warning"><b>Hypothesis:</b> When requests fail or are cancelled mid-run, the system has already consumed compute, LLM calls, and search operations but likely charges nothing. This partial work represents unrecovered cost and a structural profitability leak. Finding the reasons for cancellations (long running times, lack of credits) or failures (technical issues) might help improve profitability.</div>', unsafe_allow_html=True)

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
        (k3, f"{fc_pct_cost:.2%}", "Failed & cancelled share of total cost", "negligible"),
        (k4, f"{data['credits_cancelled']:,.0f}", "Credits recovered on cancelled requests", "charges do happen here"),
    ]:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div><div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{note}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="insight-box success" style="margin-top:0.75rem;"><b>Hypothesis disproved.</b> Their % of total requests and share of total cost is not meaningful and impactful as I thought.</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── 3 boxes: c1 = cost recovery | c2 = uncharged requests | c3 = uncharged by status
    c1, c2, c3, c4 = st.columns(4)
    uncharged_count = data['uncharged_count']
    uncharged_cost_abs = data['uncharged_cost']
    sdf = data['status_stats_df']

    with c1:
        st.markdown(f'''<div class="kpi-card-alert">
            <div class="kpi-value">{uncharged_pct:.0%}</div>
            <div class="kpi-label" style="margin-top:6px;">Unrecovered system costs</div>
            <div style="font-size:0.82rem;color:#64748b;margin-top:3px;">(~${uncharged_cost_abs:,.0f})</div>
        </div>''', unsafe_allow_html=True)

    with c2:
        st.markdown(f'''<div class="kpi-card-alert">
            <div class="kpi-value">{data["recovery_rate_usd"]:.0%}</div>
            <div class="kpi-label" style="margin-top:6px;">Cost recovery rate</div>
            <div style="font-size:0.82rem;color:#64748b;margin-top:3px;">Est. revenue ${data["total_revenue_usd"]:,.0f} / total costs ${data["total_cost_rr"]:,.0f}</div>
            <div style="font-size:0.7rem;color:#94a3b8;margin-top:4px;">* assumes $0.008/credit (PayGo rate)</div>
        </div>''', unsafe_allow_html=True)

    with c3:
        total_uncharged_all = int(sdf['uncharged_req'].sum())
        split_rows = ""
        for _, r in sdf.iterrows():
            if r['uncharged_req'] <= 0:
                continue
            pct = r['uncharged_req'] / total_uncharged_all * 100
            split_rows += (
                f'<div style="display:flex;justify-content:space-between;font-size:11px;'
                f'color:#64748b;padding:3px 0;border-bottom:0.5px solid #f1f5f9;">'
                f'<span>{r["status"]}</span>'
                f'<span style="color:#0f172a;font-weight:500;">{int(r["uncharged_req"]):,}'
                f' <span style="color:#94a3b8;font-weight:400;">({pct:.0f}%)</span></span>'
                f'</div>'
            )
        st.markdown(f'''<div class="kpi-card-alert">
            <div class="kpi-label" style="margin-bottom:8px;">Uncharged requests by status</div>
            {split_rows}
            <div style="font-size:10px;color:#94a3b8;margin-top:6px;">total {total_uncharged_all:,}</div>
        </div>''', unsafe_allow_html=True)

    with c4:
        st.markdown(f'''<div class="kpi-card-alert">
            <div class="kpi-value">{uncharged_req_pct:.0%}</div>
            <div style="font-size:0.9rem;font-weight:500;color:#0f172a;margin-top:2px;">of successful requests uncharged</div>
            <div style="font-size:0.82rem;color:#64748b;margin-top:3px;">{uncharged_count:,} / {data["success_count"]:,} requests</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-header">Cost split: charged / uncharged</div>', unsafe_allow_html=True)
        _recovered = data['charged_cost']
        _uncharged_success = data['uncharged_cost']
        _fc = data['fc_cost']
        fig = go.Figure(go.Pie(
            labels=['Recovered', 'Uncharged (success)', 'Uncharged (failed/cancelled)'],
            values=[_recovered, _uncharged_success, _fc],
            hole=0.65, marker_colors=[COLORS['green'], COLORS['red'], COLORS['gray']],
            textinfo='percent',
            textposition='inside',
            hovertemplate='<b>%{label}</b><br>%{percent} of total<br>$%{value:,.0f}<extra></extra>',
        ))
        fig.update_layout(**PLOTLY_THEME, height=300, showlegend=True,
                          legend=dict(orientation='h', yanchor='top', y=-0.05, xanchor='center', x=0.5, font=dict(size=11)),
                          margin=dict(l=10, r=10, t=10, b=60),
                          annotations=[dict(text='serving<br>cost', x=0.5, y=0.5, font_size=11, showarrow=False, font_color='#888')])
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">% of successful requests charged by plan</div>', unsafe_allow_html=True)
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

    c1, c2, c3, c4 = st.columns(4)
    _p95 = data['p95_overall']
    _early_sr = data['q3_early_sr']
    _recent_sr = data['q3_recent_sr']
    _sr_arrow = '&#8593;' if _recent_sr > _early_sr else '&#8595;'
    _sr_color = '#1D9E75' if _recent_sr > _early_sr else '#E24B4A'

    _early_bad = data['q3_early_bad']
    _recent_bad = data['q3_recent_bad']
    _bad_arrow = '&#8593;' if _recent_bad > _early_bad else '&#8595;'
    _bad_color = '#E24B4A' if _recent_bad > _early_bad else '#1D9E75'


    # Box 1: success rate + arrow (last 3 vs first 3 weeks)
    _sr_arrow = '&#8593;' if _recent_sr > _early_sr else '&#8595;'
    _sr_color = '#1D9E75' if _recent_sr > _early_sr else '#E24B4A'
    with c1:
        st.markdown(f'''<div class="kpi-card">
            <div class="kpi-value">{data["current_sr_str"]}</div>
            <div class="kpi-label" style="margin-top:6px;">Current success rate</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">based on last 3 weeks</div>
            <div style="margin-top:8px;font-size:0.85rem;color:{_sr_color};font-weight:500;">
                {_sr_arrow} vs {_early_sr:.0%} in first 3 weeks
            </div>
        </div>''', unsafe_allow_html=True)

    # Box 2: p95 + arrow (last 3 vs first 3 weeks)
    _early_p95 = data['q3_early_p95']
    _recent_p95 = data['q3_recent_p95']
    _p95_arrow = '&#8595;' if _recent_p95 < _early_p95 else '&#8593;'
    _p95_color = '#1D9E75' if _recent_p95 < _early_p95 else '#E24B4A'
    with c2:
        st.markdown(f'''<div class="kpi-card">
            <div class="kpi-value">{_p95:.0f}s</div>
            <div class="kpi-label" style="margin-top:6px;">p95 response time</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">avg across all weeks · 1 in 20 requests &gt; {_p95/60:.0f} min</div>
            <div style="margin-top:8px;font-size:0.85rem;color:{_p95_color};font-weight:500;">
                {_p95_arrow} {_recent_p95:.0f}s (last 3 weeks) vs {_early_p95:.0f}s (first 3 weeks)
            </div>
        </div>''', unsafe_allow_html=True)

    # Box 3: p50 range — no arrow, volatility is the story
    with c3:
        st.markdown(f'''<div class="kpi-card">
            <div class="kpi-value">{data["p50_range_str"]}</div>
            <div class="kpi-label" style="margin-top:6px;">Weekly p50 range</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">volatile across {data["q3_n_weeks"]} weeks · no clear trend</div>
        </div>''', unsafe_allow_html=True)

    # Box 4: % of users failing + arrow + trend line
    _early_pct = data['q3_early_pct_bad']
    _recent_pct = data['q3_recent_pct_bad']
    _pct_arrow = '&#8595;' if _recent_pct < _early_pct else '&#8593;'
    _pct_color = '#1D9E75' if _recent_pct < _early_pct else '#E24B4A'
    _pct_trend = data['q3_pct_bad_trend']
    _pct_js = json.dumps([round(v, 1) for v in _pct_trend])
    with c4:
        st.markdown(f'''<div class="kpi-card">
            <div class="kpi-value">{_recent_pct:.1f}%</div>
            <div class="kpi-label" style="margin-top:6px;">% of users failing / week</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">last 3 weeks · failed + cancelled</div>
            <div style="margin-top:8px;font-size:0.85rem;color:{_pct_color};font-weight:500;">
                {_pct_arrow} vs {_early_pct:.1f}% in first 3 weeks
            </div>
            <div style="margin-top:8px;height:36px;">
                <canvas id="pcttrend" style="width:100%;height:36px;"></canvas>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
        <script>
        (function(){{
            var canvas = document.getElementById("pcttrend");
            if(!canvas) return;
            new Chart(canvas, {{
                type:"line",
                data:{{labels:{_pct_js}.map((_,i)=>i),datasets:[{{
                    data:{_pct_js},borderColor:"{_pct_color}",borderWidth:2,
                    pointRadius:0,tension:0.4,fill:false
                }}]}},
                options:{{responsive:false,maintainAspectRatio:false,
                    plugins:{{legend:{{display:false}},tooltip:{{enabled:false}}}},
                    scales:{{x:{{display:false}},y:{{display:false}}}}
                }}
            }});
        }})();
        </script>''', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Weekly success rate</div>', unsafe_allow_html=True)
    ws = data['weekly_status']
    fig = go.Figure(go.Scatter(
        x=ws['week_str'], y=ws['success'], mode='lines+markers',
        line=dict(color=COLORS['green'], width=2), fill='tozeroy',
        fillcolor='rgba(29,158,117,0.07)', marker=dict(size=4),
        hovertemplate='%{x}: %{y:.1%}<extra></extra>',
    ))
    fig.add_hline(y=0.99, line_dash='dot', line_color='#f59e0b',
                  annotation_text='99% · industry benchmark', annotation_position='bottom right')
    fig.update_layout(**PLOTLY_THEME, height=220, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(tickformat='.0%', range=[0.75, 1.02], gridcolor='#f1f5f9', title='Success rate'),
                      xaxis=dict(showgrid=False, tickangle=45, title='Week'))
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Weekly latency - p50 vs p95 (seconds)</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-header">Mini vs pro - latency profile</div>', unsafe_allow_html=True)
        ms = data['model_stats']
        mini = ms['mini']
        pro  = ms['pro']
        p50_mult  = pro['p50'] / mini['p50']
        p95_mult  = pro['p95'] / mini['p95']
        sr_diff   = (pro['sr'] - mini['sr']) * 100

        def badge(txt, good):
            bg  = '#E1F5EE' if good else '#FCEBEB'
            col = '#0F6E56' if good else '#A32D2D'
            return f'<span style="background:{bg};color:{col};font-size:11px;font-weight:500;padding:2px 8px;border-radius:4px;">{txt}</span>'

        th = 'padding:6px 8px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;'
        rows = (
            f'<tr style="border-bottom:0.5px solid #e2e8f0;">'
            f'<td style="padding:8px;font-size:12px;color:#64748b;">Success rate</td>'
            f'<td style="padding:8px;font-size:12px;text-align:right;font-weight:500;">{mini["sr"]:.1%}</td>'
            f'<td style="padding:8px;font-size:12px;text-align:right;font-weight:500;">{pro["sr"]:.1%}</td>'
            f'<td style="padding:8px;text-align:center;">{badge(f"{sr_diff:+.1f}pp", sr_diff > 0)}</td></tr>'
            f'<tr style="border-bottom:0.5px solid #e2e8f0;">'
            f'<td style="padding:8px;font-size:12px;color:#64748b;">p50 latency</td>'
            f'<td style="padding:8px;font-size:12px;text-align:right;font-weight:500;">{mini["p50"]:.0f}s</td>'
            f'<td style="padding:8px;font-size:12px;text-align:right;font-weight:500;">{pro["p50"]:.0f}s</td>'
            f'<td style="padding:8px;text-align:center;">{badge(f"{p50_mult:.1f}x slower", False)}</td></tr>'
            f'<tr style="border-bottom:0.5px solid #e2e8f0;">'
            f'<td style="padding:8px;font-size:12px;color:#64748b;">p95 latency</td>'
            f'<td style="padding:8px;font-size:12px;text-align:right;font-weight:500;">{mini["p95"]:.0f}s</td>'
            f'<td style="padding:8px;font-size:12px;text-align:right;font-weight:500;">{pro["p95"]:.0f}s</td>'
            f'<td style="padding:8px;text-align:center;">{badge(f"{p95_mult:.1f}x slower", False)}</td></tr>'
            
        )
        table_html = (
            '<table style="width:100%;border-collapse:collapse;margin-top:8px;">'
            '<thead><tr style="border-bottom:1px solid #e2e8f0;">'
            f'<th style="{th}color:#64748b;text-align:left;">Metric</th>'
            f'<th style="{th}color:#185FA5;text-align:right;">Mini</th>'
            f'<th style="{th}color:#534AB7;text-align:right;">Pro</th>'
            f'<th style="{th}color:#64748b;text-align:center;">Pro vs mini</th>'
            '</tr></thead>'
            f'<tbody>{rows}</tbody></table>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pipeline depth: top 5% vs typical 95% of requests</div>', unsafe_allow_html=True)
    pc = data['pipeline_comparison']
    metrics = ['Avg LLM calls', 'Avg response time (s)', 'Avg cost ($)']
    top5_vals = [pc['top5_llm'], pc['top5_time'], pc['top5_cost']]
    typ_vals  = [pc['typ_llm'],  pc['typ_time'],  pc['typ_cost']]

    fig_pc = go.Figure()
    fig_pc.add_trace(go.Bar(
        name=f"Top 5% (≥{pc['llm_thresh']:.0f} LLM calls)",
        y=metrics, x=top5_vals, orientation='h',
        marker_color=COLORS['red'],
        text=[f"{v:.0f}" for v in top5_vals], textposition='outside',
        hovertemplate='%{y}<br>Top 5%: %{x:.0f}<extra></extra>',
    ))
    fig_pc.add_trace(go.Bar(
        name='Typical 95%',
        y=metrics, x=typ_vals, orientation='h',
        marker_color=COLORS['blue'],
        text=[f"{v:.0f}" for v in typ_vals], textposition='outside',
        hovertemplate='%{y}<br>Typical: %{x:.0f}<extra></extra>',
    ))
    fig_pc.update_layout(
        **PLOTLY_THEME, height=260, barmode='group',
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title='Value'),
        yaxis=dict(showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    st.plotly_chart(fig_pc, use_container_width=True)

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
</style>
<div style="font-size:1.1rem;font-weight:600;color:#0f172a;text-transform:uppercase;letter-spacing:0.1em;padding-bottom:0.5rem;border-bottom:1px solid #e2e8f0;margin-bottom:0.8rem;">Cost share by component</div>
<div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
  <span style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;">View:</span>
  <button class="tbtn active" id="btn-all" onclick="setFilter('all')">All</button>
  <button class="tbtn" id="btn-infra" onclick="setFilter('infra')">Infrastructure</button>
  <button class="tbtn" id="btn-models" onclick="setFilter('models')">Models</button>
</div>
<div id="chartWrap" style="position:relative;width:100%;">
  <canvas id="barChart"></canvas>
</div>
<script>
  const TOTAL = {grand_total};
  const infraData = {infra_js};
  const modelData = {model_js};
  const allData   = {all_js};
  let currentFilter = 'all';

  function getData(f) {{
    if (f==="infra")  return [...infraData].sort((a,b)=>a.value-b.value);
    if (f==="models") return [...modelData].sort((a,b)=>a.value-b.value);
    return allData;
  }}

  const labelPlugin = {{
    id: 'inlineLabels',
    afterDatasetsDraw(chart) {{
      const ctx2 = chart.ctx;
      const meta = chart.getDatasetMeta(0);
      ctx2.save();
      ctx2.font = '10px sans-serif';
      ctx2.fillStyle = '#64748b';
      ctx2.textAlign = 'left';
      ctx2.textBaseline = 'middle';
      meta.data.forEach(function(bar, i) {{
        const val = chart.data.datasets[0].data[i];
        if (!val) return;
        const pct = (val/TOTAL*100).toFixed(1) + '%';
        ctx2.fillText(pct, bar.x + 6, bar.y);
      }});
      ctx2.restore();
    }}
  }};

  function setFilter(f) {{
    currentFilter = f;
    ["all","infra","models"].forEach(x=>document.getElementById("btn-"+x).classList.toggle("active",x===f));
    const bar = getData(f);
    const subtotal = bar.reduce((s,d)=>s+d.value,0);
    resizeChart(bar.length);
    barChart.data.labels = bar.map(d=>d.label);
    barChart.data.datasets[0].data = bar.map(d=>d.value);
    barChart.data.datasets[0].backgroundColor = bar.map(d=>d.color);
    barChart.options.plugins.tooltip.callbacks.label = function(ctx) {{
      const val = ctx.raw;
      return ' $' + val.toLocaleString('en-US', {{maximumFractionDigits:0}});
    }};
    barChart.update();
  }}

  function resizeChart(n) {{
    const h = Math.max(120, n * 22 + 60);
    document.getElementById('chartWrap').style.height = h + 'px';
  }}

  const ib = getData('all');
  resizeChart(ib.length);
  const gc="rgba(0,0,0,0.07)", tc="#0f172a";

  const barChart = new Chart(document.getElementById("barChart"), {{
    type: "bar",
    plugins: [labelPlugin],
    data: {{
      labels: ib.map(d=>d.label),
      datasets: [{{
        data: ib.map(d=>d.value),
        backgroundColor: ib.map(d=>d.color),
        borderRadius: 3
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      layout: {{ padding: {{ right: 80 }} }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              const val = ctx.raw;
              const dollar = '$' + val.toLocaleString('en-US', {{maximumFractionDigits:0}});
              return ` ${{dollar}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{ color: tc, font: {{size:10}}, callback: v => (v/TOTAL*100).toFixed(0) + '%' }},
          grid: {{ color: gc }},
          title: {{ display: true, text: '% of total cost', color: '#94a3b8', font: {{size:11}} }}
        }},
        y: {{
          ticks: {{ color: tc, font: {{size:10}} }},
          grid: {{ display: false }},
          title: {{ display: true, text: 'Component', color: '#94a3b8', font: {{size:11}} }}
        }}
      }}
    }}
  }});
</script>
"""
    st.components.v1.html(html_code, height=560)

    st.markdown('<div class="section-header">Fixed vs Variable Cost Components</div>', unsafe_allow_html=True)

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

    weeks_js = json.dumps([str(w.date()) for w in weekly_merged['week']])
    requests_js = json.dumps(weekly_merged['requests'].tolist())

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

    ts_components_js = json.dumps(ts_components)
    scatter_below = [{'label':r['label'],'corr':round(r['corr'],3),'total':round(r['total'],0)} for _,r in corr_df[corr_df['fixed']].iterrows()]
    scatter_above = [{'label':r['label'],'corr':round(r['corr'],3),'total':round(r['total'],0)} for _,r in corr_df[~corr_df['fixed']].iterrows()]
    scatter_below_js = json.dumps(scatter_below)
    scatter_above_js = json.dumps(scatter_above)
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
  <p class="slabel">Step 1 - Observed: cost components vs research request volume over time</p>
  <p class="sdesc">Select one component to compare against research request volume. Each component uses its own Y-axis scale.</p>
  <div id="toggles"></div>
  <div class="legend-row">
    <span id="compLegend" style="display:flex;align-items:center;gap:4px;"></span>
    <span style="display:flex;align-items:center;gap:4px;margin-left:12px;">
      <span style="width:16px;height:2px;border-top:2px dashed #4ade80;display:inline-block;"></span>
      Research requests
    </span>
  </div>
  <div style="position:relative;width:100%;height:260px;"><canvas id="timeChart"></canvas></div>
</div>
<div class="card">
  <p class="slabel">Step 2 - Validated: correlation with research request volume across all components</p>
  <p class="sdesc">A threshold of 0.3 was chosen based on a natural gap in the data - all components cluster either below 0.22 or above 0.36, with nothing in between.</p>
  <div class="legend-row">
    <span style="display:flex;align-items:center;gap:4px;"><span class="dot" style="background:#534AB7;"></span>Below threshold</span>
    <span style="display:flex;align-items:center;gap:4px;"><span class="dot" style="background:#f59e0b;"></span>Above threshold</span>
  </div>
  <div style="position:relative;width:100%;height:300px;"><canvas id="scatterChart"></canvas></div>
</div>
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;">
  <p class="slabel">Step 3 - Conclusion: fixed vs variable split</p>
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
  let selected = components.find(c=>c.preset) || components[0];

  function buildDatasets() {{
    return [
      {{ label:selected.label, data:selected.data, borderColor:selected.color,
        backgroundColor:'transparent', borderWidth:2, pointRadius:2, yAxisID:'y' }},
      {{ label:'Research requests', data:requestsData, borderColor:'#4ade80',
        backgroundColor:'rgba(74,222,128,0.06)', borderWidth:1.5, borderDash:[4,4],
        pointRadius:2, fill:true, yAxisID:'y2' }}
    ];
  }}

  function getRange(data) {{
    const vals = data.filter(v=>v!=null);
    const mn = Math.min(...vals);
    const mx = Math.max(...vals);
    const pad = (mx - mn) * 0.1 || 1;
    return {{ min: mn - pad, max: mx + pad }};
  }}

  function applyRange() {{
    const r = getRange(selected.data);
    chart.options.scales.y.min = r.min;
    chart.options.scales.y.max = r.max;
  }}

  function updateLegend() {{
    const leg = document.getElementById('compLegend');
    leg.innerHTML = `<span style="width:16px;height:2px;background:${{selected.color}};display:inline-block;border-radius:2px;"></span> ${{selected.label}}`;
  }}

  function buildToggles() {{
    const cont = document.getElementById('toggles');
    cont.innerHTML = '';
    components.forEach(c=>{{
      const on = c.label === selected.label;
      const btn = document.createElement('button');
      btn.className = 'tbtn';
      btn.textContent = c.label;
      btn.style.border = `1px solid ${{c.color}}`;
      btn.style.fontWeight = '500';
      btn.style.background = on ? c.color : 'transparent';
      btn.style.color = on ? '#ffffff' : c.color;
      btn.onclick = () => {{
        selected = c;
        chart.data.datasets = buildDatasets();
        chart.options.scales.y.ticks.color = c.color;
        chart.options.scales.y.title.text = c.label + ' ($/hr)';
        chart.options.scales.y.title.color = c.color;
        applyRange();
        chart.update();
        buildToggles();
        updateLegend();
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
        x:{{ ticks:{{ color:tc, font:{{size:10}}, maxRotation:45 }}, grid:{{ color:gc }},
             title:{{ display:true, text:'Week', color:tc, font:{{size:10}} }} }},
        y:{{ ticks:{{ color:selected.color, font:{{size:10}}, callback:v=>'$'+v.toFixed(0) }},
             grid:{{ color:gc }},
             title:{{ display:true, text:selected.label+' ($/hr)', color:selected.color, font:{{size:10}} }} }},
        y2:{{ position:'right', ticks:{{ color:'#4ade80', font:{{size:10}} }}, grid:{{ display:false }},
              title:{{ display:true, text:'Requests/hr', color:'#4ade80', font:{{size:10}} }} }}
      }}
    }}
  }});
  applyRange();
  chart.update();
  buildToggles();
  updateLegend();
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
          title:{{ display:true, text:'Correlation with research request volume', color:tc, font:{{size:10}} }},
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

    st.markdown('<div class="section-header">Cost Spikes - Total Cost vs Research Request Volume</div>', unsafe_allow_html=True)

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

    daily_js = json.dumps({
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
    hourly_js = json.dumps(hourly_js_dict)

    _spike_counts = {}
    _months_label = {'2025-12':'Dec 2025','2026-01':'Jan 2026','2026-02':'Feb 2026','2026-03':'Mar 2026'}
    for _m in ['2025-12','2026-01','2026-02','2026-03']:
        _sub = merged_spike[merged_spike['month'] == _m].copy()
        _mean_h = float(_sub['total_cost'].mean())
        _std_h = float(_sub['total_cost'].std())
        _thresh_h = _mean_h + 2*_std_h
        _spike_counts[_m] = int((_sub['total_cost'] > _thresh_h).sum())

    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    for _col, (_month, _count) in zip([_sc1, _sc2, _sc3, _sc4], _spike_counts.items()):
        with _col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{_count}</div><div class="kpi-label">Hourly spikes</div><div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{_months_label[_month]}</div></div>', unsafe_allow_html=True)
    st.markdown("")

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
  <span style="display:flex;align-items:center;gap:4px;"><span style="width:8px;height:8px;border-radius:50%;background:#f87171;display:inline-block;"></span>Cost spike (&gt;mean+2sigma)</span>
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
      ctx2.fillText('mean+2sigma', xs.right-70, yPx-4); ctx2.restore();
    }};
  }}
  function getVdata(view) {{
    if(view==='daily') return {{ ...DAILY, labels:DAILY.labels, costs:DAILY.costs, reqs:DAILY.reqs, costUnit:'$/day', reqUnit:'req/day', granularity:'Daily (>=50 req/day)' }};
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

    st.markdown('<div class="section-header">Cost Efficiency - Cost per Research Request Over Time (Weekly)</div>', unsafe_allow_html=True)

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
