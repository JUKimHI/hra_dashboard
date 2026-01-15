# -*- coding: utf-8 -*-
"""
Marine Biodiversity Risk Dashboard (Single-page integrated version)

Sections:
1) Top: Risk map + explanation
2) Top-1 stressor for High-risk regions in selected month
3) Home-like summary: KPI + monthly risk distribution
4) Data explorer: Integrated Data / risk_total / risk_stress
"""

from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st

# -------------------------------------------------
# Basic settings
# -------------------------------------------------
st.set_page_config(
    page_title="해양 환경 예측 기반 해양 생물 다양성 리스크 대시보드",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)
alt.data_transformers.disable_max_rows()

# -------------------------------------------------
# Common paths / loader utils
# -------------------------------------------------
DATA_FILES = {
    "rreal": "rreal_final_ALL_predicted.csv",
    "label_total": "hra_label_total_2025_2028.csv",
    "pairwise": "hra_pairwise_2025_2028.csv",
}

def find_data_path(filename: str):
    """Search CSV file from several possible locations."""
    cands = [
        Path.cwd() / "data" / filename,
        Path(__file__).parent / "data" / filename,
        Path.cwd() / filename,
        Path(__file__).parent / filename,
        Path("/mnt/data") / filename,
    ]
    for p in cands:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def load_csv_auto(name_key: str) -> pd.DataFrame:
    """Load CSV with auto-encoding trials."""
    fn = DATA_FILES[name_key]
    p = find_data_path(fn)
    if p is None:
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {fn}")
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(p)

# -------------------------------------------------
# Schema fix utils (region / year_month)
# -------------------------------------------------
def _ensure_region(df: pd.DataFrame):
    cand = [c for c in df.columns if str(c).lower() in ("region", "지역")]
    if cand:
        df = df.rename(columns={cand[0]: "region"})
    return df

def _coerce_year_month_series(s: pd.Series) -> pd.Series:
    """
    Convert various year-month formats to 'month-start Timestamp'.
    Examples: '2025-01', '202501', '2025/01' ...
    """
    s2 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    try:
        if s2.isna().mean() > 0.9:
            ss = s.astype(str).str.replace(r"[^0-9]", "", regex=True)
            mask6 = ss.str.len() == 6
            ss.loc[mask6] = ss[mask6] + "01"
            s2 = pd.to_datetime(ss, errors="coerce", format="%Y%m%d")
    except Exception:
        pass
    return s2.dt.to_period("M").dt.to_timestamp(how="start")

def _ensure_year_month(df: pd.DataFrame):
    """Create year_month if missing from similar columns."""
    lower = {str(c).lower(): c for c in df.columns}
    for key in ("year_month", "ym", "date", "month", "dt", "yearmonth", "날짜", "월"):
        if key in lower:
            col = lower[key]
            s = _coerce_year_month_series(df[col])
            if s.notna().any():
                df = df.copy()
                df["year_month"] = s
                return df
    y_key = next((lower[k] for k in ("year", "yr", "연", "년도", "연도") if k in lower), None)
    m_key = next((lower[k] for k in ("month", "mo", "mn", "월") if k in lower), None)
    if y_key is not None and m_key is not None:
        try:
            df = df.copy()
            df["year_month"] = _coerce_year_month_series(
                df[y_key].astype(int).astype(str)
                + "-" + df[m_key].astype(int).astype(str)
                + "-01"
            )
            return df
        except Exception:
            pass
    return df

def ensure_month_start_datetime(df: pd.DataFrame):
    if "year_month" in df.columns:
        df = df.copy()
        df["year_month"] = _coerce_year_month_series(df["year_month"])
    return df

def soft_schema_fix(df: pd.DataFrame):
    """Common fix for region and year_month."""
    return ensure_month_start_datetime(_ensure_year_month(_ensure_region(df)))

# -------------------------------------------------
# Data explorer utils (from 01_Data.py)
# -------------------------------------------------
def _numeric_columns(df: pd.DataFrame):
    drop_like = {"region", "year_month", "label", "class", "category"}
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not any(k in str(c).lower() for k in drop_like)
    ]

def _min_max_dt(df: pd.DataFrame):
    if "year_month" not in df.columns:
        return (None, None)
    try:
        s = pd.to_datetime(df["year_month"])
        return (s.min(), s.max())
    except Exception:
        return (None, None)

def _alt_line_chart(df, x_col, y_col, color_col="region", title=None):
    ch = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(x_col, title=None),
            y=alt.Y(y_col, title=y_col),
            color=alt.Color(color_col, legend=alt.Legend(title="지역")),
            tooltip=[color_col, x_col, alt.Tooltip(y_col, format=",.3f")],
        )
        .properties(height=420)
    )
    return ch.properties(title=title) if title else ch

def _to_py_datetime(x):
    return x.to_pydatetime() if isinstance(x, pd.Timestamp) else x

def _to_pd_timestamp(x):
    return pd.Timestamp(x).to_period("M").to_timestamp(how="start")

# -------------------------------------------------
# Load data
# -------------------------------------------------
load_ok, errors = True, []

try:
    df_rreal = soft_schema_fix(load_csv_auto("rreal"))
except Exception as e:
    df_rreal = pd.DataFrame()
    load_ok = False
    errors.append(("rreal_final_ALL_predicted.csv", str(e)))

try:
    df_label_total = soft_schema_fix(load_csv_auto("label_total"))
except Exception as e:
    df_label_total = pd.DataFrame()
    load_ok = False
    errors.append(("hra_label_total_2025_2028.csv", str(e)))

try:
    df_pairwise = soft_schema_fix(load_csv_auto("pairwise"))
except Exception as e:
    df_pairwise = pd.DataFrame()
    load_ok = False
    errors.append(("hra_pairwise_2025_2028.csv", str(e)))

# -------------------------------------------------
# Title + global description (Korean kept)
# -------------------------------------------------
st.title("해양 환경 예측 기반 해양 생물 다양성 리스크 대시보드")

st.markdown(
    """
이 대시보드는 **인천 · 거제 · 울릉도** 세 지역을 대상으로,  
수온(SST), 염분, 용존산소(O₂), pH, 엘니뇨/라니냐 지수(ENSO) 등 해양 환경 예측과  
해양 생물 종 수 예측을 결합하여 **미래(2025–2028년) 해양 생물 다양성 위험도**를 보여주는 도구입니다.  

대시보드는 크게 세 부분으로 구성됩니다.

1. **Risk map (월별 누적 위험도 지도)**  
   - 월별·지역별 누적 위험도(**R_sum**)를 지도 위 원의 색(위험 등급)과 크기(강도)로 표현합니다.  
   - 색상: *녹색=Low, 노란색=Medium, 빨간색=High*  
   - 선택한 연·월에 대해 **어느 지역이 더 위험한지 한 눈에** 볼 수 있습니다.

2. **High 지역의 주요 스트레스 요인**  
   - 선택한 연·월에 High(고위험)로 분류된 지역이 있다면,  
     각 지역의 누적 위험도에 **가장 크게 기여한 스트레스 요인(stressor)** 을 Top-1로 보여줍니다.  

3. **세부 요약·탐색**  
   - (요약 섹션) 선택 월 기준의 위험도 분포와 지역별 위험도 스택 막대를 통해 **전반적인 패턴**을 확인합니다.  
   - (데이터 탐색 섹션) 원자료(통합 데이터, 위험도 라벨, 스트레스 pairwise 데이터)를  
     시계열/연도/월 패턴 관점에서 **직접 필터링하면서 탐색**할 수 있습니다.

> 위험도 산정은 InVEST HRA(Habitat Risk Assessment) 개념을 참고하여,  
> 스트레스 요인별 **노출(E)**, **영향(C)**를 점수화하고,  
> 유클리드 거리 기반으로 위험도 **R**을 계산한 뒤,  
> 각 월·지역에 대해 스트레스별 R을 합산한 **R_sum** 기준으로 Low/Medium/High로 나눈 것입니다.
"""
)

if not load_ok:
    st.error("일부 데이터 로드에 실패했습니다. 아래 오류 내용을 확인해 주세요.")
    with st.expander("데이터 로드 오류 상세 보기", expanded=False):
        for fname, msg in errors:
            st.markdown(f"- **{fname}**")
            st.code(msg)
    st.stop()

if df_label_total.empty:
    st.error("위험도 라벨 데이터(hra_label_total_2025_2028.csv)가 비어 있어 대시보드를 구성할 수 없습니다.")
    st.stop()

# -------------------------------------------------
# Common: risk level normalization + colors
# -------------------------------------------------
if "risk_level" in df_label_total.columns:
    df_label_total["risk_level"] = (
        df_label_total["risk_level"].astype(str).str.strip().str.title()
    )
elif "label_sum" in df_label_total.columns:
    df_label_total["risk_level"] = (
        df_label_total["label_sum"]
        .map({1: "Low", 2: "Medium", 3: "High"})
        .fillna("Medium")
    )
else:
    df_label_total["risk_level"] = "Medium"

if "R_sum" not in df_label_total.columns:
    df_label_total["R_sum"] = 1.0

lat_col = next((c for c in df_label_total.columns if str(c).lower() in ("lat", "latitude", "위도")), None)
lon_col = next((c for c in df_label_total.columns if str(c).lower() in ("lon", "lng", "longitude", "경도")), None)

REGION_COORDS = {
    "Incheon": (37.456, 126.705),
    "Geoje": (34.880, 128.620),
    "Ulleungdo": (37.500, 130.900),
    "인천": (37.456, 126.705),
    "거제": (34.880, 128.620),
    "울릉": (37.500, 130.900),
    "울릉도": (37.500, 130.900),
    "울릉군": (37.500, 130.900),
}

def add_coords(_df: pd.DataFrame) -> pd.DataFrame:
    _df = _df.copy()
    if lat_col and lon_col:
        _df["lat"] = _df[lat_col]
        _df["lon"] = _df[lon_col]
        return _df
    _df["lat"] = _df["region"].map(lambda r: REGION_COORDS.get(str(r), (np.nan, np.nan))[0])
    _df["lon"] = _df["region"].map(lambda r: REGION_COORDS.get(str(r), (np.nan, np.nan))[1])
    return _df

df_map = add_coords(df_label_total).dropna(subset=["region", "year_month", "lat", "lon"]).copy()

CATS = ["Low", "Medium", "High"]
COLOR_MAP = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
df_map["risk_level"] = pd.Categorical(df_map["risk_level"], categories=CATS, ordered=True)
df_map["ym_str"] = df_map["year_month"].dt.strftime("%Y_%m")

# Pairwise schema unify: stressor / R
cols_lower = {c.lower(): c for c in df_pairwise.columns}
stressor_col = next((c for c in df_pairwise.columns if "stress" in c.lower()), None)
r_col = next((cols_lower[k] for k in ("r", "risk") if k in cols_lower), None)

if stressor_col:
    df_pairwise = df_pairwise.rename(columns={stressor_col: "stressor"})
if r_col:
    df_pairwise = df_pairwise.rename(columns={r_col: "R"})

# -------------------------------------------------
# 1. Risk map (ENGLISH — screenshot part)
# -------------------------------------------------
st.header("1. Monthly Cumulative Risk Map (Risk map)")

st.markdown(
    """
**How to read this map**

- Each **circle represents one region-month**.
- **Color**: risk level  
  - 🟢 **Low** – relatively stable condition so far  
  - 🟡 **Medium** – one or more stressors are affecting the system  
  - 🔴 **High** – multiple stressors (SST, salinity, low O₂, pH, ENSO, etc.) are strongly acting **at the same time**
- **Size**: **monthly cumulative risk (R_sum)**  
  Larger **R_sum** → larger circle.

Use the filters to select **year / month / regions**.  
The map and the description below will update accordingly.
"""
)

# ---- Filters: regions / year / month ----
all_regions = sorted(df_map["region"].dropna().unique().tolist())
preferred_regions = [
    r
    for r in all_regions
    if str(r) in ["Incheon", "인천", "Geoje", "거제", "Ulleungdo", "울릉", "울릉도", "울릉군"]
]
default_regions_map = preferred_regions if preferred_regions else all_regions[:3]

colA, colB, colC = st.columns([2, 1, 1])
sel_regions_map = colA.multiselect(
    "Regions to display",
    all_regions,
    default=default_regions_map,
    help="Select only the regions you want to show on the map.",
)
size_scale = colB.slider(
    "Bubble size scale",
    min_value=5,
    max_value=40,
    value=20,
    help="Controls how large the bubbles become as R_sum increases.",
)

years_map = sorted(df_map["year_month"].dt.year.unique().tolist())
default_year = max(years_map) if years_map else 2025
sel_year = colC.selectbox(
    "Select year",
    years_map,
    index=years_map.index(default_year) if years_map else 0,
)

# Available months within the selected year
months_avail = sorted(
    df_map.loc[df_map["year_month"].dt.year.eq(sel_year), "year_month"]
    .dt.month.unique()
    .tolist()
)
m_labels = [f"{m:02d}" for m in months_avail]
default_mm = f"{max(months_avail):02d}" if months_avail else "01"
sel_month = int(
    st.selectbox(
        "Select month",
        m_labels,
        index=m_labels.index(default_mm) if m_labels else 0,
        help="Choose a month available for the selected year.",
    )
)

sel_ts = pd.Timestamp(f"{sel_year}-{sel_month:02d}-01")
st.caption(f"Current month: **{sel_year}_{sel_month:02d}**")

# ---- Filter data for the map ----
df_v = df_map.copy()
if sel_regions_map:
    df_v = df_v[df_v["region"].isin(sel_regions_map)]

df_m = df_v[
    (df_v["year_month"].dt.year == sel_year)
    & (df_v["year_month"].dt.month == sel_month)
].copy()

center_korea = {"lat": 36.2, "lon": 128.0}

def base_hover_cols(_df: pd.DataFrame):
    cols = {"risk_level": True, "ym_str": True}
    if "R_sum" in _df.columns:
        cols["R_sum"] = ":.3f"
    return cols

if df_m.empty:
    st.info("No data found for the selected year / month / region filters.")
else:
    fig = px.scatter_mapbox(
        df_m,
        lat="lat",
        lon="lon",
        color="risk_level",
        color_discrete_map=COLOR_MAP,
        category_orders={"risk_level": CATS},
        size="R_sum",
        size_max=size_scale,
        hover_name="region",
        hover_data=base_hover_cols(df_m),
        zoom=5.3,
        center=center_korea,
        height=560,
    )

    # Always show all risk levels in legend
    present = {tr.name for tr in fig.data}
    for lvl in CATS:
        if lvl not in present:
            fig.add_scattermapbox(
                lat=[None],
                lon=[None],
                mode="markers",
                marker=dict(size=10, color=COLOR_MAP[lvl]),
                name=lvl,
                showlegend=True,
            )

    _order = {name: i for i, name in enumerate(CATS)}
    fig.data = tuple(sorted(fig.data, key=lambda tr: _order.get(tr.name, 99)))

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="Risk level",
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 1-2. Top-1 stressor for High-risk regions (Korean kept)
# -------------------------------------------------
st.subheader("1-2. 선택 월 High 지역의 주요 스트레스 요인(Top-1)")

st.markdown(
    """
여기서는 **위의 Risk map에서 선택한 연·월**을 기준으로,  
High(고위험) 등급으로 분류된 지역이 있다면 각 지역별로  
**누적 위험도에 가장 크게 기여한 스트레스 요인**을 1개씩 보여줍니다.

- 예를 들어, 인천이 High라면 `엘니뇨`, `수온(SST)` 중 어느 요인이 더 위험도에 크게 기여했는지 확인할 수 있습니다.  
- 값 **R_mean**은 해당 지역·시점에서 그 스트레스 요인의 평균 위험도(R)를 의미합니다.
"""
)

high_regions = df_m.loc[df_m["risk_level"] == "High", "region"].dropna().unique().tolist()

if not high_regions:
    st.info("선택한 연·월에는 High(고위험)로 분류된 지역이 없습니다.")
else:
    need_cols = {"year_month", "region", "stressor", "R"}
    if not need_cols.issubset(set(map(str, df_pairwise.columns))):
        st.warning("pairwise 데이터에 필요한 컬럼(year_month, region, stressor, R)이 부족합니다.")
    else:
        dfx = df_pairwise[
            (df_pairwise["year_month"] == sel_ts)
            & (df_pairwise["region"].isin(high_regions))
        ].copy()
        if dfx.empty:
            st.info("선택한 연·월의 High 지역에 대해 pairwise 레코드가 없습니다.")
        else:
            g = (
                dfx.groupby(["region", "stressor"], as_index=False)["R"]
                .mean()
                .rename(columns={"R": "R_mean"})
            )
            top1 = (
                g.sort_values(["region", "R_mean"], ascending=[True, False])
                .groupby("region", as_index=False)
                .head(1)
            )
            top1["R_mean"] = top1["R_mean"].round(3)
            st.dataframe(
                top1.rename(columns={"region": "지역", "stressor": "최대 R 요인", "R_mean": "R값(평균)"}),
                use_container_width=True,
            )

st.divider()

# -------------------------------------------------
# 2. Home-like summary (Korean kept)
# -------------------------------------------------
st.header("2. 선택 월 위험도 요약")

st.markdown(
    """
이 섹션에서는 **전체 기간 관점의 기본 통계**와  
**위에서 선택한 연·월 기준의 위험도 분포**를 함께 보여줍니다.

1. 상단 **지표 카드(KPI)**  
   - 분석에 사용된 지역 수, 전체 레코드 수, 데이터가 커버하는 기간(시작~종료)을 요약해줍니다.

2. 하단 두 개의 막대 그래프  
   - **이번 달 위험도 분포**: Low/Medium/High가 전체에서 각각 몇 건인지  
   - **지역별 위험도 분포(스택)**: 각 지역별로 Low/Medium/High가 어떻게 섞여 있는지  
"""
)

regions_all = sorted(df_label_total["region"].dropna().unique().tolist())
first_dt = df_label_total["year_month"].min()
last_dt = df_label_total["year_month"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📍 지역 수", f"{len(regions_all):,}")
c2.metric("🧾 전체 레코드", f"{len(df_label_total):,}")
c3.metric("⏱️ 기간 시작", first_dt.strftime("%Y-%m") if pd.notna(first_dt) else "-")
c4.metric("⏱️ 기간 종료", last_dt.strftime("%Y-%m") if pd.notna(last_dt) else "-")

st.caption(f"아래 그래프는 **{sel_year}_{sel_month:02d}** 기준으로 집계되었습니다.")

ORDER = ["Low", "Medium", "High"]
COLOR = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}

df_label_total["risk_name"] = df_label_total["risk_level"].astype(str).str.strip().str.title()
df_label_total["risk_name"] = pd.Categorical(df_label_total["risk_name"], categories=ORDER, ordered=True)

df_m2 = df_label_total[df_label_total["year_month"].eq(sel_ts)].copy()

if df_m2.empty:
    st.info("선택한 연·월에 해당하는 위험도 라벨 데이터가 없습니다.")
else:
    lcol, rcol = st.columns([1.1, 1.3])

    with lcol:
        st.subheader("이번 달 위험도 등급 분포 (전체)")
        dist = (
            df_m2["risk_name"]
            .value_counts()
            .reindex(ORDER)
            .fillna(0)
            .rename_axis("risk_name")
            .reset_index(name="count")
        )
        chart = (
            alt.Chart(dist)
            .mark_bar()
            .encode(
                x=alt.X("risk_name:N", title="risk level", sort=ORDER),
                y=alt.Y("count:Q", title="건수"),
                color=alt.Color(
                    "risk_name:N",
                    scale=alt.Scale(domain=ORDER, range=[COLOR[k] for k in ORDER]),
                    legend=alt.Legend(title="risk level"),
                ),
                tooltip=[
                    alt.Tooltip("risk_name:N", title="risk level"),
                    alt.Tooltip("count:Q", title="건수", format=","),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    with rcol:
        st.subheader("지역별 위험도 분포 (스택 막대)")
        rc = (
            df_m2.groupby(["region", "risk_name"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        stacked = (
            alt.Chart(rc)
            .mark_bar()
            .encode(
                x=alt.X("region:N", title="지역"),
                y=alt.Y("count:Q", stack="zero", title="건수"),
                color=alt.Color(
                    "risk_name:N",
                    scale=alt.Scale(domain=ORDER, range=[COLOR[k] for k in ORDER]),
                    legend=alt.Legend(title="risk level"),
                ),
                tooltip=[
                    alt.Tooltip("region:N", title="지역"),
                    alt.Tooltip("risk_name:N", title="risk level"),
                    alt.Tooltip("count:Q", title="건수", format=","),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(stacked, use_container_width=True)

st.divider()

# -------------------------------------------------
# 3. Data explorer (Korean kept)
# -------------------------------------------------
st.header("3. 원자료 탐색 (Integrated Data / risk_total / risk_stress)")

st.markdown(
    """
이 섹션에서는 **모델에 사용된 원자료**를 직접 보면서,  
변수별 시계열 패턴·연도별 경향·월 패턴(계절성)과  
위험도 라벨(risk_total), 스트레스 pairwise(risk_stress)를 함께 탐색할 수 있습니다.

- 첫 번째 탭 **Integrated Data** : SST, 염분, O₂, pH, ENSO, 종 수 등 통합 데이터를 시계열/연도/월 패턴으로 확인  
- 두 번째 탭 **risk_total** : 월·지역별 위험도 등급 분포  
- 세 번째 탭 **risk_stress** : 스트레스 요인별 위험도(R)의 크기와 지역×스트레스 히트맵, 각 지역 Top-3 스트레스 요인
"""
)

tab_data, tab_total, tab_stress = st.tabs(
    ["🔗 Integrated Data", "⚠️ risk_total", "⚠️ risk_stress"]
)

# ===== Integrated Data =====
with tab_data:
    st.subheader("Integrated Data (rreal_final_ALL_predicted.csv)")
    with st.expander("원자료 미리보기", expanded=False):
        n = st.slider("표시 행 수", 5, 50, 10, key="n_rreal_preview")
        st.dataframe(df_rreal.head(n), use_container_width=True)

    t1, t2, t3 = st.tabs(["📅 월별 시계열", "📆 연도별 집계", "🗓 월 패턴(계절성)"])

    numeric_cols = _numeric_columns(df_rreal)
    regions_rreal = sorted(df_rreal["region"].dropna().unique()) if "region" in df_rreal.columns else []
    default_regions_rreal = regions_rreal[:5] if len(regions_rreal) > 5 else regions_rreal

    # --- t1: monthly time series ---
    with t1:
        cA, cB, cC = st.columns([2, 2, 2])
        var = cA.selectbox("변수 선택", numeric_cols, index=0 if numeric_cols else None, key="var_m")
        sel_regions = cB.multiselect("지역 선택", regions_rreal, default=default_regions_rreal, key="regions_m")
        agg = cC.selectbox("집계 방식", ["mean", "sum", "median", "first", "last"], index=0, key="agg_m")

        dt_min, dt_max = _min_max_dt(df_rreal)
        dr = None
        if dt_min is not None and dt_max is not None:
            dr_py = st.slider(
                "기간 선택",
                min_value=_to_py_datetime(dt_min),
                max_value=_to_py_datetime(dt_max),
                value=(_to_py_datetime(dt_min), _to_py_datetime(dt_max)),
                key="range_m",
            )
            dr = (_to_pd_timestamp(dr_py[0]), _to_pd_timestamp(dr_py[1]))

        if var and sel_regions:
            df = df_rreal.copy()
            if dr:
                df = df[(df["year_month"] >= dr[0]) & (df["year_month"] <= dr[1])]
            df = df[df["region"].isin(sel_regions)]
            df["year_month"] = _coerce_year_month_series(df["year_month"])
            df = df.groupby(["region", "year_month"], as_index=False).agg({var: agg})

            st.altair_chart(
                _alt_line_chart(
                    df.dropna(subset=[var]),
                    x_col="year_month:T",
                    y_col=var,
                    color_col="region",
                    title=f"[월별] {var} — {agg}",
                ),
                use_container_width=True,
            )
            with st.expander("표 (현재 필터 적용)", expanded=False):
                st.dataframe(df.head(200), use_container_width=True)

    # --- t2: yearly aggregation ---
    with t2:
        cA, cB, cC = st.columns([2, 2, 2])
        var_y = cA.selectbox("변수 선택", numeric_cols, index=0 if numeric_cols else None, key="var_y")
        sel_regions_y = cB.multiselect("지역 선택", regions_rreal, default=default_regions_rreal, key="regions_y")
        agg_y = cC.selectbox("집계 방식", ["mean", "sum", "median", "first", "last"], index=0, key="agg_y")

        if "year_month" in df_rreal.columns and var_y:
            df_year = df_rreal.copy()
            df_year["year_month"] = _coerce_year_month_series(df_year["year_month"])
            df_year["year"] = df_year["year_month"].dt.year
            years = sorted(df_year["year"].dropna().unique())
            yr = None
            if years:
                y_min, y_max = int(min(years)), int(max(years))
                yr = st.slider("연도 범위", y_min, y_max, (y_min, y_max), key="yr_y")
            dfy = df_year[df_year["region"].isin(sel_regions_y)] if sel_regions_y else df_year.copy()
            if yr:
                dfy = dfy[(dfy["year"] >= yr[0]) & (dfy["year"] <= yr[1])]
            dfy = dfy.groupby(["region", "year"], as_index=False).agg({var_y: agg_y})

            st.altair_chart(
                _alt_line_chart(
                    dfy,
                    x_col="year:O",
                    y_col=var_y,
                    color_col="region",
                    title=f"[연도별] {var_y} — {agg_y}",
                ),
                use_container_width=True,
            )
            with st.expander("표 (현재 필터 적용)", expanded=False):
                st.dataframe(dfy.head(200), use_container_width=True)
        else:
            st.warning("year_month 또는 선택한 변수가 없어 연도 집계를 수행할 수 없습니다.")

    # --- t3: seasonal month pattern ---
    with t3:
        cA, cB = st.columns([2, 2])
        var_s = cA.selectbox("변수 선택", numeric_cols, index=0 if numeric_cols else None, key="var_s")
        sel_regions_s = cB.multiselect("지역 선택", regions_rreal, default=default_regions_rreal, key="regions_s")

        if var_s and "year_month" in df_rreal.columns and sel_regions_s:
            dfs = df_rreal.copy()
            dfs["year_month"] = _coerce_year_month_series(dfs["year_month"])
            dfs = dfs[dfs["region"].isin(sel_regions_s)]
            dfs["month"] = dfs["year_month"].dt.month
            dfs = dfs.groupby(["region", "month"], as_index=False)[var_s].mean()

            st.altair_chart(
                _alt_line_chart(
                    dfs,
                    x_col="month:O",
                    y_col=var_s,
                    color_col="region",
                    title=f"[월 패턴] {var_s} — 월 평균(전체 연도)",
                ),
                use_container_width=True,
            )
            with st.expander("표 (월 평균)", expanded=False):
                st.dataframe(dfs, use_container_width=True)

# ===== risk_total =====
with tab_total:
    st.subheader("risk_total (hra_label_total_2025_2028.csv)")

    n = st.slider("표시 행 수", 5, 50, 10, key="n_label")
    st.dataframe(df_label_total.head(n), use_container_width=True)

    df_lt = df_label_total.copy()
    if "risk_level" in df_lt.columns:
        df_lt["risk_name"] = df_lt["risk_level"].astype(str).str.strip().str.title()
    else:
        df_lt["risk_name"] = (
            df_lt.get("label_sum", pd.Series(index=df_lt.index))
            .map({1: "Low", 2: "Medium", 3: "High"})
            .fillna("Unknown")
        )

    ORDER = ["Low", "Medium", "High"]
    COLOR = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
    df_lt = df_lt[df_lt["risk_name"].isin(ORDER)].copy()
    df_lt["risk_name"] = pd.Categorical(df_lt["risk_name"], categories=ORDER, ordered=True)

    st.divider()

    overall = (
        df_lt["risk_name"]
        .value_counts()
        .reindex(ORDER)
        .fillna(0)
        .rename_axis("risk_name")
        .reset_index(name="count")
    )

    sel = alt.selection_multi(fields=["risk_name"], bind="legend")
    chart_overall = (
        alt.Chart(overall)
        .mark_bar()
        .encode(
            x=alt.X("risk_name:N", title="risk level", sort=ORDER),
            y=alt.Y("count:Q", title="건수"),
            color=alt.Color(
                "risk_name:N",
                scale=alt.Scale(domain=ORDER, range=[COLOR[k] for k in ORDER]),
                legend=alt.Legend(title="risk level"),
            ),
            tooltip=[
                alt.Tooltip("risk_name:N", title="risk level"),
                alt.Tooltip("count:Q", title="건수", format=","),
            ],
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)),
        )
        .add_selection(sel)
        .properties(title="전체 위험도 건수 요약", height=320)
    )
    st.altair_chart(chart_overall, use_container_width=True)

    rc = df_lt.groupby(["region", "risk_name"], dropna=False).size().reset_index(name="count")
    chart_region = (
        alt.Chart(rc)
        .mark_bar()
        .encode(
            x=alt.X("region:N", title="지역", sort="-y"),
            y=alt.Y("count:Q", title="건수"),
            color=alt.Color(
                "risk_name:N",
                scale=alt.Scale(domain=ORDER, range=[COLOR[k] for k in ORDER]),
                legend=alt.Legend(title="risk level"),
            ),
            tooltip=[
                alt.Tooltip("region:N", title="지역"),
                alt.Tooltip("risk_name:N", title="risk level"),
                alt.Tooltip("count:Q", title="건수", format=","),
            ],
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)),
        )
        .add_selection(sel)
        .properties(title="지역별 위험도 건수(그룹형 막대)", height=420)
    )
    st.altair_chart(chart_region, use_container_width=True)

# ===== risk_stress =====
with tab_stress:
    st.subheader("risk_stress (hra_pairwise_2025_2028.csv)")

    n = st.slider("표시 행 수", 5, 50, 10, key="n_pair")
    st.dataframe(df_pairwise.head(n), use_container_width=True)

    if "stressor" not in df_pairwise.columns or "R" not in df_pairwise.columns or "region" not in df_pairwise.columns:
        st.warning("필수 컬럼(region / stressor / R)이 없어 시각화를 건너뜁니다.")
        st.stop()

    dfp = df_pairwise.copy()
    if "year_month" in dfp.columns:
        dfp["year_month"] = _coerce_year_month_series(dfp["year_month"])

    all_regions_rs = sorted(dfp["region"].dropna().unique().tolist())
    preferred_rs = [
        r for r in all_regions_rs
        if str(r) in ["Incheon", "인천", "Geoje", "거제", "Ulleungdo", "울릉", "울릉도", "울릉군"]
    ]
    default_regions_rs = preferred_rs if preferred_rs else all_regions_rs
    all_stress = sorted(dfp["stressor"].dropna().unique().tolist())

    cA, cB, cC = st.columns([3, 3, 2])
    sel_regions_rs = cA.multiselect("지역(다중 선택)", all_regions_rs, default=default_regions_rs, key="rs_regions")
    sel_stress_rs = cB.multiselect("스트레스 요인", all_stress, default=all_stress, key="rs_stress")
    agg = cC.selectbox("집계방식", ["mean", "max", "sum", "median"], index=0, key="rs_agg")

    if "year_month" in dfp.columns and dfp["year_month"].notna().any():
        dt_min, dt_max = _min_max_dt(dfp)
        dr_py = st.slider(
            "기간 선택",
            min_value=_to_py_datetime(dt_min),
            max_value=_to_py_datetime(dt_max),
            value=(_to_py_datetime(dt_min), _to_py_datetime(dt_max)),
            key="rs_range",
        )
        dr = (_to_pd_timestamp(dr_py[0]), _to_pd_timestamp(dr_py[1]))
        dfp = dfp[(dfp["year_month"] >= dr[0]) & (dfp["year_month"] <= dr[1])]

    if sel_regions_rs:
        dfp = dfp[dfp["region"].isin(sel_regions_rs)]
    if sel_stress_rs:
        dfp = dfp[dfp["stressor"].isin(sel_stress_rs)]

    agg_map = {"mean": "mean", "max": "max", "sum": "sum", "median": "median"}
    g = (
        dfp.groupby(["region", "stressor"], as_index=False)["R"]
        .agg(agg_map[agg])
        .rename(columns={"R": "R_value"})
    )

    if g.empty:
        st.info("선택한 조건에서 데이터가 없습니다.")
        st.stop()

    bar = (
        alt.Chart(g)
        .mark_bar()
        .encode(
            x=alt.X("stressor:N", title="stressor"),
            y=alt.Y("R_value:Q", title=f"{agg} of R"),
            color=alt.Color("region:N", legend=alt.Legend(title="지역")),
            tooltip=[
                alt.Tooltip("region:N", title="지역"),
                alt.Tooltip("stressor:N", title="stressor"),
                alt.Tooltip("R_value:Q", title=f"{agg}(R)", format=",.3f"),
            ],
        )
        .properties(title=f"지역별 스트레스 위험 ({agg} of R)", height=360)
    )
    heat = (
        alt.Chart(g)
        .mark_rect()
        .encode(
            x=alt.X("stressor:N", title="stressor"),
            y=alt.Y("region:N", title="지역"),
            color=alt.Color("R_value:Q", title=f"{agg} of R"),
            tooltip=[
                alt.Tooltip("region:N", title="지역"),
                alt.Tooltip("stressor:N", title="stressor"),
                alt.Tooltip("R_value:Q", title=f"{agg}(R)", format=",.3f"),
            ],
        )
        .properties(title=f"지역 × 스트레스 히트맵 ({agg} of R)", height=420)
    )
    st.altair_chart(bar, use_container_width=True)
    st.altair_chart(heat, use_container_width=True)

    top3 = (
        g.sort_values(["region", "R_value"], ascending=[True, False])
        .groupby("region", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    top3["rank"] = top3.groupby("region")["R_value"].rank(method="first", ascending=False).astype(int)
    top3 = top3.sort_values(["region", "rank"]).copy()
    top3["R_value"] = top3["R_value"].round(3)

    st.markdown("**각 지역 Top-3 스트레스 요인**")
    st.dataframe(
        top3[["region", "rank", "stressor", "R_value"]].rename(
            columns={"stressor": "stressor(top3)", "R_value": f"{agg}(R)"}
        ),
        use_container_width=True,
    )
