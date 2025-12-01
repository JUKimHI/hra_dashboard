# -*- coding: utf-8 -*-
# pages/02_risk_map.py — 월별 위험도 지도 + High 지역 주요 스트레스(Top-1)

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Risk Map",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "hra_label_total_2025_2028.csv"
PAIRWISE_FILE = "hra_pairwise_2025_2028.csv"

def find_data_path(filename: str):
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
def load_label_total() -> pd.DataFrame:
    p = find_data_path(DATA_FILE)
    if p is None: raise FileNotFoundError(f"데이터 없음: {DATA_FILE}")
    for enc in ("utf-8-sig","utf-8","cp949"):
        try: df = pd.read_csv(p, encoding=enc); break
        except Exception: continue

    r_like = [c for c in df.columns if str(c).lower() in ("region","지역")]
    if r_like: df = df.rename(columns={r_like[0]:"region"})

    ym_like = None
    for key in ("year_month","ym","date","dt","yearmonth","월","날짜"):
        hit = [c for c in df.columns if str(c).lower()==key]
        if hit: ym_like = hit[0]; break
    if ym_like is not None:
        ym = pd.to_datetime(df[ym_like], errors="coerce", infer_datetime_format=True)
        if ym.isna().mean() > 0.9:
            s = df[ym_like].astype(str).str.replace(r"[^0-9]","",regex=True)
            mask6 = s.str.len()==6; s.loc[mask6] = s[mask6] + "01"
            ym = pd.to_datetime(s, errors="coerce", format="%Y%m%d")
        df["year_month"] = ym.dt.to_period("M").dt.to_timestamp(how="start")

    if "risk_level" not in df.columns:
        label_like = [c for c in df.columns if "label" in c.lower() or "risk" in c.lower()]
        if label_like: df = df.rename(columns={label_like[0]:"risk_level"})
    return df

@st.cache_data(show_spinner=False)
def load_pairwise() -> pd.DataFrame:
    p = find_data_path(PAIRWISE_FILE)
    if p is None: raise FileNotFoundError(f"데이터 없음: {PAIRWISE_FILE}")
    for enc in ("utf-8-sig","utf-8","cp949"):
        try: df = pd.read_csv(p, encoding=enc); break
        except Exception: continue

    r_like = [c for c in df.columns if str(c).lower() in ("region","지역")]
    if r_like: df = df.rename(columns={r_like[0]:"region"})

    ym_col = None
    for key in ("year_month","ym","date","dt","yearmonth","월","날짜"):
        hit = [c for c in df.columns if str(c).lower()==key]
        if hit: ym_col = hit[0]; break
    if ym_col is not None:
        ym = pd.to_datetime(df[ym_col], errors="coerce", infer_datetime_format=True)
        if ym.isna().mean() > 0.9:
            s = df[ym_col].astype(str).str.replace(r"[^0-9]","",regex=True)
            mask6 = s.str.len()==6; s.loc[mask6] = s[mask6] + "01"
            ym = pd.to_datetime(s, errors="coerce", format="%Y%m%d")
        df["year_month"] = ym.dt.to_period("M").dt.to_timestamp(how="start")

    s_col = next((c for c in df.columns if "stress" in c.lower()), None)
    if s_col: df = df.rename(columns={s_col: "stressor"})
    r_col = next((c for c in df.columns if str(c).lower() in ("r","risk")), None)
    if r_col: df = df.rename(columns={r_col:"R"})
    e_col = next((c for c in df.columns if (str(c).lower()=="e") or ("exposure" in str(c).lower())), None)
    if e_col: df = df.rename(columns={e_col:"E"})
    c_col = next((c for c in df.columns if (str(c).lower()=="c") or ("consequence" in str(c).lower()) or ("impact" in str(c).lower())), None)
    if c_col: df = df.rename(columns={c_col:"C"})
    return df

df = load_label_total().copy()
df_pw = load_pairwise().copy()

# 좌표 보강
lat_col = next((c for c in df.columns if str(c).lower() in ("lat","latitude","위도")), None)
lon_col = next((c for c in df.columns if str(c).lower() in ("lon","lng","longitude","경도")), None)
REGION_COORDS = {
    "Incheon": (37.456, 126.705), "Geoje": (34.880, 128.620), "Ulleungdo": (37.500, 130.900),
    "인천": (37.456, 126.705), "거제": (34.880, 128.620),
    "울릉": (37.500, 130.900), "울릉도": (37.500, 130.900), "울릉군": (37.500, 130.900),
}
def add_coords(_df: pd.DataFrame) -> pd.DataFrame:
    _df = _df.copy()
    if lat_col and lon_col:
        _df["lat"] = _df[lat_col]; _df["lon"] = _df[lon_col]; return _df
    _df["lat"] = _df["region"].map(lambda r: REGION_COORDS.get(str(r),(np.nan,np.nan))[0])
    _df["lon"] = _df["region"].map(lambda r: REGION_COORDS.get(str(r),(np.nan,np.nan))[1])
    return _df

df = add_coords(df).dropna(subset=["region","year_month","lat","lon"]).copy()

# 컬러/라벨
if "risk_level" in df.columns:
    df["risk_level"] = df["risk_level"].astype(str).str.strip().str.title()
else:
    if "R_sum" in df.columns:
        bins = pd.qcut(df["R_sum"].rank(method="first"), q=3, labels=["Low","Medium","High"])
        df["risk_level"] = bins.astype(str)
    else:
        df["risk_level"] = "Medium"

COLOR_MAP = {"Low":"#4CAF50","Medium":"#FFC107","High":"#F44336"}
CATS = ["Low","Medium","High"]
df["risk_level"] = pd.Categorical(df["risk_level"], categories=CATS, ordered=True)
df["ym_str"] = df["year_month"].dt.strftime("%Y_%m")
if "R_sum" not in df.columns: df["R_sum"] = 1.0

# ----------------------------
# UI (단일 월)
# ----------------------------
st.title("Risk map")

all_regions = sorted(df["region"].unique().tolist())
default_regions = [r for r in all_regions if str(r) in
                   ["Incheon","인천","Geoje","거제","Ulleungdo","울릉","울릉도","울릉군"]] or all_regions[:3]

colA, colB = st.columns([2,2])
sel_regions = colA.multiselect("지역 선택", all_regions, default=default_regions)
size_scale = colB.slider("버블 크기 스케일", 5, 40, 20)

df_v = df[df["region"].isin(sel_regions)].copy() if sel_regions else df.copy()

if "year_month" in df_v.columns and df_v["year_month"].notna().any():
    years = sorted(df_v["year_month"].dt.year.unique().tolist())
    colY, colM = st.columns([1,1])
    sel_year = colY.selectbox("연도 선택", years, index=0, key="year_sel")
    months_avail = sorted(df_v.loc[df_v["year_month"].dt.year.eq(sel_year), "year_month"].dt.month.unique().tolist())
    m_labels = [f"{m:02d}" for m in months_avail]
    sel_month = int(colM.selectbox("월 선택", m_labels, index=0, key="month_sel"))

    sel_month_str = f"{sel_year}_{sel_month:02d}"
    st.caption(f"year_month = {sel_month_str}")
    df_m = df_v[(df_v["year_month"].dt.year == sel_year) & (df_v["year_month"].dt.month == sel_month)].copy()
else:
    df_m = df_v.copy()
    sel_year = int(df_m["year_month"].dt.year.min()); sel_month = int(df_m["year_month"].dt.month.min())

center_korea = {"lat":36.2,"lon":128.0}
def base_hover_cols(_df: pd.DataFrame):
    cols = {"risk_level":True,"ym_str":True}
    if "R_sum" in _df.columns: cols["R_sum"] = ":.3f"
    return cols

fig = px.scatter_mapbox(
    df_m, lat="lat", lon="lon",
    color="risk_level", color_discrete_map=COLOR_MAP,
    category_orders={"risk_level": CATS},
    size="R_sum", size_max=size_scale,
    hover_name="region", hover_data=base_hover_cols(df_m),
    zoom=5.3, center=center_korea, height=560,
)

present = {tr.name for tr in fig.data}
for lvl in CATS:
    if lvl not in present:
        fig.add_scattermapbox(lat=[None], lon=[None], mode="markers",
                              marker=dict(size=10, color=COLOR_MAP[lvl]),
                              name=lvl, showlegend=True)
_order = {name:i for i,name in enumerate(CATS)}
fig.data = tuple(sorted(fig.data, key=lambda tr: _order.get(tr.name, 99)))
fig.update_layout(mapbox_style="open-street-map",
                  margin=dict(l=0,r=0,t=0,b=0),
                  legend_title_text="Risk level")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 선택 월 High 지역 → Top-1 스트레스
# ----------------------------
st.markdown("### 🔎 High 지역의 주요 스트레스 요인(Top-1)")
high_regions = df_m.loc[df_m["risk_level"]=="High","region"].dropna().unique().tolist()
if not high_regions:
    st.info("선택한 연/월에는 High로 분류된 지역이 없습니다.")
else:
    ym_sel = pd.Timestamp(f"{sel_year}-{sel_month:02d}-01")
    need_cols = {"year_month","region","stressor","R"}
    if not need_cols.issubset(set(map(str, df_pw.columns))):
        st.warning("pairwise 데이터에 필요한 컬럼(year_month, region, stressor, R)이 부족합니다.")
    else:
        dfx = df_pw[(df_pw["year_month"]==ym_sel) & (df_pw["region"].isin(high_regions))].copy()
        if dfx.empty:
            st.info("선택한 연/월/지역에 대한 pairwise 레코드가 없습니다.")
        else:
            g = (dfx.groupby(["region","stressor"], as_index=False)["R"].mean()
                    .rename(columns={"R":"R_mean"}))
            top1 = (g.sort_values(["region","R_mean"], ascending=[True,False])
                      .groupby("region", as_index=False).head(1))
            top1["R_mean"] = top1["R_mean"].round(3)
            st.dataframe(top1.rename(columns={"region":"지역","stressor":"최대 R 요인","R_mean":"R값(평균)"}),
                         use_container_width=True)