# -*- coding: utf-8 -*-
# Home.py — 해양 생물다양성 리스크 대시보드 (요약/하이라이트/빠른 이동)

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="HRA 대시보드 — Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()

# -----------------------------
# 경로/로더/스키마 보정
# -----------------------------
DATA_FILES = {
    "rreal": "rreal_final_ALL_predicted.csv",
    "label_total": "hra_label_total_2025_2028.csv",
    "pairwise": "hra_pairwise_2025_2028.csv",
}

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
def load_csv(name_key: str) -> pd.DataFrame:
    fn = DATA_FILES[name_key]
    p = find_data_path(fn)
    if p is None:
        raise FileNotFoundError(f"데이터 파일을 찾지 못했습니다: {fn}")
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(p)

def _coerce_year_month(s: pd.Series) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if s2.isna().mean() > 0.9:
        ss = s.astype(str).str.replace(r"[^0-9]", "", regex=True)
        mask6 = ss.str.len() == 6
        ss.loc[mask6] = ss[mask6] + "01"
        s2 = pd.to_datetime(ss, errors="coerce", format="%Y%m%d")
    return s2.dt.to_period("M").dt.to_timestamp(how="start")

def soft_fix(df: pd.DataFrame):
    r_like = [c for c in df.columns if str(c).lower() in ("region", "지역")]
    if r_like: df = df.rename(columns={r_like[0]: "region"})
    if "year_month" in df.columns:
        df = df.copy(); df["year_month"] = _coerce_year_month(df["year_month"])
    else:
        lower = {str(c).lower(): c for c in df.columns}
        for k in ("ym", "date", "dt", "yearmonth", "월", "날짜"):
            if k in lower:
                df = df.copy(); df["year_month"] = _coerce_year_month(df[lower[k]])
                break
    return df

# -----------------------------
# 데이터 로드
# -----------------------------
load_errors = []
try:
    df_label = soft_fix(load_csv("label_total"))
except Exception as e:
    df_label = pd.DataFrame(); load_errors.append(("hra_label_total_2025_2028.csv", str(e)))

try:
    df_pw = soft_fix(load_csv("pairwise"))
except Exception as e:
    df_pw = pd.DataFrame(); load_errors.append(("hra_pairwise_2025_2028.csv", str(e)))

# -----------------------------
# 헤더
# -----------------------------
st.title("🌊 해양 생물다양성 리스크 대시보드")
st.caption("왼쪽 사이드바에서 페이지를 이동할 수 있어요. 아래에서 최근 위험도 하이라이트를 빠르게 확인하세요.")

if load_errors:
    with st.expander("⚠️ 데이터 로드 오류(펼쳐보기)"):
        for fn, msg in load_errors:
            st.write(f"**{fn}**"); st.code(msg)

if df_label.empty:
    st.error("`hra_label_total_2025_2028.csv` 를 불러오지 못했습니다. data/ 폴더를 확인하세요.")
    st.stop()

# risk_level 표준화 (없으면 label_sum → 이름)
if "risk_level" in df_label.columns:
    df_label["risk_name"] = df_label["risk_level"].astype(str).str.strip().str.title()
else:
    df_label["risk_name"] = (
        df_label.get("label_sum", pd.Series(index=df_label.index))
                .map({1: "Low", 2: "Medium", 3: "High"})
                .fillna("Unknown")
    )

ORDER = ["Low", "Medium", "High"]
COLOR = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
df_label["risk_name"] = pd.Categorical(df_label["risk_name"], categories=ORDER, ordered=True)

# -----------------------------
# KPI 카드
# -----------------------------
regions = sorted(df_label["region"].dropna().unique().tolist()) if "region" in df_label.columns else []
first_dt = df_label["year_month"].min() if "year_month" in df_label.columns else None
last_dt  = df_label["year_month"].max() if "year_month" in df_label.columns else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("📍 지역 수", f"{len(regions):,}")
c2.metric("🧾 전체 레코드", f"{len(df_label):,}")
c3.metric("⏱️ 기간 시작", first_dt.strftime("%Y-%m") if pd.notna(first_dt) else "-")
c4.metric("⏱️ 기간 종료", last_dt.strftime("%Y-%m") if pd.notna(last_dt) else "-")

st.divider()

# -----------------------------
# 연/월 선택 → 이번 달 요약
# -----------------------------
years = sorted(df_label["year_month"].dt.year.unique().tolist())
default_year = max(years) if years else 2025
months_avail = sorted(df_label.loc[df_label["year_month"].dt.year.eq(default_year), "year_month"]
                      .dt.month.unique().tolist())

colY, colM = st.columns([1, 1])
sel_year = colY.selectbox("연도 선택", years, index=years.index(default_year) if years else 0)
m_labels = [f"{m:02d}" for m in months_avail]
default_mm = f"{max(months_avail):02d}" if months_avail else "01"
sel_month = int(colM.selectbox("월 선택", m_labels, index=m_labels.index(default_mm) if m_labels else 0))

sel_stamp = pd.Timestamp(f"{sel_year}-{sel_month:02d}-01")
st.caption(f"선택 월: **{sel_year}_{sel_month:02d}**")
df_m = df_label[df_label["year_month"].eq(sel_stamp)].copy()

# 분포(전체/지역별)
lcol, rcol = st.columns([1.1, 1.3])

with lcol:
    st.subheader("이번 달 위험도 분포")
    dist = (df_m["risk_name"].value_counts()
                      .reindex(ORDER).fillna(0)
                      .rename_axis("risk_name").reset_index(name="count"))
    chart = (alt.Chart(dist).mark_bar().encode(
        x=alt.X("risk_name:N", title="risk level", sort=ORDER),
        y=alt.Y("count:Q", title="건수"),
        color=alt.Color("risk_name:N",
                        scale=alt.Scale(domain=ORDER, range=[COLOR[k] for k in ORDER]),
                        legend=alt.Legend(title="risk level")),
        tooltip=[alt.Tooltip("risk_name:N", title="risk level"),
                 alt.Tooltip("count:Q", title="건수", format=",")],
    ).properties(height=300))
    st.altair_chart(chart, use_container_width=True)

with rcol:
    st.subheader("지역별 위험도 분포(스택)")
    rc = (df_m.groupby(["region", "risk_name"], dropna=False)
               .size().reset_index(name="count"))
    stacked = (alt.Chart(rc).mark_bar().encode(
        x=alt.X("region:N", title="지역"),
        y=alt.Y("count:Q", stack="zero", title="건수"),
        color=alt.Color("risk_name:N",
                        scale=alt.Scale(domain=ORDER, range=[COLOR[k] for k in ORDER]),
                        legend=alt.Legend(title="risk level")),
        tooltip=[alt.Tooltip("region:N", title="지역"),
                 alt.Tooltip("risk_name:N", title="risk level"),
                 alt.Tooltip("count:Q", title="건수", format=",")],
    ).properties(height=300))
    st.altair_chart(stacked, use_container_width=True)

st.divider()

# -----------------------------
# High 지역 하이라이트 + pairwise Top-1 스트레스
# -----------------------------
st.subheader("🔎 High 지역 하이라이트 (선택 월)")
high_regions = df_m.loc[df_m["risk_name"] == "High", "region"].dropna().unique().tolist()
if not high_regions:
    st.info("선택한 연/월에는 **High**로 분류된 지역이 없습니다.")
else:
    if df_pw.empty or not set(["region", "year_month"]).issubset(df_pw.columns):
        st.warning("pairwise 데이터가 없거나 스키마가 맞지 않아 스트레스 요인을 볼 수 없습니다.")
    else:
        dfp = df_pw.copy()
        if "stressor" not in dfp.columns:
            s_col = next((c for c in dfp.columns if "stress" in c.lower()), None)
            if s_col: dfp = dfp.rename(columns={s_col: "stressor"})
        r_col = next((c for c in dfp.columns if str(c).lower() in ("r", "risk")), None)
        if r_col and r_col != "R": dfp = dfp.rename(columns={r_col: "R"})

        dfp = dfp[(dfp["year_month"] == sel_stamp) & (dfp["region"].isin(high_regions))].copy()
        if dfp.empty or ("stressor" not in dfp.columns) or ("R" not in dfp.columns):
            st.info("선택한 월의 pairwise 레코드가 없거나 필수 컬럼(stressor/R)이 없습니다.")
        else:
            g = (dfp.groupby(["region", "stressor"], as_index=False)["R"]
                    .mean().rename(columns={"R": "R_mean"}))
            top1 = (g.sort_values(["region", "R_mean"], ascending=[True, False])
                      .groupby("region", as_index=False).head(1))
            top1["R_mean"] = top1["R_mean"].round(3)
            st.dataframe(
                top1.rename(columns={"region":"지역","stressor":"최대 R 요인","R_mean":"R값(평균)"}),
                use_container_width=True
            )

st.divider()

# -----------------------------
# 빠른 이동
# -----------------------------
st.subheader("⚡ 빠른 이동")
col1, col2 = st.columns(2)
with col1:
    if hasattr(st, "page_link"):
        st.page_link("pages/01_Data.py", label="📦 Data (Integrated / risk_total / risk_stress)")
    else:
        st.markdown("📦 **Data** 페이지는 사이드바에서 **Data**를 클릭하세요.")
with col2:
    if hasattr(st, "page_link"):
        st.page_link("pages/02_Risk.py", label="🗺️ Risk map (월별 지도)")
    else:
        st.markdown("🗺️ **Risk map** 페이지는 사이드바에서 **risk map**을 클릭하세요.")