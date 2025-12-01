# -*- coding: utf-8 -*-
# pages/01_Data.py — 통합 EDA + 위험 레이블 요약 + 스트레스 분석(Top-3)

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="HRA 대시보드 — Data",
    layout="wide",
    page_icon="📦",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()

# -----------------------------
# 파일 경로/로더
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
def load_csv_auto(name_key: str) -> pd.DataFrame:
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

# -----------------------------
# 스키마 보정
# -----------------------------
def _ensure_region(df: pd.DataFrame):
    cand = [c for c in df.columns if str(c).lower() in ("region","지역")]
    if cand: df = df.rename(columns={cand[0]:"region"})
    return df

def _coerce_year_month_series(s: pd.Series) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    try:
        if s2.isna().mean() > 0.9:
            ss = s.astype(str).str.replace(r"[^0-9]","",regex=True)
            mask6 = ss.str.len()==6
            ss.loc[mask6] = ss[mask6] + "01"
            s2 = pd.to_datetime(ss, errors="coerce", format="%Y%m%d")
    except Exception:
        pass
    return s2.dt.to_period("M").dt.to_timestamp(how="start")

def _ensure_year_month(df: pd.DataFrame):
    lower = {str(c).lower(): c for c in df.columns}
    for key in ("year_month","ym","date","month","dt","yearmonth","날짜","월"):
        if key in lower:
            col = lower[key]
            s = _coerce_year_month_series(df[col])
            if s.notna().any():
                df = df.copy(); df["year_month"] = s; return df
    y_key = next((lower[k] for k in ("year","yr","연","년도","연도") if k in lower), None)
    m_key = next((lower[k] for k in ("month","mo","mn","월") if k in lower), None)
    if y_key is not None and m_key is not None:
        try:
            df = df.copy()
            df["year_month"] = _coerce_year_month_series(
                df[y_key].astype(int).astype(str) + "-" + df[m_key].astype(int).astype(str) + "-01"
            ); return df
        except Exception:
            pass
    return df

def ensure_month_start_datetime(df: pd.DataFrame):
    if "year_month" in df.columns:
        df = df.copy(); df["year_month"] = _coerce_year_month_series(df["year_month"])
    return df

def soft_schema_fix(df: pd.DataFrame):
    return ensure_month_start_datetime(_ensure_year_month(_ensure_region(df)))

# -----------------------------
# 공통 유틸
# -----------------------------
def _numeric_columns(df: pd.DataFrame):
    drop_like = {"region","year_month","label","class","category"}
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not any(k in str(c).lower() for k in drop_like)]

def _min_max_dt(df: pd.DataFrame):
    if "year_month" not in df.columns: return (None, None)
    try:
        s = pd.to_datetime(df["year_month"]); return (s.min(), s.max())
    except Exception:
        return (None, None)

def _alt_line_chart(df, x_col, y_col, color_col="region", title=None):
    ch = (alt.Chart(df).mark_line().encode(
        x=alt.X(x_col, title=None),
        y=alt.Y(y_col, title=y_col),
        color=alt.Color(color_col, legend=alt.Legend(title="지역")),
        tooltip=[color_col, x_col, alt.Tooltip(y_col, format=",.3f")],
    ).properties(height=420))
    return ch.properties(title=title) if title else ch

def _to_py_datetime(x): return x.to_pydatetime() if isinstance(x, pd.Timestamp) else x
def _to_pd_timestamp(x): return pd.Timestamp(x).to_period("M").to_timestamp(how="start")

# -----------------------------
# 데이터 로드
# -----------------------------
load_ok, errors = True, []
try:
    df_rreal = soft_schema_fix(load_csv_auto("rreal"))
except Exception as e:
    load_ok = False; errors.append(("rreal_final_ALL_predicted.csv", str(e)))
try:
    df_label_total = soft_schema_fix(load_csv_auto("label_total"))
except Exception as e:
    load_ok = False; errors.append(("hra_label_total_2025_2028.csv", str(e)))
try:
    df_pairwise = soft_schema_fix(load_csv_auto("pairwise"))
except Exception as e:
    load_ok = False; errors.append(("hra_pairwise_2025_2028.csv", str(e)))

# -----------------------------
# 화면
# -----------------------------
st.title("📊 Dataset")

if not load_ok:
    st.error("일부 데이터 로드 실패")
    for fname, msg in errors:
        with st.expander(f"{fname} 오류 상세"):
            st.code(msg)
    st.stop()

tab_data, tab_total, tab_stress = st.tabs(
    ["🔗 Integrated Data", "⚠️ risk_total", "⚠️ risk_stress"]
)

# ===== Integrated Data =====
with tab_data:
    st.subheader("Integrated Data")
    with st.expander("원자료 미리보기", expanded=False):
        n = st.slider("표시 행 수", 5, 50, 10, key="n_rreal_preview")
        st.dataframe(df_rreal.head(n), use_container_width=True)

    t1, t2, t3 = st.tabs(["📅 월별 시계열", "📆 연도별 집계", "🗓 월 패턴(계절성)"])

    numeric_cols = _numeric_columns(df_rreal)
    regions = sorted(df_rreal["region"].dropna().unique()) if "region" in df_rreal.columns else []
    default_regions = regions[:5] if len(regions) > 5 else regions

    # 월별
    with t1:
        cA, cB, cC = st.columns([2,2,2])
        var = cA.selectbox("변수 선택", numeric_cols, index=0 if numeric_cols else None, key="var_m")
        sel_regions = cB.multiselect("지역 선택", regions, default=default_regions, key="regions_m")
        agg = cC.selectbox("집계 방식", ["mean","sum","median","first","last"], index=0, key="agg_m")

        dt_min, dt_max = _min_max_dt(df_rreal); dr = None
        if dt_min is not None and dt_max is not None:
            dr_py = st.slider("기간 선택", min_value=_to_py_datetime(dt_min),
                              max_value=_to_py_datetime(dt_max),
                              value=(_to_py_datetime(dt_min), _to_py_datetime(dt_max)), key="range_m")
            dr = (_to_pd_timestamp(dr_py[0]), _to_pd_timestamp(dr_py[1]))

        if var and sel_regions:
            df = df_rreal.copy()
            if dr: df = df[(df["year_month"] >= dr[0]) & (df["year_month"] <= dr[1])]
            df = df[df["region"].isin(sel_regions)]
            df["year_month"] = _coerce_year_month_series(df["year_month"])
            df = df.groupby(["region","year_month"], as_index=False).agg({var: agg})

            st.altair_chart(
                _alt_line_chart(df.dropna(subset=[var]),
                                x_col="year_month:T", y_col=var, color_col="region",
                                title=f"[월별] {var} — {agg}"),
                use_container_width=True
            )
            with st.expander("표(현재 필터 적용)"):
                st.dataframe(df.head(200), use_container_width=True)

    # 연도별
    with t2:
        cA, cB, cC = st.columns([2,2,2])
        var_y = cA.selectbox("변수 선택", numeric_cols, index=0 if numeric_cols else None, key="var_y")
        sel_regions_y = cB.multiselect("지역 선택", regions, default=default_regions, key="regions_y")
        agg_y = cC.selectbox("집계 방식", ["mean","sum","median","first","last"], index=0, key="agg_y")

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
            if yr: dfy = dfy[(dfy["year"]>=yr[0]) & (dfy["year"]<=yr[1])]
            dfy = dfy.groupby(["region","year"], as_index=False).agg({var_y: agg_y})

            st.altair_chart(
                _alt_line_chart(dfy, x_col="year:O", y_col=var_y, color_col="region",
                                title=f"[연도별] {var_y} — {agg_y}"),
                use_container_width=True
            )
            with st.expander("표(현재 필터 적용)"):
                st.dataframe(dfy, use_container_width=True)
        else:
            st.warning("year_month 또는 선택한 변수가 없어 연도 집계를 수행할 수 없습니다.")

    # 월 패턴
    with t3:
        cA, cB = st.columns([2,2])
        var_s = cA.selectbox("변수 선택", numeric_cols, index=0 if numeric_cols else None, key="var_s")
        sel_regions_s = cB.multiselect("지역 선택", regions, default=default_regions, key="regions_s")

        if var_s and "year_month" in df_rreal.columns and sel_regions_s:
            dfs = df_rreal.copy()
            dfs["year_month"] = _coerce_year_month_series(dfs["year_month"])
            dfs = dfs[dfs["region"].isin(sel_regions_s)]
            dfs["month"] = dfs["year_month"].dt.month
            dfs = dfs.groupby(["region","month"], as_index=False)[var_s].mean()

            st.altair_chart(
                _alt_line_chart(dfs, x_col="month:O", y_col=var_s, color_col="region",
                                title=f"[월 패턴] {var_s} — 월 평균(연도 전반)"),
                use_container_width=True
            )
            with st.expander("표(월 평균)"):
                st.dataframe(dfs, use_container_width=True)

# ===== risk_total =====
with tab_total:
    st.subheader("risk_total")
    n = st.slider("표시 행 수", 5, 50, 10, key="n_label")
    st.dataframe(df_label_total.head(n), use_container_width=True)

    df_lt = df_label_total.copy()
    if "risk_level" in df_lt.columns:
        df_lt["risk_name"] = df_lt["risk_level"].astype(str).str.strip().str.title()
    else:
        df_lt["risk_name"] = df_lt.get("label_sum", pd.Series(index=df_lt.index)).map({1:"Low",2:"Medium",3:"High"}).fillna("Unknown")

    ORDER = ["Low", "Medium", "High"]
    COLOR = {"Low": "#4CAF50", "Medium": "#FFC107", "High": "#F44336"}
    df_lt = df_lt[df_lt["risk_name"].isin(ORDER)].copy()
    df_lt["risk_name"] = pd.Categorical(df_lt["risk_name"], categories=ORDER, ordered=True)

    st.divider()

    # 전체 건수
    overall = (df_lt["risk_name"].value_counts()
               .reindex(ORDER).fillna(0)
               .rename_axis("risk_name").reset_index(name="count"))

    sel = alt.selection_multi(fields=["risk_name"], bind="legend")
    chart_overall = (alt.Chart(overall).mark_bar().encode(
        x=alt.X("risk_name:N", title="risk level", sort=ORDER),
        y=alt.Y("count:Q", title="건수"),
        color=alt.Color("risk_name:N",
                        scale=alt.Scale(domain=ORDER, range=[COLOR[k] for k in ORDER]),
                        legend=alt.Legend(title="risk level")),
        tooltip=[alt.Tooltip("risk_name:N", title="risk level"),
                 alt.Tooltip("count:Q", title="건수", format=",")],
        opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)),
    ).add_selection(sel).properties(title="전체 위험도 건수 요약", height=320))
    st.altair_chart(chart_overall, use_container_width=True)

    # 지역별 건수
    rc = (df_lt.groupby(["region", "risk_name"], dropna=False).size().reset_index(name="count"))
    chart_region = (alt.Chart(rc).mark_bar().encode(
        x=alt.X("region:N", title="지역", sort="-y"),
        y=alt.Y("count:Q", title="건수"),
        color=alt.Color("risk_name:N",
                        scale=alt.Scale(domain=ORDER, range=[COLOR[k] for k in ORDER]),
                        legend=alt.Legend(title="risk level")),
        tooltip=[alt.Tooltip("region:N", title="지역"),
                 alt.Tooltip("risk_name:N", title="risk level"),
                 alt.Tooltip("count:Q", title="건수", format=",")],
        opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)),
    ).add_selection(sel).properties(title="지역별 위험도 건수(그룹형 막대)", height=420))
    st.altair_chart(chart_region, use_container_width=True)

# ===== risk_stress =====
with tab_stress:
    st.subheader("risk_stress")

    n = st.slider("표시 행 수", 5, 50, 10, key="n_pair")
    st.dataframe(df_pairwise.head(n), use_container_width=True)

    cols_lower = {c.lower(): c for c in df_pairwise.columns}
    stressor_col = next((c for c in df_pairwise.columns if "stress" in c.lower()), None)
    r_col = next((cols_lower[k] for k in ("r","risk") if k in cols_lower), None)
    if stressor_col is None or r_col is None or "region" not in df_pairwise.columns:
        st.warning("필수 컬럼(region / stressor / R)이 없어 시각화를 건너뜁니다.")
        st.stop()

    dfp = df_pairwise.copy()
    dfp["year_month"] = _coerce_year_month_series(dfp["year_month"]) if "year_month" in dfp.columns else None

    all_regions = sorted(dfp["region"].dropna().unique().tolist())
    preferred = [r for r in all_regions if str(r) in ["Incheon","인천","Geoje","거제","Ulleungdo","울릉","울릉도","울릉군"]]
    default_regions = preferred if preferred else all_regions
    all_stress = sorted(dfp[stressor_col].dropna().unique().tolist())

    cA, cB, cC = st.columns([3, 3, 2])
    sel_regions = cA.multiselect("지역(다중 선택)", all_regions, default=default_regions, key="rs_regions")
    sel_stress  = cB.multiselect("스트레스 요인", all_stress, default=all_stress, key="rs_stress")
    agg = cC.selectbox("집계방식", ["mean", "max", "sum", "median"], index=0, key="rs_agg")

    if "year_month" in dfp.columns and dfp["year_month"].notna().any():
        dt_min, dt_max = _min_max_dt(dfp)
        dr_py = st.slider("기간 선택", min_value=_to_py_datetime(dt_min), max_value=_to_py_datetime(dt_max),
                          value=(_to_py_datetime(dt_min), _to_py_datetime(dt_max)), key="rs_range")
        dr = (_to_pd_timestamp(dr_py[0]), _to_pd_timestamp(dr_py[1]))
        dfp = dfp[(dfp["year_month"] >= dr[0]) & (dfp["year_month"] <= dr[1])]

    if sel_regions: dfp = dfp[dfp["region"].isin(sel_regions)]
    if sel_stress:  dfp = dfp[dfp[stressor_col].isin(sel_stress)]

    agg_map = {"mean":"mean","max":"max","sum":"sum","median":"median"}
    g = (dfp.groupby(["region", stressor_col], as_index=False)[r_col]
            .agg(agg_map[agg]).rename(columns={r_col:"R_value", stressor_col:"stressor"}))

    if g.empty:
        st.info("선택한 조건에서 데이터가 없습니다."); st.stop()

    # 막대 + 히트맵
    bar = (alt.Chart(g).mark_bar().encode(
        x=alt.X("stressor:N", title="stressor"),
        y=alt.Y("R_value:Q", title=f"{agg} of R"),
        color=alt.Color("region:N", legend=alt.Legend(title="지역")),
        tooltip=[alt.Tooltip("region:N", title="지역"),
                 alt.Tooltip("stressor:N", title="stressor"),
                 alt.Tooltip("R_value:Q", title=f"{agg}(R)", format=",.3f")],
    ).properties(title=f"지역별 스트레스 위험", height=360))
    heat = (alt.Chart(g).mark_rect().encode(
        x=alt.X("stressor:N", title="stressor"),
        y=alt.Y("region:N", title="지역"),
        color=alt.Color("R_value:Q", title=f"{agg} of R"),
        tooltip=[alt.Tooltip("region:N", title="지역"),
                 alt.Tooltip("stressor:N", title="stressor"),
                 alt.Tooltip("R_value:Q", title=f"{agg}(R)", format=",.3f")],
    ).properties(title=f"지역 × 스트레스 히트맵", height=420))
    st.altair_chart(bar, use_container_width=True)
    st.altair_chart(heat, use_container_width=True)

    # Top-3
    top3 = (g.sort_values(["region","R_value"], ascending=[True,False])
              .groupby("region", as_index=False).head(3).reset_index(drop=True))
    top3["rank"] = top3.groupby("region")["R_value"].rank(method="first", ascending=False).astype(int)
    top3 = top3.sort_values(["region","rank"]).copy()
    top3["R_value"] = top3["R_value"].round(3)

    st.markdown("**각 지역 Top-3 스트레스 요인**")
    st.dataframe(
        top3[["region","rank","stressor","R_value"]]
            .rename(columns={"stressor":"stressor(top3)", "R_value":f"{agg}(R)"}),
        use_container_width=True
    )
