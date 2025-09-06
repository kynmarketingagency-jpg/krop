# Streamlit app for Krop — Phase 2A (Data Explorer + Map)
# Friendly teacher comments inside 🌱

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# --- Load data (CSV "bookshelves") ---
@st.cache_data(show_spinner=False)
def load_csvs(data_dir: Path):
    def maybe_read(name):
        p = data_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    return (
        pd.read_csv(data_dir / "countries.csv"),
        maybe_read("bans.csv"),
        pd.read_csv(data_dir / "reg_policies.csv"),
        maybe_read("studies.csv"),
        maybe_read("incidents.csv"),
        maybe_read("companies.csv"),
    )

countries, bans, policies, studies, incidents, companies = load_csvs(DATA)
# --- Load data (CSV "bookshelves") ---
countries = pd.read_csv(DATA / "countries.csv")
bans = pd.read_csv(DATA / "bans.csv")
policies = pd.read_csv(DATA / "reg_policies.csv")
studies = pd.read_csv(DATA / "studies.csv")
incidents = (pd.read_csv(DATA / "incidents.csv") if (DATA / "incidents.csv").exists() else pd.DataFrame())
companies = (pd.read_csv(DATA / "companies.csv") if (DATA / "companies.csv").exists() else pd.DataFrame())
# Prefer diet-specific mapping file, fall back to legacy name if present
_mapping_diet = DATA / "mapping_diet.csv"
_mapping_legacy = DATA / "mapping.csv"
if _mapping_diet.exists():
    mapping = pd.read_csv(_mapping_diet)
elif _mapping_legacy.exists():
    mapping = pd.read_csv(_mapping_legacy)
else:
    mapping = pd.DataFrame()

# --- Page title ---
st.set_page_config(page_title="Krop Explorer", layout="wide")
st.title("Krop 🌾 — GMO Explorer")
st.caption("Click a country to see stance, bans, approvals, studies, incidents. All rows link to sources.")

# --- Sidebar filters ---
st.sidebar.header("Filters")
region = st.sidebar.selectbox("Region", ["All"] + sorted(countries["region"].dropna().unique().tolist()))
stance = st.sidebar.selectbox("Stance", ["All"] + sorted(countries["gmo_stance"].dropna().unique().tolist()))
search = st.sidebar.text_input("Search country")

# Filter countries table
filtered = countries.copy()
if region != "All":
    filtered = filtered[filtered["region"] == region]
if stance != "All":
    filtered = filtered[filtered["gmo_stance"] == stance]
if search.strip():
    filtered = filtered[filtered["country"].str.contains(search.strip(), case=False, na=False)]

# --- Map coloring by stance (Ban / Moratorium / Restrictive / Cultivates / etc.) ---
def normalize_stance(s):
    s = (s or "").lower()
    if "ban" in s and "moratorium" not in s:
        return "Ban"
    if "moratorium" in s:
        return "Moratorium"
    if "restrict" in s:
        return "Restrictive"
    if "cultivat" in s or "allow" in s or "approve" in s:
        return "Cultivates"
    return "Other"

countries["stance_simple"] = countries["gmo_stance"].apply(normalize_stance)

# Build a choropleth using country names (Plotly handles most names)
map_df = countries.copy()
if region != "All":
    map_df = map_df[map_df["region"] == region]

stance_order = ["Ban", "Moratorium", "Restrictive", "Cultivates", "Other"]
map_df["stance_simple"] = pd.Categorical(map_df["stance_simple"], categories=stance_order, ordered=True)

fig = px.choropleth(
    map_df,
    locations="country",
    locationmode="country names",
    color="stance_simple",
    category_orders={"stance_simple": stance_order},
    hover_name="country",
    hover_data={"region": True, "gmo_stance": True},
    title="World overview by GMO stance",
)
st.plotly_chart(fig, use_container_width=True)

# ---- Region & stance summary chart ----
st.markdown("### Region breakdown")

tmp = countries.copy()
tmp["stance_simple"] = tmp["gmo_stance"].apply(normalize_stance)

agg = (
    tmp.groupby(["region", "stance_simple"], dropna=True)
      .size().reset_index(name="count")
)

stance_order = ["Ban", "Moratorium", "Restrictive", "Cultivates", "Other"]
bar = px.bar(
    agg,
    x="region",
    y="count",
    color="stance_simple",
    category_orders={"stance_simple": stance_order},
    barmode="stack",
    title="Countries by stance, per region"
)
st.plotly_chart(bar, use_container_width=True)

# --- Country picker & profile ---
st.subheader("Country profile")
col1, col2 = st.columns([1, 2])

with col1:
    pick = st.selectbox("Choose a country", sorted(filtered["country"].unique().tolist()))
    c_info = countries[countries["country"] == pick].iloc[0].to_dict()
    st.markdown(f"**Region:** {c_info.get('region','')}")
    st.markdown(f"**Stance:** {c_info.get('gmo_stance','')}")
    if isinstance(c_info.get("notes"), str) and c_info["notes"]:
        st.markdown(f"**Notes:** {c_info['notes']}")
    if isinstance(c_info.get("source"), str) and c_info["source"]:
        st.markdown(f"[Country source]({c_info['source']})")

with col2:
    # Bans
    st.markdown("### Bans / Moratoria")
    b = bans[bans["country"] == pick].copy()
    if len(b):
        b = b[["scope","type","year","notes","source"]].sort_values("year", ascending=False)
        st.dataframe(b, use_container_width=True)
    else:
        st.info("No bans recorded for this country.")


 # ---- CSV download helper ----
def download_df_button(df: pd.DataFrame, label: str, fname: str | None = None):
    if df is None or df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=fname or (label.lower().replace(" ", "_") + ".csv"),
        mime="text/csv",
        use_container_width=True,
    )

# ---- Diet & Prediction helpers ----
def _first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None
def get_mapping_cols(mapping_df: pd.DataFrame):
    col_food   = _first_existing_col(mapping_df, ["food","product","item","name"])
    col_crop   = _first_existing_col(mapping_df, ["crop","crops"])
    col_use    = _first_existing_col(mapping_df, ["use","consumer","audience"])
    col_region = _first_existing_col(mapping_df, ["country_or_region","country","region"])
    col_notes  = _first_existing_col(mapping_df, ["notes","note"])
    col_src    = _first_existing_col(mapping_df, ["source","url","link"])
    return col_food, col_crop, col_use, col_region, col_notes, col_src
def get_country_mapping(mapping_df: pd.DataFrame, country: str):
    if mapping_df is None or mapping_df.empty:
        return pd.DataFrame()
    cf, cc, cu, cr, cn, cs = get_mapping_cols(mapping_df)
    if not all([cf, cc, cu, cr]):
        return pd.DataFrame()
    # Filter rows for the selected country, else fall back to region/global rows
    df = mapping_df.copy()
    # Normalize text
    for col in [cf, cc, cu, cr]:
        df[col] = df[col].astype(str)
    country_lower = (country or "").lower()
    mask_country = df[cr].str.lower().eq(country_lower)
    mask_global  = df[cr].str.lower().isin(["global","world","any","all"])
    filtered = df[mask_country | mask_global].copy()
    if filtered.empty:
        filtered = df.copy()
    # Standardize output columns
    out = pd.DataFrame({
        "food":   filtered[cf].astype(str).str.strip(),
        "crop":   filtered[cc].astype(str).str.strip(),
        "use":    filtered[cu].astype(str).str.strip(),
        "where":  filtered[cr].astype(str).str.strip(),
        "notes":  filtered[cn] if cn else "",
        "source": filtered[cs] if cs else "",
    })
    # Expand rows where crop contains pipe or comma separated list
    out = out.assign(crop=out["crop"].str.replace(";", ","))
    out = out.assign(crop_list=out["crop"].str.split(r"[|,]"))
    out = out.explode("crop_list")
    out["crop"] = out["crop_list"].str.strip()
    out = out.drop(columns=["crop_list"])
    out = out[out["crop"]!=""]
    return out.drop_duplicates().reset_index(drop=True)
def evidence_for_crops(crops_list, country):
    """Collect simple evidence slices for the chosen crops/country."""
    crops_list = sorted({c.strip().lower() for c in crops_list if isinstance(c, str) and c.strip()})
    if not crops_list:
        return {"policies": pd.DataFrame(), "bans": pd.DataFrame(), "studies": pd.DataFrame(), "incidents": pd.DataFrame()}
    pol = policies[policies["crop"].str.lower().isin(crops_list)].copy() if len(policies) and "crop" in policies.columns else pd.DataFrame()
    if len(pol) and "country" in pol.columns:
        pol_local = pol[pol["country"]==country].copy()
    else:
        pol_local = pd.DataFrame()
    bn = bans[bans["crop"].str.lower().isin(crops_list)].copy() if len(bans) and "crop" in bans.columns else pd.DataFrame()
    if len(bn) and "country" in bn.columns:
        bn_local = bn[bn["country"]==country].copy()
    else:
        bn_local = pd.DataFrame()
    stx = studies[studies["crop"].str.lower().isin(crops_list)].copy() if len(studies) and "crop" in studies.columns else pd.DataFrame()
    inc = incidents[incidents["crop"].str.lower().isin(crops_list)].copy() if len(incidents) and "crop" in incidents.columns else pd.DataFrame()
    return {"policies_all": pol, "policies_local": pol_local, "bans_all": bn, "bans_local": bn_local, "studies": stx, "incidents": inc}
def confidence_from_counts(counts_dict):
    """Very simple confidence heuristic based on how many independent tables have entries."""
    score = 0
    if not counts_dict: return "Low"
    for k,v in counts_dict.items():
        score += 1 if v>0 else 0
    if score >= 3:
        return "High"
    if score == 2:
        return "Medium"
    return "Low"

st.markdown("### Export")

# 1) filtered countries (from sidebar filters)
download_df_button(filtered, "Download filtered countries")


# 2) this country's bans/policies
country_bans = bans[bans["country"] == pick] if len(bans) else pd.DataFrame()
country_policies = policies[policies["country"] == pick] if len(policies) else pd.DataFrame()

c1, c2, c3 = st.columns(3)
with c1:
    download_df_button(country_bans, "Download this country's bans", f"{pick}_bans.csv")
with c2:
    download_df_button(country_policies, "Download this country's policies", f"{pick}_policies.csv")
with c3:
    country_incidents = incidents[incidents["country"] == pick] if len(incidents) else pd.DataFrame()
    download_df_button(country_incidents, "Download this country's incidents", f"{pick}_incidents.csv")   



    # Policies (approvals/decisions)
    st.markdown("### Regulatory decisions")
    p = policies[policies["country"] == pick].copy()
    if len(p):
        p = p[["crop","policy_type","decision","year","notes","source"]].sort_values("year", ascending=False)
        st.dataframe(p, use_container_width=True)
    else:
        st.info("No policies recorded for this country yet.")
# ---- Link health alert for the selected country ----
# This checks tools/output/link_report.csv (made by tools/check_links.py)
# and shows a red alert if any sources for this country are broken/blocked.

from pathlib import Path

@st.cache_data(show_spinner=False)
def load_link_report(report_path: Path) -> pd.DataFrame:
    if not report_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(report_path)
    # Normalize columns we rely on
    for col in ["file", "row", "row_idx", "context", "original_url", "final_url", "result", "content", "note", "suggestion"]:
        if col not in df.columns:
            df[col] = ""
    return df

def is_bad_result(result: str, content: str) -> bool:
    # Treat anything starting with ❌ as bad; also catch BLOCKED/ERROR/NOT_FOUND flags
    result = str(result or "")
    content = str(content or "").upper()
    if result.strip().startswith("❌"):
        return True
    return content in {"BLOCKED", "ERROR", "NOT_FOUND"}# --- Studies + Incidents + Companies tabs ---
st.subheader("Research & Context")
tab1, tab2, tab3 = st.tabs(["Studies", "Incidents", "Companies"])

with tab1:
    if len(studies):
        s = studies.copy()
        s = s[s["source"].notna()]
        st.dataframe(s[["study_id","crop","focus","findings","year","source"]].sort_values("year", ascending=False), use_container_width=True)
    else:
        st.info("No studies.csv found.")

with tab2:
    if len(incidents):
        i = incidents.copy()
        # If we want only incidents for picked country, uncomment:
        # i = i[i["country"] == pick]
        st.dataframe(i[["country","year","type","description","source"]].sort_values(["year","country"], ascending=[False, True]), use_container_width=True)
    else:
        st.info("No incidents.csv found.")

with tab3:
    if len(companies):
        c = companies.copy()
        st.dataframe(c[["company","product","crop","trait","year","notes","source"]].sort_values("year", ascending=False), use_container_width=True)
    else:
        st.info("No companies.csv found.")

# ---- Diet & Prediction (Beta) ----
st.markdown("### Diet & Prediction (Beta)")
if 'mapping' not in globals():
    mapping = pd.DataFrame()
country_map = get_country_mapping(mapping, pick) if not mapping.empty else pd.DataFrame()
if country_map.empty:
    st.info("No mapping_diet.csv (or mapping.csv) found or it has missing columns. Add a mapping file to enable the checklist (columns like food,crop,use,country_or_region,source).")
else:
    # Build a checklist of foods for this country (or global)
    foods_list = sorted(country_map["food"].dropna().unique().tolist())
    if not foods_list:
        st.info("We couldn't find any foods in mapping_diet.csv (or mapping.csv) for this country.")
    else:
        st.caption("Tick everything you ate today. This will map foods → crops and show evidence.")
        cols = st.columns(3)
        selected_foods = []
        for i, food in enumerate(foods_list):
            with cols[i % 3]:
                if st.checkbox(food, key=f"food_chk_{i}"):
                    selected_foods.append(food)
        if selected_foods:
            sel = country_map[country_map["food"].isin(selected_foods)].copy()
            # Collect crops from selection
            crops_sel = sel["crop"].dropna().str.lower().unique().tolist()
            ev = evidence_for_crops(crops_sel, pick)
            # Show a compact evidence summary per food
            for f in selected_foods:
                block = sel[sel["food"]==f]
                crops_for_food = sorted(block["crop"].unique().tolist())
                use_kinds = ", ".join(sorted(block["use"].str.lower().unique().tolist()))
                # counts for confidence
                counts = {
                    "policies_local": len(ev["policies_local"][ev["policies_local"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])]) if isinstance(ev["policies_local"], pd.DataFrame) and not ev["policies_local"].empty else 0,
                    "bans_local": len(ev["bans_local"][ev["bans_local"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])]) if isinstance(ev["bans_local"], pd.DataFrame) and not ev["bans_local"].empty else 0,
                    "studies": len(ev["studies"][ev["studies"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])]) if isinstance(ev["studies"], pd.DataFrame) and not ev["studies"].empty else 0,
                    "incidents": len(ev["incidents"][ev["incidents"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])]) if isinstance(ev["incidents"], pd.DataFrame) and not ev["incidents"].empty else 0,
                }
                conf = confidence_from_counts(counts)
                with st.expander(f"🍽️ {f} → {', '.join(crops_for_food)}  •  use: {use_kinds}  •  confidence: {conf}"):
                    # Sources from mapping rows
                    srcs = block["source"].dropna().unique().tolist()
                    if srcs:
                        st.markdown("**Mapping sources:**")
                        for surl in srcs[:10]:
                            st.markdown(f"- {surl}")
                    # Local policies/bans
                    if not ev["policies_local"].empty:
                        st.markdown("**Local policies:**")
                        st.dataframe(ev["policies_local"][ev["policies_local"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])][["crop","policy_type","decision","year","source"]].sort_values("year", ascending=False), use_container_width=True)
                    if not ev["bans_local"].empty:
                        st.markdown("**Local bans/moratoria:**")
                        st.dataframe(ev["bans_local"][ev["bans_local"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])][["crop","type","scope","year","source"]].sort_values("year", ascending=False), use_container_width=True)
                    # Studies & incidents (global)
                    if not ev["studies"].empty:
                        st.markdown("**Studies (global, filtered by crop):**")
                        st.dataframe(ev["studies"][ev["studies"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])][["study_id","crop","focus","findings","year","source"]].sort_values("year", ascending=False), use_container_width=True)
                    if not ev["incidents"].empty:
                        st.markdown("**Incidents (global, filtered by crop):**")
                        show_cols = [c for c in ["country","crop","year","type","description","source"] if c in ev["incidents"].columns]
                        st.dataframe(ev["incidents"][ev["incidents"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])][show_cols].sort_values(["year"], ascending=[False]), use_container_width=True)
            # Export a flat report
            report_rows = []
            for _, r in sel.iterrows():
                c = r["crop"]
                counts = {
                    "policies_local": len(ev["policies_local"][ev["policies_local"]["crop"].str.lower()==c.lower()]) if not ev["policies_local"].empty else 0,
                    "bans_local": len(ev["bans_local"][ev["bans_local"]["crop"].str.lower()==c.lower()]) if not ev["bans_local"].empty else 0,
                    "studies": len(ev["studies"][ev["studies"]["crop"].str.lower()==c.lower()]) if not ev["studies"].empty else 0,
                    "incidents": len(ev["incidents"][ev["incidents"]["crop"].str.lower()==c.lower()]) if not ev["incidents"].empty else 0,
                }
                report_rows.append({
                    "country": pick,
                    "food": r["food"],
                    "crop": c,
                    "use": r["use"],
                    "confidence": confidence_from_counts(counts),
                    "mapping_where": r["where"],
                    "mapping_source": r.get("source",""),
                })
            report_df = pd.DataFrame(report_rows)
            st.markdown("**Export this analysis**")
            download_df_button(report_df, "Download diet analysis", f"{pick}_diet_prediction.csv")
        else:
            st.caption("Pick one or more foods to see mapped crops and attached evidence.")