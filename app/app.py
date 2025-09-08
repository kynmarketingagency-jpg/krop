# Streamlit app for Krop — Phase 2A (Data Explorer + Map)
# Friendly teacher comments inside 🌱

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import re

# --- Visual assets path and background injector ---
import os, base64
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

def add_static_graffiti_bg(image_path):
    """
    Set a fixed graffiti-style background using CSS. The image is read and inlined as Base64 so
    it works on Streamlit Cloud without extra static hosting.
    """
    try:
        p = Path(image_path)
        if not p.exists():
            # Try assets folder relative to repo root
            p = ASSETS / str(image_path)
        if not p.exists():
            st.warning(f"Background not found: {image_path}")
            return
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{b64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                background-repeat: no-repeat;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                inset: 0;
                background: rgba(0,0,0,0.35); /* keep content readable */
                pointer-events: none;
                z-index: 0;
            }}
            .block-container {{ position: relative; z-index: 1; }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Couldn't set background: {e}")

# --- Vector line-art background (urban hieroglyphic vibe) ---
def add_urban_hiero_bg():
    """
    Inject a lightweight, repeating SVG pattern as a background.
    This avoids heavy image files and keeps lines crisp at any screen size.
    The pattern is monochrome (off-white lines on near-black) so content stays readable.
    """
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220' viewBox='0 0 220 220'>"
        "<rect width='100%' height='100%' fill='black'/>"
        # grid of thin lines (subtle urban texture)
        "<g stroke='white' stroke-opacity='0.12' stroke-width='1'>"
        "<path d='M0 20 H220 M0 60 H220 M0 100 H220 M0 140 H220 M0 180 H220'/>"
        "<path d='M20 0 V220 M60 0 V220 M100 0 V220 M140 0 V220 M180 0 V220'/>"
        "</g>"
        # line-style hieroglyphic/urban icons (no fills, only strokes)
        "<g stroke='white' stroke-opacity='0.55' stroke-width='2' fill='none' stroke-linecap='round' stroke-linejoin='round'>"
        # ankh-like symbol
        "<path d='M35 45 c0 -10 8 -18 18 -18 s18 8 18 18 c0 18 -36 18 -36 0 M53 45 V80'/>"
        # eye motif
        "<path d='M145 55 q-25 0 -45 18 q20 18 45 18 q25 0 45 -18 q-20 -18 -45 -18 z M145 73 a4 4 0 1 0 0.1 0'/>"
        # seated figure (abstract line style)
        "<path d='M30 150 q10 -20 30 -10 q18 8 18 26 v22 h-48 z M51 150 v-10 q0 -10 10 -10 h10'/>"
        # bird-outline glyph
        "<path d='M160 140 q20 -15 35 -5 q12 8 5 20 q-10 18 -40 10 q-10 15 -25 15 q15 -15 10 -25 q5 -5 15 -15 z'/>"
        # zigzag/lightning motif
        "<path d='M95 120 l20 -20 l-12 0 l18 -18 l-12 0 l20 -20'/>"
        # sun rays motif
        "<path d='M110 30 a12 12 0 1 1 0.1 0 M110 10 V0 M110 60 V50 M90 30 H80 M140 30 H130 M95 15 L88 8 M132 52 l-7 -7 M88 52 l7 -7 M132 8 l-7 7'/>"
        "</g>"
        "</svg>"
    )
    import base64
    b64 = base64.b64encode(svg.encode('utf-8')).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/svg+xml;base64,{b64}");
            background-size: 420px 420px;   /* pattern scale */
            background-color: #000;         /* fallback */
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed; inset: 0;
            background: rgba(0,0,0,0.25);   /* soften contrast for readability */
            pointer-events: none; z-index: 0;
        }}
        .block-container {{ position: relative; z-index: 1; }}
        .stApp, .stApp * {{ color: #f5f5f5; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- robust CSV reader: tolerate extra commas in notes so every row has 6 fields ---
def read_mapping_loose(path: Path):
    import csv
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for i, row in enumerate(r, start=2):  # header is line 1
            if not row:
                continue
            # Ensure exactly 6 fields: food,crop,use,country_or_region,notes,source
            if len(row) < 6:
                # skip malformed short rows
                continue
            if len(row) > 6:
                # merge middle cells into the notes column
                row = [row[0], row[1], row[2], row[3], ",".join(row[4:-1]), row[-1]]
            rows.append(row)
    return pd.DataFrame(rows, columns=["food","crop","use","country_or_region","notes","source"])


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
mapping_name = None
if _mapping_diet.exists():
    mapping = read_mapping_loose(_mapping_diet)
    mapping_name = "mapping_diet.csv"
elif _mapping_legacy.exists():
    mapping = read_mapping_loose(_mapping_legacy)
    mapping_name = "mapping.csv"
else:
    mapping = pd.DataFrame(columns=["food","crop","use","country_or_region","notes","source"])
    mapping_name = "none"

# Optional: global issues library (crop_issue.csv)
crop_issues = (pd.read_csv(DATA / "crop_issue.csv") if (DATA / "crop_issue.csv").exists() else pd.DataFrame())

# --- Page title ---
st.set_page_config(page_title="Krop Explorer", layout="wide")
st.title("Krop 🌾 — GMO Explorer (Beta)")
st.caption("Click a country to see stance, bans, approvals, studies, incidents. All rows link to sources.")

# --- Visual style (fixed) ---
# Temporarily lock the app to the Urban hieroglyphic line background and hide style options
add_urban_hiero_bg()
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

# ---- Extra helpers for curated meals & issues ----
def issues_for_crops(crops):
    """Return rows from crop_issue.csv whose crop list overlaps with provided crops."""
    if 'crop_issues' not in globals() or crop_issues is None or crop_issues.empty:
        return pd.DataFrame()
    want = {c.strip().lower() for c in crops if isinstance(c, str) and c.strip()}
    df = crop_issues.copy()
    if "crop" not in df.columns:
        return pd.DataFrame()
    def matches(row_crop):
        parts = [p.strip().lower() for p in str(row_crop).replace(";", "|").split("|") if p.strip()]
        return any(p in want for p in parts)
    mask = df["crop"].map(matches)
    cols = [c for c in ["crop","issue_type","description","country","year","source"] if c in df.columns]
    return df[mask][cols].sort_values(["year"], ascending=[False])

# Curated checklist for Nigeria (meal -> underlying GMO-relevant crops)
CURATED_NG = [
    ("Ogi / pap (akamu)", "maize"),
    ("Moi-moi (steamed bean pudding)", "cowpea"),
    ("Akara (bean fritter)", "cowpea"),
    ("Jollof rice (with vegetable oil)", "rice|soy|cottonseed|palm"),
    ("Fried rice (with chicken)", "rice|maize|soy"),
    ("Chicken stew", "maize|soy"),
    ("Egg omelette", "maize|soy"),
    ("Tomato stew (with vegetable oil)", "soy|cottonseed|palm"),
    ("Indomie noodles (with seasoning oil)", "wheat|soy|palm"),
    ("Suya (spiced beef)", "maize|soy"),
    ("Eba (cassava fufu)", "cassava"),
    ("Garri (cassava granules)", "cassava"),
    ("Yam porridge (with vegetable oil)", "soy|cottonseed|palm"),
    ("Meat pie (with filling)", "maize|soy"),
]

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
# Debug summary for mapping
with st.expander("⛏️ Mapping debug", expanded=False):
    st.write("Data folder:", str(DATA))
    st.write("Using file:", mapping_name)
    st.write("Total rows loaded:", 0 if mapping is None else len(mapping))
    st.write("Rows matching country/global:", 0 if country_map is None else len(country_map))
    if mapping_name != "none":
        st.write("Sample of loaded mapping:")
        st.dataframe(mapping.head(20), use_container_width=True)
if country_map.empty:
    st.info("No mapping_diet.csv (or mapping.csv) found or it has missing columns. Add a mapping file to enable the checklist (columns like food,crop,use,country_or_region,source).")
else:
    # Build a checklist of foods for this country (or global)
    st.caption(f"Mapping source: **{mapping_name}** • foods available for {pick}: {len(country_map)}")

    # Choose data source: curated list for Nigeria, otherwise mapping file
    use_curated = (pick.strip().lower() == "nigeria")
    meal_to_crops = {}
    if use_curated:
        for name, cropspec in CURATED_NG:
            meal_to_crops[name] = cropspec
        foods_list = list(meal_to_crops.keys())
    else:
        foods_list = sorted(country_map["food"].dropna().unique().tolist())
        for _, r in country_map.iterrows():
            meal_to_crops.setdefault(r["food"], r["crop"])

    if not foods_list:
        st.info("We couldn't find any foods for this country.")
    else:
        st.caption("Tick everything you ate today. This will map foods → crops and show evidence.")
        cols = st.columns(3)
        selected_foods = []
        for i, food in enumerate(foods_list):
            with cols[i % 3]:
                if st.checkbox(food, key=f"food_chk_{i}"):
                    selected_foods.append(food)

        # List of meals treated as human-only (curated for Nigeria)
        HUMAN_ONLY_MEALS = [
            "ogi / pap (akamu)",
            "moi-moi (steamed bean pudding)",
            "akara (bean fritter)",
            "jollof rice (with vegetable oil)",
            "tomato stew (with vegetable oil)",
            "indomie noodles (with seasoning oil)",
            "eba (cassava fufu)",
            "garri (cassava granules)",
            "yam porridge (with vegetable oil)",
            "meat pie (with filling)",
            "egg omelette",
        ]

        if selected_foods:
            # Aggregate crops across all selected foods (for quick export/evidence pulls)
            if use_curated:
                crops_sel = sorted({c for f in selected_foods for c in re.split(r"[|,]", meal_to_crops.get(f, "")) if c})
                sel = pd.DataFrame([{"food": f, "crop": c.strip(), "use": ("human" if f.lower() in HUMAN_ONLY_MEALS else "human|animal"), "where": "Nigeria", "source": ""} for f in selected_foods for c in re.split(r"[|,]", meal_to_crops.get(f, "")) if c.strip()])
            else:
                sel = country_map[country_map["food"].isin(selected_foods)].copy()
                crops_sel = sel["crop"].dropna().str.lower().unique().tolist()

            ev = evidence_for_crops([c.lower() for c in crops_sel], pick)

            # Show a compact evidence summary per food
            for f in selected_foods:
                if use_curated:
                    crops_for_food = [c.strip() for c in re.split(r"[|,]", meal_to_crops.get(f, "")) if c.strip()]
                    use_kinds = ("human" if f.lower() in HUMAN_ONLY_MEALS else "human|animal")
                    srcs = []
                else:
                    block = sel[sel["food"]==f]
                    crops_for_food = sorted(block["crop"].unique().tolist())
                    use_kinds = ", ".join(sorted(block["use"].str.lower().unique().tolist()))
                    srcs = block["source"].dropna().unique().tolist()

                # counts for confidence
                counts = {
                    "policies_local": len(ev["policies_local"][ev["policies_local"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])]) if isinstance(ev["policies_local"], pd.DataFrame) and not ev["policies_local"].empty else 0,
                    "bans_local": len(ev["bans_local"][ev["bans_local"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])]) if isinstance(ev["bans_local"], pd.DataFrame) and not ev["bans_local"].empty else 0,
                    "studies": len(ev["studies"][ev["studies"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])]) if isinstance(ev["studies"], pd.DataFrame) and not ev["studies"].empty else 0,
                    "incidents": len(ev["incidents"][ev["incidents"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])]) if isinstance(ev["incidents"], pd.DataFrame) and not ev["incidents"].empty else 0,
                }
                conf = confidence_from_counts(counts)

                with st.expander(f"🍽️ {f} → {', '.join(crops_for_food)}  •  use: {use_kinds}  •  confidence: {conf}"):
                    # Mapping sources (if any)
                    if srcs:
                        st.markdown("**Mapping sources:**")
                        for surl in srcs[:10]:
                            st.markdown(f"- {surl}")

                    # Local policies/bans
                    if not ev["policies_local"].empty:
                        st.markdown("**Local policies:**")
                        st.dataframe(
                            ev["policies_local"][ev["policies_local"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])][["crop","policy_type","decision","year","source","notes"] if "notes" in ev["policies_local"].columns else ["crop","policy_type","decision","year","source"]].sort_values("year", ascending=False),
                            use_container_width=True
                        )
                    if not ev["bans_local"].empty:
                        st.markdown("**Local bans/moratoria:**")
                        st.dataframe(
                            ev["bans_local"][ev["bans_local"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])][["crop","type","scope","year","source"]].sort_values("year", ascending=False),
                            use_container_width=True
                        )

                    # Issues observed in other countries (from crop_issue.csv)
                    iss = issues_for_crops(crops_for_food)
                    if not iss.empty:
                        st.markdown("**Issues observed in other countries:**")
                        st.dataframe(iss, use_container_width=True)

                    # Studies & incidents (global)
                    if not ev["studies"].empty:
                        st.markdown("**Studies (global, filtered by crop):**")
                        st.dataframe(
                            ev["studies"][ev["studies"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])][["study_id","crop","focus","findings","year","source"]].sort_values("year", ascending=False),
                            use_container_width=True
                        )
                    if not ev["incidents"].empty:
                        st.markdown("**Incidents (global, filtered by crop):**")
                        show_cols = [c for c in ["country","crop","year","type","description","source"] if c in ev["incidents"].columns]
                        st.dataframe(
                            ev["incidents"][ev["incidents"]["crop"].str.lower().isin([c.lower() for c in crops_for_food])][show_cols].sort_values(["year"], ascending=[False]),
                            use_container_width=True
                        )

            # Export a flat report
            report_rows = []
            for f in selected_foods:
                if use_curated:
                    for c in [x.strip() for x in re.split(r"[|,]", meal_to_crops.get(f, "")) if x.strip()]:
                        counts = {
                            "policies_local": len(ev["policies_local"][ev["policies_local"]["crop"].str.lower()==c.lower()]) if not ev["policies_local"].empty else 0,
                            "bans_local": len(ev["bans_local"][ev["bans_local"]["crop"].str.lower()==c.lower()]) if not ev["bans_local"].empty else 0,
                            "studies": len(ev["studies"][ev["studies"]["crop"].str.lower()==c.lower()]) if not ev["studies"].empty else 0,
                            "incidents": len(ev["incidents"][ev["incidents"]["crop"].str.lower()==c.lower()]) if not ev["incidents"].empty else 0,
                        }
                        report_rows.append({
                            "country": pick,
                            "food": f,
                            "crop": c,
                            "use": ("human" if f.lower() in HUMAN_ONLY_MEALS else "human|animal"),
                            "confidence": confidence_from_counts(counts),
                            "mapping_where": pick,
                            "mapping_source": "",
                        })
                else:
                    for _, r in sel[sel["food"]==f].iterrows():
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
            st.caption("Tick at least one meal to see mapped crops and evidence.")

# --- Tiny footer credit ---
st.markdown(
    """
    <style>
    .krop-footer{position:fixed;left:0;bottom:0;width:100%;text-align:center;font-size:12px;color:#6b7280;padding:6px 0;background:rgba(255,255,255,0.7);backdrop-filter:blur(4px);z-index:10000;}
    @media (max-width: 768px){ .krop-footer{font-size:11px;} }
    </style>
    <div class="krop-footer">Created by <strong>Ark Studios</strong> / <strong>KYN Studios</strong></div>
    """,
    unsafe_allow_html=True,
)