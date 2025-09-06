# Streamlit app for Krop — Phase 2A (Data Explorer + Map)
# Friendly teacher comments inside 🌱

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import re

from urllib.parse import quote_plus
import requests, textwrap, json, time

def gh_headers():
    return {
        "Authorization": f"token {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "krop-vote"
    }

def create_vote_issue(country: str, cultivation: str, products: str) -> tuple[bool, str]:
    owner = st.secrets.get("GITHUB_OWNER", "kynmarketingagency-jpg")
    repo  = st.secrets.get("GITHUB_REPO", "krop")
    url   = f"https://api.github.com/repos/{owner}/{repo}/issues"
    title = f"[vote] {country} — cultivation:{cultivation} products:{products}"
    body  = textwrap.dedent(f"""
    **Auto vote from app**

    - Country: {country}
    - Cultivation: {cultivation}
    - Products: {products}
    """).strip()
    labels = ["vote", f"country:{country}", f"cultivation:{cultivation}", f"products:{products}"]
    r = requests.post(url, headers=gh_headers(), json={"title": title, "body": body, "labels": labels}, timeout=15)
    if r.status_code in (200,201): return True, r.json().get("html_url","")
    return False, f"{r.status_code}: {r.text[:200]}"

@st.cache_data(ttl=120)
def get_vote_counts(country: str):
    """Return counts dict for cultivation/products by label from GitHub Issues."""
    owner = st.secrets.get("GITHUB_OWNER", "kynmarketingagency-jpg")
    repo  = st.secrets.get("GITHUB_REPO", "krop")
    base  = f"https://api.github.com/repos/{owner}/{repo}/issues"
    def count(q):  # q is the label query
        params = {"state":"open", "labels": q, "per_page": 1}
        r = requests.get(base, headers=gh_headers(), params=params, timeout=15)
        # GitHub doesn’t give total in one shot; do a quick paged count (simple, small scale)
        if r.status_code!=200: return 0
        # Use search API for exact counts (faster)
        sr = requests.get("https://api.github.com/search/issues",
                          headers=gh_headers(),
                          params={"q": f"repo:{owner}/{repo} is:issue is:open label:{q}"}, timeout=15)
        if sr.status_code!=200: return 0
        return sr.json().get("total_count", 0)

    return {
        "cultivation": {
            "Yes": count(f"vote, country:{country}, cultivation:Yes"),
            "No": count(f"vote, country:{country}, cultivation:No"),
            "Unsure": count(f"vote, country:{country}, cultivation:Unsure"),
        },
        "products": {
            "Yes": count(f"vote, country:{country}, products:Yes"),
            "No": count(f"vote, country:{country}, products:No"),
            "Unsure": count(f"vote, country:{country}, products:Unsure"),
        }
    }

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

st.markdown("### Public vote (community pulse)")
col1, col2 = st.columns(2)
with col1:
    cultivation = st.radio("Allow GMO cultivation here?", ["Yes","No","Unsure"], horizontal=True, key="vote_cult")
with col2:
    products = st.radio("Allow GMO products in food?", ["Yes","No","Unsure"], horizontal=True, key="vote_prod")

if st.button("Submit vote"):
    ok, msg = create_vote_issue(pick, cultivation, products)
    if ok:
        st.success("Vote submitted. Thank you!")
        st.cache_data.clear()  # refresh counts
    else:
        st.error(f"Could not submit vote: {msg}")

counts = get_vote_counts(pick)
st.write("**Cultivation**:", counts["cultivation"])
st.write("**Products**:", counts["products"])

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