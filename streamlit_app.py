import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard HP Bekas", layout="wide")

# =========================
# Konfigurasi sumber data
# =========================
DEFAULT_DATA_URL = "https://raw.githubusercontent.com/nanocassander-lab/HpBekasKu/main/hp_device_clean.csv"
DATA_URL = st.secrets.get("DATA_URL", os.getenv("DATA_URL", DEFAULT_DATA_URL))

st.title("📱 Dashboard Analitik HP Bekas")
st.caption(
    "Dataset: hasil cleaning dari hp_device_data.csv. "
    "Catatan: kolom harga bersifat *normalized* (bukan Rupiah)."
)

# =========================
# Loader data
# =========================
@st.cache_data(show_spinner=True)
def load_data_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def normalize_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["has_4g", "has_5g"]:
        if col in df.columns:
            if df[col].dtype != bool:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
                )
    return df

# Opsi upload file (kalau kamu belum upload file clean ke GitHub)
with st.expander("📥 Opsional: Upload file CSV clean (kalau tidak ingin pakai URL)", expanded=False):
    uploaded = st.file_uploader("Upload hp_device_clean.csv", type=["csv"])
    st.write("Atau pakai URL (default):", DATA_URL)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_data_from_url(DATA_URL)

df = normalize_bool_columns(df)

# =========================
# Validasi kolom yang diharapkan
# =========================
expected_cols = {
    "listing_id","brand","os","release_year","days_used","age_years","age_bucket",
    "has_4g","has_5g","screen_size","ram_gb","storage_gb",
    "rear_cam_mp","front_cam_mp","battery_mah","weight_g",
    "used_price_norm","new_price_norm","depreciation_norm","used_to_new_ratio"
}

missing = sorted(list(expected_cols - set(df.columns)))
if missing:
    st.error(
        "Kolom yang diharapkan tidak lengkap. "
        "Pastikan kamu memakai file hasil cleaning (hp_device_clean.csv).\n\n"
        f"Kolom yang hilang: {missing}"
    )
    st.stop()

# Pastikan tipe data numeric untuk kolom penting
num_cols = [
    "release_year","days_used","age_years","screen_size","ram_gb","storage_gb",
    "rear_cam_mp","front_cam_mp","battery_mah","weight_g",
    "used_price_norm","new_price_norm","depreciation_norm","used_to_new_ratio"
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================
# Sidebar filter
# =========================
st.sidebar.header("🔎 Filter")

brands = sorted(df["brand"].dropna().unique())
selected_brands = st.sidebar.multiselect("Brand", brands)

os_list = sorted(df["os"].dropna().unique())
selected_os = st.sidebar.multiselect("OS", os_list)

min_year, max_year = int(df["release_year"].min()), int(df["release_year"].max())
year_range = st.sidebar.slider("Release Year", min_year, max_year, (min_year, max_year))

age_buckets = ["Semua"] + list(df["age_bucket"].dropna().astype(str).unique())
selected_age_bucket = st.sidebar.selectbox("Age Bucket", age_buckets, index=0)

def yn_filter(label: str, series: pd.Series):
    return st.sidebar.selectbox(label, ["Semua", "Ya", "Tidak"], index=0)

f_4g = yn_filter("4G", df["has_4g"])
f_5g = yn_filter("5G", df["has_5g"])

def range_slider(col: str, label: str, cast=float):
    lo = float(np.nanmin(df[col].values))
    hi = float(np.nanmax(df[col].values))
    # Untuk menghindari slider error jika lo == hi
    if np.isclose(lo, hi):
        return (cast(lo), cast(hi))
    return st.sidebar.slider(label, cast(lo), cast(hi), (cast(lo), cast(hi)))

ram_range = range_slider("ram_gb", "RAM (GB)", float)
storage_range = range_slider("storage_gb", "Storage (GB)", float)
screen_range = range_slider("screen_size", "Screen Size (inch)", float)
used_price_range = range_slider("used_price_norm", "Used Price (Normalized)", float)
days_used_range = range_slider("days_used", "Days Used", float)

# =========================
# Apply filters
# =========================
dff = df.copy()

if selected_brands:
    dff = dff[dff["brand"].isin(selected_brands)]

if selected_os:
    dff = dff[dff["os"].isin(selected_os)]

dff = dff[(dff["release_year"] >= year_range[0]) & (dff["release_year"] <= year_range[1])]
dff = dff[(dff["ram_gb"] >= ram_range[0]) & (dff["ram_gb"] <= ram_range[1])]
dff = dff[(dff["storage_gb"] >= storage_range[0]) & (dff["storage_gb"] <= storage_range[1])]
dff = dff[(dff["screen_size"] >= screen_range[0]) & (dff["screen_size"] <= screen_range[1])]
dff = dff[(dff["used_price_norm"] >= used_price_range[0]) & (dff["used_price_norm"] <= used_price_range[1])]
dff = dff[(dff["days_used"] >= days_used_range[0]) & (dff["days_used"] <= days_used_range[1])]

if selected_age_bucket != "Semua":
    dff = dff[dff["age_bucket"].astype(str) == selected_age_bucket]

if f_4g == "Ya":
    dff = dff[dff["has_4g"] == True]
elif f_4g == "Tidak":
    dff = dff[dff["has_4g"] == False]

if f_5g == "Ya":
    dff = dff[dff["has_5g"] == True]
elif f_5g == "Tidak":
    dff = dff[dff["has_5g"] == False]

# =========================
# Header metrics
# =========================
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Jumlah Listing", f"{len(dff):,}")
m2.metric("Median Used (Norm)", f"{dff['used_price_norm'].median():.4f}" if len(dff) else "-")
m3.metric("Median New (Norm)", f"{dff['new_price_norm'].median():.4f}" if len(dff) else "-")
m4.metric("Median Depreciation", f"{dff['depreciation_norm'].median():.4f}" if len(dff) else "-")
m5.metric("Median Used/New", f"{dff['used_to_new_ratio'].median():.4f}" if len(dff) else "-")

if len(dff) == 0:
    st.warning("Filter terlalu ketat. Longgarkan filter agar data muncul.")
    st.stop()

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📌 Overview", "💰 Pricing", "⚙️ Specs", "📄 Data"])

with tab1:
    colA, colB = st.columns(2)

    with colA:
        top_n = st.slider("Top N brand (berdasarkan jumlah listing)", 5, 25, 10)
        brand_counts = dff["brand"].value_counts().head(top_n).reset_index()
        brand_counts.columns = ["brand", "count"]
        fig = px.bar(brand_counts, x="brand", y="count", title=f"Top {top_n} Brand (Jumlah Listing)")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig2 = px.histogram(dff, x="used_price_norm", nbins=40, title="Distribusi Used Price (Normalized)")
        st.plotly_chart(fig2, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        fig3 = px.scatter(
            dff,
            x="age_years",
            y="used_price_norm",
            color="brand",
            hover_data=["os","release_year","ram_gb","storage_gb","has_5g"],
            title="Age (tahun) vs Used Price (Normalized)"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with colD:
        # 5G vs price
        tmp = dff.copy()
        tmp["has_5g"] = tmp["has_5g"].map({True: "5G", False: "Non-5G"})
        fig4 = px.box(tmp, x="has_5g", y="used_price_norm", title="Used Price: 5G vs Non-5G")
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    colA, colB = st.columns(2)
    with colA:
        # Used vs New
        fig = px.scatter(
            dff,
            x="new_price_norm",
            y="used_price_norm",
            color="brand",
            hover_data=["os","release_year","ram_gb","storage_gb"],
            title="New Price (Norm) vs Used Price (Norm)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        # Depreciation distribution
        fig2 = px.histogram(dff, x="depreciation_norm", nbins=40, title="Distribusi Depreciation (Norm)")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 (berdasarkan harga & depresiasi)")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.caption("Top 10 Used Price tertinggi")
        st.dataframe(
            dff.sort_values("used_price_norm", ascending=False)[
                ["listing_id","brand","os","release_year","ram_gb","storage_gb","used_price_norm"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

    with c2:
        st.caption("Top 10 New Price tertinggi")
        st.dataframe(
            dff.sort_values("new_price_norm", ascending=False)[
                ["listing_id","brand","os","release_year","ram_gb","storage_gb","new_price_norm"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

    with c3:
        st.caption("Top 10 Depreciation tertinggi")
        st.dataframe(
            dff.sort_values("depreciation_norm", ascending=False)[
                ["listing_id","brand","os","release_year","days_used","depreciation_norm","used_to_new_ratio"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

with tab3:
    colA, colB = st.columns(2)

    with colA:
        fig = px.scatter(
            dff,
            x="storage_gb",
            y="used_price_norm",
            color="brand",
            hover_data=["ram_gb","screen_size","rear_cam_mp","battery_mah"],
            title="Storage (GB) vs Used Price (Norm)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig2 = px.scatter(
            dff,
            x="ram_gb",
            y="used_price_norm",
            color="brand",
            hover_data=["storage_gb","screen_size","rear_cam_mp","battery_mah"],
            title="RAM (GB) vs Used Price (Norm)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Korelasi (numeric)")
    corr_cols = [
        "screen_size","ram_gb","storage_gb","rear_cam_mp","front_cam_mp",
        "battery_mah","weight_g","days_used","used_price_norm","new_price_norm","depreciation_norm","used_to_new_ratio"
    ]
    corr = dff[corr_cols].corr(numeric_only=True)
    fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Data setelah filter")

    # Download filtered data
    csv_buf = io.StringIO()
    dff.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download data (CSV) hasil filter",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="hp_device_filtered.csv",
        mime="text/csv",
    )

    st.dataframe(dff, use_container_width=True, hide_index=True)

with st.expander("📚 Data Dictionary (kolom)", expanded=False):
    st.markdown(
        """
- `listing_id`: ID buatan (1..N) untuk memudahkan dashboard
- `brand`: merk device
- `os`: sistem operasi
- `release_year`: tahun rilis
- `days_used`: lama pemakaian (hari)
- `age_years`: umur pemakaian (tahun)
- `age_bucket`: kategori umur
- `has_4g`, `has_5g`: dukungan 4G/5G
- `screen_size`: ukuran layar (inch)
- `ram_gb`, `storage_gb`: RAM & storage
- `rear_cam_mp`, `front_cam_mp`: kamera (MP)
- `battery_mah`: kapasitas baterai (mAh)
- `weight_g`: berat (gram)
- `used_price_norm`, `new_price_norm`: harga normalized (bukan Rupiah)
- `depreciation_norm`: penurunan harga normalized (new - used)
- `used_to_new_ratio`: rasio used/new
"""
    )
