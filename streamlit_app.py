import os
import io
from typing import Dict, List, Tuple

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
# Helper: Content-based recommender (Cosine Similarity)
# =========================
FEATURE_LABELS: Dict[str, str] = {
    "release_year": "Release Year",
    "days_used": "Days Used",
    "age_years": "Age (Years)",
    "screen_size": "Screen Size (inch)",
    "ram_gb": "RAM (GB)",
    "storage_gb": "Storage (GB)",
    "rear_cam_mp": "Rear Camera (MP)",
    "front_cam_mp": "Front Camera (MP)",
    "battery_mah": "Battery (mAh)",
    "weight_g": "Weight (gram)",
    "used_price_norm": "Used Price (Normalized)",
    "new_price_norm": "New Price (Normalized)",
    "depreciation_norm": "Depreciation (Norm) = new - used",
    "used_to_new_ratio": "Used/New Ratio",
}

# Default fitur untuk rekomendasi (selaras dengan contoh pada laporan: RAM, Storage, Battery)
DEFAULT_RECO_FEATURES: List[str] = ["ram_gb", "storage_gb", "battery_mah"]


@st.cache_data(show_spinner=False)
def compute_minmax_params(df_: pd.DataFrame, features: Tuple[str, ...]) -> Dict[str, Tuple[float, float]]:
    """Hitung min-max untuk setiap fitur berdasarkan *dataset penuh* (lebih stabil meskipun filter berubah)."""
    params: Dict[str, Tuple[float, float]] = {}
    for f in features:
        s = pd.to_numeric(df_[f], errors="coerce")
        lo = float(np.nanmin(s.values)) if np.isfinite(np.nanmin(s.values)) else 0.0
        hi = float(np.nanmax(s.values)) if np.isfinite(np.nanmax(s.values)) else 1.0
        # fallback aman jika data aneh
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        params[f] = (lo, hi)
    return params


def minmax_scale_array(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = x.astype(float)
    if np.isclose(lo, hi):
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def build_weighted_matrix(
    dff_: pd.DataFrame,
    features: List[str],
    params: Dict[str, Tuple[float, float]],
    weights: Dict[str, float],
) -> np.ndarray:
    """Bangun matriks fitur ter-normalisasi (min-max) dan dibobot."""
    cols = []
    for f in features:
        lo, hi = params[f]
        v = pd.to_numeric(dff_[f], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        v_scaled = minmax_scale_array(v, lo, hi)
        cols.append(v_scaled * float(weights.get(f, 1.0)))
    X = np.vstack(cols).T  # (n, m)
    return X


def cosine_similarity(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Cosine similarity antara setiap baris X dan vektor u."""
    u = np.asarray(u, dtype=float)
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return np.zeros(X.shape[0], dtype=float)
    X_norm = np.linalg.norm(X, axis=1)
    denom = X_norm * u_norm
    sim = np.divide(X.dot(u), denom, out=np.zeros_like(X_norm, dtype=float), where=denom != 0)
    # Clamp untuk keamanan tampilan
    return np.clip(sim, -1.0, 1.0)

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
tab_reco, tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendation", "📌 Overview", "💰 Pricing", "⚙️ Specs", "📄 Data"])

# =========================
# Recommendation (Content-Based + Cosine Similarity)
# =========================
with tab_reco:
    st.subheader("🎯 Top-N Recommendation (Cosine Similarity)")
    st.caption(
        "Rekomendasi berbasis atribut (content-based). "
        "Fitur numerik akan di-*min-max normalize* (0-1), lalu dihitung kemiripannya "
        "menggunakan cosine similarity. Kandidat yang dipakai adalah *data setelah filter* di sidebar."
    )

    if len(dff) < 2:
        st.warning("Data hasil filter terlalu sedikit untuk rekomendasi. Coba longgarkan filter.")
    else:
        mode = st.radio(
            "Mode rekomendasi",
            [
                "Berdasarkan Preferensi (input)",
                "Berdasarkan Item (cari yang mirip dengan 1 listing)",
            ],
            horizontal=True,
        )

        feature_options = list(FEATURE_LABELS.keys())

        # Pakai min-max dari dataset penuh agar stabil meskipun filter berubah
        minmax_params = compute_minmax_params(df, tuple(feature_options))

        st.markdown("#### 1) Pilih fitur untuk menghitung kemiripan")
        features = st.multiselect(
            "Fitur numerik (untuk vektor content-based)",
            options=feature_options,
            default=DEFAULT_RECO_FEATURES,
            format_func=lambda x: FEATURE_LABELS.get(x, x),
        )

        if not features:
            st.info("Pilih minimal 1 fitur agar bisa menghitung cosine similarity.")
        else:
            st.markdown("#### 2) Preferensi & bobot")
            left, right = st.columns([2, 1])

            weights: Dict[str, float] = {}
            user_values: Dict[str, float] = {}

            with left:
                if mode == "Berdasarkan Preferensi (input)":
                    st.write("Masukkan nilai preferensi untuk tiap fitur yang dipilih.")
                    for f in features:
                        lo, hi = minmax_params[f]
                        # default: median dari dataset penuh
                        default_val = float(pd.to_numeric(df[f], errors="coerce").median())
                        # langkah (step) sederhana agar input nyaman
                        span = float(hi - lo)
                        step = span / 100.0 if span > 0 else 0.1
                        step = 0.1 if step <= 0 else step
                        user_values[f] = st.number_input(
                            f"Preferensi: {FEATURE_LABELS.get(f, f)}",
                            min_value=float(lo),
                            max_value=float(hi),
                            value=float(np.clip(default_val, lo, hi)),
                            step=float(step),
                            key=f"pref_{f}",
                        )

                else:
                    # Mode item-to-item: pilih satu listing sebagai query
                    st.write("Pilih 1 listing sebagai *query item* (akan dicarikan yang paling mirip).")
                    # buat label ringkas agar mudah dicari di selectbox
                    dff_small = dff[[
                        "listing_id",
                        "brand",
                        "os",
                        "release_year",
                        "ram_gb",
                        "storage_gb",
                        "used_price_norm",
                    ]].copy()
                    dff_small["label"] = (
                        dff_small["listing_id"].astype(str)
                        + " | "
                        + dff_small["brand"].astype(str)
                        + " | "
                        + dff_small["os"].astype(str)
                        + " | "
                        + dff_small["release_year"].astype(int).astype(str)
                        + " | RAM "
                        + dff_small["ram_gb"].round(1).astype(str)
                        + "GB | Storage "
                        + dff_small["storage_gb"].round(0).astype(int).astype(str)
                        + "GB | Used "
                        + dff_small["used_price_norm"].round(2).astype(str)
                    )
                    label_to_id = dict(zip(dff_small["label"], dff_small["listing_id"]))
                    selected_label = st.selectbox(
                        "Pilih listing",
                        options=list(label_to_id.keys()),
                        key="query_listing",
                    )
                    query_id = int(label_to_id[selected_label])
                    query_row = dff[dff["listing_id"] == query_id].iloc[0]
                    for f in features:
                        user_values[f] = float(query_row[f])

                st.caption("Tips: Anda bisa memperketat/longgarkan kandidat menggunakan filter sidebar (brand, OS, tahun, RAM, dll).")

            with right:
                st.write("Atur bobot fitur (opsional)")
                for f in features:
                    weights[f] = st.slider(
                        f"Bobot: {FEATURE_LABELS.get(f, f)}",
                        0.0,
                        5.0,
                        1.0,
                        0.1,
                        key=f"w_{f}",
                    )

            st.markdown("#### 3) Hitung Top-N")
            topn = st.slider("Top N rekomendasi", 3, 50, 10)
            use_value = st.checkbox(
                "Gabungkan dengan skor 'value for money' (1 - used_to_new_ratio)",
                value=True,
            )

            w_sim = 1.0
            if use_value:
                w_sim = st.slider("Bobot similarity (w_sim)", 0.0, 1.0, 0.8, 0.05)
            w_val = 1.0 - w_sim

            # Bangun matriks fitur kandidat (dff) + vektor user
            X = build_weighted_matrix(dff, features, minmax_params, weights)
            u_vec = []
            for f in features:
                lo, hi = minmax_params[f]
                u_scaled = minmax_scale_array(np.array([user_values[f]], dtype=float), lo, hi)[0]
                u_vec.append(u_scaled * float(weights.get(f, 1.0)))
            u = np.array(u_vec, dtype=float)

            sim = cosine_similarity(X, u)

            reco = dff.copy()
            reco["similarity"] = sim
            # value_score: semakin kecil used/new, semakin besar value
            reco["value_score"] = (1.0 - pd.to_numeric(reco["used_to_new_ratio"], errors="coerce")).clip(0.0, 1.0)

            if use_value:
                reco["final_score"] = (w_sim * reco["similarity"]) + (w_val * reco["value_score"])
            else:
                reco["final_score"] = reco["similarity"]

            # Jika mode item-to-item, jangan rekomendasikan item yang sama
            if mode == "Berdasarkan Item (cari yang mirip dengan 1 listing)":
                reco = reco.sort_values(["final_score"], ascending=False)
                # buang query_id jika ada
                try:
                    reco = reco[reco["listing_id"] != query_id]
                except Exception:
                    pass

            reco_top = reco.sort_values(["final_score"], ascending=False).head(topn)

            st.success(f"Menampilkan Top-{len(reco_top)} rekomendasi dari {len(dff):,} kandidat hasil filter.")

            show_cols = [
                "listing_id",
                "brand",
                "os",
                "release_year",
                "ram_gb",
                "storage_gb",
                "battery_mah",
                "screen_size",
                "used_price_norm",
                "new_price_norm",
                "depreciation_norm",
                "used_to_new_ratio",
                "similarity",
                "value_score",
                "final_score",
            ]
            show_cols = [c for c in show_cols if c in reco_top.columns]

            st.dataframe(
                reco_top[show_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Download hasil rekomendasi
            csv_buf = io.StringIO()
            reco_top[show_cols].to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download Top-N rekomendasi (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="hp_device_recommendation_topn.csv",
                mime="text/csv",
            )

            with st.expander("ℹ️ Detail metode", expanded=False):
                st.markdown(
                    """
**Representasi fitur**
- Setiap listing HP direpresentasikan sebagai vektor dari fitur numerik yang Anda pilih.
- Nilai fitur diubah menjadi skala 0–1 dengan *min-max normalization* berdasarkan dataset penuh.

**Cosine Similarity**
- Kemiripan dihitung dengan rumus:  
  \( \text{sim}(u, i) = \frac{u \cdot i}{\|u\| \; \|i\|} \)

**Skor akhir (opsional)**
- Jika Anda mengaktifkan opsi value, skor akhir dihitung:
  \( \text{final} = w_{sim} \cdot sim + (1-w_{sim}) \cdot (1-\text{used\_to\_new\_ratio}) \)
"""
                )

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
