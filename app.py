import streamlit as st  # Streamlit — builds the web app interface
import pandas as pd  # Pandas — reads/manipulates the CSV data as tables (dataframes)
import matplotlib.pyplot as plt  # Matplotlib — generates all the charts/graphs

# --- PAGE SETUP ---
st.set_page_config(page_title="SCC Lost Heatmap", page_icon="📦", layout="wide")
st.title("SCC Lost Parcel Heatmap")
st.markdown("---")

# --- CONSTANTS ---
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]  # each station gets a unique colour
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]  # Amazon UK parcel size tiers
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # forces Mon→Sun order
SHIFT_ORDER = ["NS", "AM", "PM"]  # shift display order (Night Sort, AM, PM)
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen"}  # unique colour per shift

# Shift definitions — updated to match DRM2 operations:
# NS does stow, finishes by ~09:00
# AM does pick, stage, dispatch from 09:00
# PM starts at 14:00
SHIFT_DEFINITIONS = {
    "NS": "00:00 – 08:59 (Night Sort — stow)",
    "AM": "09:00 – 13:59 (AM — pick, stage, dispatch)",
    "PM": "14:00 – 23:59 (PM — dispatch, RELO)"
}

# Shift classification by hour — maps each hour (0-23) to a shift
# NS runs midnight to 08:59 (stow period, finishes ~09:00)
# AM runs 09:00 to 13:59 (pick + stage + morning dispatch)
# PM runs 14:00 to 23:59 (afternoon dispatch + RELO)
SHIFT_HOUR_MAP = {
    0: "NS", 1: "NS", 2: "NS", 3: "NS", 4: "NS",    # 00–04 = NS
    5: "NS", 6: "NS", 7: "NS", 8: "NS",               # 05–08 = NS (stow still happening)
    9: "AM", 10: "AM", 11: "AM", 12: "AM", 13: "AM",  # 09–13 = AM (pick, stage, dispatch)
    14: "PM", 15: "PM", 16: "PM", 17: "PM", 18: "PM", # 14–18 = PM
    19: "PM", 20: "PM", 21: "PM", 22: "PM", 23: "PM"  # 19–23 = PM
}

# Sensitive columns to auto-remove (personal data)
SENSITIVE_COLS = [
    "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
    "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
    "Payment Method", "District", "Scheduled Delivery End Time"
]

# Required columns for the app to function
REQUIRED_COLS = [
    "Tracking ID", "Sort Zone", "Aisle", "Cluster",
    "Package Length", "Package Width", "Package Height",
    "DSP Name", "Assigned Cycle", "Last Updated Time"
]

# Chart size constants (change here to resize ALL charts)
CHART_FULL = (7, 2.5)   # standard chart
CHART_SMALL = (6, 2)    # per-station smaller charts
CHART_WIDE = (8, 2.5)   # wide charts with many labels
DSP_NAME_MAX = 20       # truncate DSP names beyond this length


# --- HELPER FUNCTIONS ---

def get_size(longest):
    """Classify parcel by longest side (cm) into Amazon UK size tiers."""
    if pd.isna(longest):
        return "Unknown"
    elif longest <= 35:
        return "Small"
    elif longest <= 45:
        return "Medium"
    elif longest <= 61:
        return "Small Oversize"
    else:
        return "Large Oversize"


def classify_shift(hour):
    """Map an hour (0-23) to a shift name using SHIFT_HOUR_MAP. Returns 'Unknown' if NaN."""
    if pd.isna(hour):
        return "Unknown"
    return SHIFT_HOUR_MAP.get(int(hour), "Unknown")


def clean_data(df):
    """
    Full cleaning pipeline:
    1. Remove sensitive columns
    2. Fix package dimensions (strip "cm", convert to float)
    3. Calculate longest side + size category
    4. Parse timestamps + extract day of week
    5. Classify shift from Dispatch Time (fallback: Assigned Cycle)
    """
    # 1. Remove sensitive columns
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])

    # 2. Clean dimension columns
    for col in ["Package Length", "Package Width", "Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm", "").str.replace("cm", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Longest side + size
    dim_cols = ["Package Length", "Package Width", "Package Height"]
    if all(c in df.columns for c in dim_cols):
        df["Longest Side"] = df[dim_cols].max(axis=1)
    else:
        df["Longest Side"] = float("nan")
    df["Size Category"] = df["Longest Side"].apply(get_size)

    # 4. Timestamps + day of week
    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], errors="coerce")
        df["Day of Week"] = df["Last Updated Time"].dt.day_name()

    # 5. Shift classification
    if "Dispatch Time" in df.columns:
        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], errors="coerce")
        df["Dispatch Hour"] = df["Dispatch Time"].dt.hour
        df["Shift"] = df["Dispatch Hour"].apply(classify_shift)
    elif "Assigned Cycle" in df.columns:
        def cycle_to_shift(cycle):
            """Infer shift from cycle name text patterns."""
            if pd.isna(cycle):
                return "Unknown"
            c = str(cycle).upper().strip()
            if "NS" in c or "NIGHT" in c:
                return "NS"
            elif "PM" in c or "RELO" in c or "C2" in c:
                return "PM"
            elif "AM" in c or "C1" in c:
                return "AM"
            else:
                return "Unknown"
        df["Shift"] = df["Assigned Cycle"].apply(cycle_to_shift)
        df["Dispatch Hour"] = float("nan")
    else:
        df["Shift"] = "Unknown"
        df["Dispatch Hour"] = float("nan")

    return df


def get_station_name(df, filename):
    """Get station name from 'Station' column or filename fallback."""
    if "Station" in df.columns and len(df["Station"].dropna()) > 0:
        return df["Station"].dropna().iloc[0]
    return filename.replace(".csv", "").replace("_", " ").strip()[:20]


def get_date_range(df):
    """Return formatted date range string."""
    start = df["Last Updated Time"].min().strftime("%d %b %Y")
    end = df["Last Updated Time"].max().strftime("%d %b %Y")
    return start if start == end else f"{start} - {end}"


def safe_top(series, n=1):
    """Safely get most common value. Returns 'N/A' if empty."""
    counts = series.dropna().value_counts()
    if len(counts) == 0:
        return "N/A" if n == 1 else pd.Series(dtype="object")
    return counts.index[0] if n == 1 else counts.head(n)


def truncate_labels(labels, max_len=DSP_NAME_MAX):
    """Shorten labels to max_len chars + '...' to prevent chart overlap."""
    return [str(l)[:max_len] + "..." if len(str(l)) > max_len else str(l) for l in labels]


def plot_bar(data, xlabel, ylabel, title, color="steelblue", horizontal=False, figsize=CHART_FULL):
    """Reusable bar chart. Horizontal for rankings. Labels always rotation=0."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = truncate_labels(data.index)
    if horizontal:
        ax.barh(labels, data.values, color=color)
        ax.invert_yaxis()
    else:
        ax.bar(labels, data.values, color=color)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def plot_dsp(data, title, color="orange", figsize=CHART_FULL):
    """Horizontal bar chart for DSP names. Height auto-scales."""
    n_bars = len(data)
    auto_height = max(2, n_bars * 0.3)
    fig, ax = plt.subplots(figsize=(figsize[0], auto_height))
    labels = truncate_labels(data.index)
    ax.barh(labels, data.values, color=color)
    ax.invert_yaxis()
    ax.set_xlabel("Lost Parcels", fontsize=8)
    ax.set_ylabel("DSP", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def plot_shift(data, title, figsize=CHART_FULL):
    """
    Shift bar chart — ALWAYS shows all 3 shifts (NS, AM, PM) even if count is 0.
    Uses shift-specific colours for visual distinction.
    """
    # Force all 3 shifts to appear, fill missing with 0
    data = data.reindex(SHIFT_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [SHIFT_COLORS[s] for s in SHIFT_ORDER]  # consistent colours
    ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER], color=colors)
    ax.set_xlabel("Shift", fontsize=8)
    ax.set_ylabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def make_table(series, col1_name, col2_name):
    """Convert value_counts() into numbered display table."""
    tbl = series.reset_index()
    tbl.columns = [col1_name, col2_name]
    tbl.index = range(1, len(tbl) + 1)
    return tbl


def make_shift_leaderboard(df, total):
    """
    Creates shift leaderboard — ALWAYS shows all 3 shifts (NS, AM, PM).
    If a shift has 0 losts, it still appears in the table with 0 and 0%.
    Sorted worst (most losts) first for competition/accountability.
    """
    # Count losts per shift (exclude Unknown)
    shift_counts = df[df["Shift"] != "Unknown"]["Shift"].value_counts()

    # Build rows for ALL 3 shifts, even if count is 0
    rows = []
    for shift_name in SHIFT_ORDER:
        count = shift_counts.get(shift_name, 0)  # default to 0 if shift not in data
        pct = round(count / total * 100, 1) if total > 0 else 0
        rows.append({"Shift": shift_name, "Lost Parcels": int(count), "% of Total": f"{pct}%"})

    # Sort by count descending (worst first) for leaderboard ranking
    rows.sort(key=lambda r: r["Lost Parcels"], reverse=True)

    tbl = pd.DataFrame(rows)
    tbl.index = range(1, len(tbl) + 1)  # rank numbers starting from 1
    return tbl


def render_shift_tab(df, total, date_range_text, key_prefix=""):
    """
    Renders the full Shift tab content for one station:
    - Shift definitions (transparency)
    - Leaderboard (all 3 shifts always shown)
    - Bar chart (all 3 shifts always shown)
    - Per-shift parcel tables with dispatch times (accountability)
    """
    # --- SHIFT DEFINITIONS (full transparency on classification rules) ---
    st.subheader("📋 Shift Definitions")
    st.caption("Parcels assigned to shifts based on **Dispatch Time** (last scan before lost state).")

    # Display time windows clearly in 3 columns
    def_cols = st.columns(3)
    for idx, (shift, definition) in enumerate(SHIFT_DEFINITIONS.items()):
        with def_cols[idx]:
            st.markdown(f"**{shift}:** {definition}")

    st.markdown("---")

    # --- SHIFT LEADERBOARD (always shows all 3 shifts) ---
    shift_known = df[df["Shift"] != "Unknown"]  # rows with known shift

    st.subheader("🏆 Shift Leaderboard")
    st.caption("All shifts ranked by lost parcels. 0 means no losts assigned to that shift.")

    # Always show leaderboard (even if some shifts have 0)
    leaderboard = make_shift_leaderboard(df, total)
    st.dataframe(leaderboard, use_container_width=False)

    # Shift bar chart (always shows all 3 bars)
    if len(shift_known) > 0:
        shift_counts = shift_known["Shift"].value_counts()
    else:
        shift_counts = pd.Series(dtype="int64")  # empty series, plot_shift will fill with 0s
    st.pyplot(plot_shift(shift_counts, f"Lost Parcels by Shift ({date_range_text})"))

    st.markdown("---")

    # --- PER-SHIFT PARCEL TABLES (no plausible deniability) ---
    st.subheader("📦 Parcels Lost Per Shift")
    st.caption("Each shift's parcels listed with dispatch time for verification. "
               "Sorted by time so shift managers can cross-check their records.")

    # Loop through all 3 shifts in order (worst first based on leaderboard)
    for row in leaderboard.itertuples():
        shift_name = row.Shift  # current shift name
        count = row._2  # Lost Parcels column (position 2)
        pct_str = row._3  # % of Total string

        # Get parcels for this shift
        shift_df = df[df["Shift"] == shift_name].copy()

        # Always show the expander even if 0 parcels — makes it clear that shift had none
        with st.expander(f"**{shift_name} Shift** — {count} parcels ({pct_str}) | {SHIFT_DEFINITIONS[shift_name]}", expanded=False):
            if count == 0:
                st.success(f"✅ No parcels lost on {shift_name} shift.")
            else:
                # Columns to display (must include Dispatch Time for verification)
                display_cols = ["Tracking ID", "Dispatch Time", "Cluster", "Aisle",
                                "Sort Zone", "DSP Name", "Size Category", "Assigned Cycle"]
                display_cols = [c for c in display_cols if c in df.columns]

                # Sort by Dispatch Time ascending (earliest first for timeline clarity)
                if "Dispatch Time" in display_cols:
                    shift_df = shift_df.sort_values("Dispatch Time", ascending=True)

                # Format Dispatch Time nicely (DD/MM/YYYY HH:MM)
                if "Dispatch Time" in shift_df.columns:
                    shift_df["Dispatch Time"] = shift_df["Dispatch Time"].dt.strftime("%d/%m/%Y %H:%M")

                # Clean numbered table
                display_df = shift_df[display_cols].reset_index(drop=True)
                display_df.index = range(1, len(display_df) + 1)
                st.dataframe(display_df, use_container_width=True)

    # Warning for unknown parcels (couldn't be assigned to a shift)
    unknown_count = len(df[df["Shift"] == "Unknown"])
    if unknown_count > 0:
        st.warning(f"⚠️ {unknown_count} parcels could not be assigned to a shift "
                   f"(no Dispatch Time available). Excluded from shift rankings.")

    if len(shift_known) == 0:
        st.warning("Shift analysis requires 'Dispatch Time' column in your SCC export. "
                   "Add it to your SCC filters and re-export.")


# --- MODE TOGGLE ---
mode = st.radio("Mode:", ["Single Station", "Multi-Station Compare"], horizontal=True, key="mode_toggle")


# =====================================================================
# SINGLE STATION MODE
# =====================================================================
if mode == "Single Station":

    uploaded_file = st.file_uploader("Upload SCC export (.csv)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        found_sensitive = [c for c in SENSITIVE_COLS if c in df.columns]
        if found_sensitive:
            st.warning(f"Sensitive columns found and removed: {', '.join(found_sensitive)}")

        df = clean_data(df)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.info("Check your SCC export includes these fields, then re-upload.")
        elif len(df) == 0:
            st.warning("File has no data rows.")
        else:
            st.success(f"Data loaded — {len(df)} packages ready.")
            date_range_text = get_date_range(df)

            # --- SUMMARY METRICS (5 boxes) ---
            st.subheader(f"Quick Summary ({date_range_text})")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Lost", len(df))
            c2.metric("Worst Cluster", safe_top(df["Cluster"]))
            c3.metric("Worst Aisle", safe_top(df["Aisle"]))
            c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])
            shift_known = df[df["Shift"] != "Unknown"]["Shift"]
            c5.metric("Worst Shift", safe_top(shift_known) if len(shift_known) > 0 else "N/A")

            if len(df) < 5:
                st.info("Small dataset — consider uploading a full week.")

            # Cluster sweep priority
            cl_ranked = df["Cluster"].dropna().value_counts()
            if len(cl_ranked) > 0:
                sweep_parts = [f"#{i+1} {cl} ({n})" for i, (cl, n) in enumerate(cl_ranked.head(3).items())]
                st.info(f"🧹 **Sweep Priority:** {' → '.join(sweep_parts)}")

            # --- 8 TABS ---
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
                ["Overview", "Location", "Rankings", "DSP & Cycle", "Shift", "Time", "Export", "Bridge"]
            )

            # TAB 1: OVERVIEW
            with tab1:
                st.caption("Size breakdown and cluster summary.")
                view = st.radio("Display:", ["Table", "Chart"], horizontal=True, key="ov_view")
                if view == "Table":
                    st.subheader("Size Breakdown")
                    st.write(df["Size Category"].value_counts())
                    st.subheader("Cluster × Size")
                    tbl = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0)
                    tbl["Total"] = tbl.sum(axis=1)
                    st.dataframe(tbl)
                else:
                    size_counts = df["Size Category"].value_counts()
                    if len(size_counts) > 0:
                        colors = ["green", "orange", "red", "darkred", "grey"][:len(size_counts)]
                        st.pyplot(plot_bar(size_counts, "Size Category", "Lost Parcels",
                                           f"Lost by Size ({date_range_text})", color=colors))
                    cl_counts = df["Cluster"].dropna().value_counts()
                    if len(cl_counts) > 0:
                        st.pyplot(plot_bar(cl_counts, "Cluster", "Lost Parcels",
                                           f"Lost by Cluster ({date_range_text})"))

            # TAB 2: LOCATION
            with tab2:
                st.caption("Drill into a cluster to see aisle/zone hotspots.")
                clusters = sorted(df["Cluster"].dropna().unique())
                if clusters:
                    sel_cluster = st.selectbox("Cluster:", clusters, key="cl_sel")
                    filt = df[df["Cluster"] == sel_cluster]
                    st.write(f"{len(filt)} parcels in Cluster {sel_cluster}")
                    view_by = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="view_by")
                    loc_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="loc_view")
                    loc_data = filt[view_by].dropna().value_counts()
                    if loc_view == "Chart":
                        if len(loc_data) > 0:
                            st.pyplot(plot_bar(loc_data, view_by, "Lost Parcels",
                                               f"Cluster {sel_cluster} by {view_by}", figsize=CHART_WIDE))
                        st.subheader(f"Size by Aisle in Cluster {sel_cluster}")
                        sel_size = st.selectbox("Size:", sorted(df["Size Category"].dropna().unique()), key="sz_sel")
                        size_filt = filt[filt["Size Category"] == sel_size]
                        st.write(f"{len(size_filt)} '{sel_size}' parcels")
                        sz_data = size_filt["Aisle"].dropna().value_counts()
                        if len(sz_data) > 0:
                            st.pyplot(plot_bar(sz_data, "Aisle", "Lost Parcels",
                                               f"'{sel_size}' in Cluster {sel_cluster}", color="red"))
                        else:
                            st.info(f"No '{sel_size}' parcels in this cluster.")
                    else:
                        if len(loc_data) > 0:
                            st.dataframe(make_table(loc_data, "Location", "Lost Parcels"))
                        st.subheader("Size Breakdown")
                        sz_tbl = filt.groupby(["Aisle", "Size Category"]).size().unstack(fill_value=0)
                        sz_tbl["Total"] = sz_tbl.sum(axis=1)
                        st.dataframe(sz_tbl)
                else:
                    st.info("No cluster data available.")

            # TAB 3: RANKINGS
            with tab3:
                st.caption("Top 10 worst locations.")
                rank_by = st.selectbox("Rank by:", ["Sort Zone", "Aisle"], key="rank_sel")
                rank_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="rank_view")
                rank_data = df[rank_by].dropna().value_counts().head(10)
                if len(rank_data) > 0:
                    if rank_view == "Chart":
                        st.pyplot(plot_bar(rank_data, "Lost Parcels", rank_by,
                                           f"Top 10 {rank_by}s ({date_range_text})",
                                           color="darkred", horizontal=True))
                    else:
                        st.dataframe(make_table(rank_data, rank_by, "Lost Parcels"))
                else:
                    st.info("No data available.")

            # TAB 4: DSP & CYCLE
            with tab4:
                st.caption("DSP performance and cycle distribution.")
                dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dsp_view")
                dsp_data = df["DSP Name"].dropna().value_counts()
                cycle_data = df["Assigned Cycle"].dropna().value_counts()
                if dsp_view == "Chart":
                    if len(dsp_data) > 0:
                        st.pyplot(plot_dsp(dsp_data, f"Lost by DSP ({date_range_text})", color="orange"))
                    if len(cycle_data) > 0:
                        st.pyplot(plot_bar(cycle_data, "Cycle", "Lost Parcels",
                                           f"Lost by Cycle ({date_range_text})", color="purple"))
                else:
                    if len(dsp_data) > 0:
                        st.subheader("DSP (alphabetical)")
                        st.dataframe(make_table(dsp_data.sort_index(), "DSP", "Lost Parcels"))
                    if len(cycle_data) > 0:
                        st.subheader("Cycle")
                        st.dataframe(make_table(cycle_data, "Cycle", "Lost Parcels"))

            # TAB 5: SHIFT (dedicated tab)
            with tab5:
                render_shift_tab(df, len(df), date_range_text, key_prefix="single")

            # TAB 6: TIME
            with tab6:
                st.caption("Day-of-week patterns and tracking ID lookup.")
                day_data = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                time_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="time_view")
                if time_view == "Chart":
                    st.pyplot(plot_bar(day_data, "Day", "Lost Parcels",
                                       f"Lost by Day ({date_range_text})", color="green"))
                else:
                    st.dataframe(make_table(day_data, "Day", "Lost Parcels"))

                st.subheader("Parcel Details by Day")
                available_days = [d for d in DAY_ORDER if d in df["Day of Week"].values]
                if available_days:
                    sel_day = st.selectbox("Day:", available_days, key="day_sel")
                    day_df = df[df["Day of Week"] == sel_day]
                    st.write(f"{len(day_df)} parcels on {sel_day}")
                    show_cols = [c for c in ["Tracking ID", "Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category", "Shift"] if c in df.columns]
                    st.dataframe(day_df[show_cols])

            # TAB 7: EXPORT
            with tab7:
                st.caption("Download cleaned data.")
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "Lost_Parcels_Cleaned.csv", "text/csv")

            # TAB 8: BRIDGE
            with tab8:
                st.caption("Auto-generated bridge with dynamic action plans.")
                total = len(df)

                cl_counts = df["Cluster"].dropna().value_counts()
                ai_counts = df["Aisle"].dropna().value_counts()
                dsp_counts = df["DSP Name"].dropna().value_counts()
                sz_counts = df["Size Category"].value_counts()
                cy_counts = df["Assigned Cycle"].dropna().value_counts()
                day_counts = df["Day of Week"].dropna().value_counts()
                shift_counts_bridge = df[df["Shift"] != "Unknown"]["Shift"].value_counts()

                w_cluster = cl_counts.index[0] if len(cl_counts) > 0 else "N/A"
                w_cluster_n = cl_counts.values[0] if len(cl_counts) > 0 else 0
                w_cluster_pct = round(w_cluster_n / total * 100, 1) if total > 0 else 0
                w_aisle = ai_counts.index[0] if len(ai_counts) > 0 else "N/A"
                w_aisle_n = ai_counts.values[0] if len(ai_counts) > 0 else 0
                avg_aisle = ai_counts.mean() if len(ai_counts) > 0 else 1
                w_dsp = dsp_counts.index[0] if len(dsp_counts) > 0 else "N/A"
                w_dsp_n = dsp_counts.values[0] if len(dsp_counts) > 0 else 0
                avg_dsp = dsp_counts.mean() if len(dsp_counts) > 0 else 1
                dsp_mult = round(w_dsp_n / avg_dsp, 1) if avg_dsp > 0 else 1.0
                w_size = sz_counts.index[0] if len(sz_counts) > 0 else "N/A"
                w_size_n = sz_counts.values[0] if len(sz_counts) > 0 else 0
                top_cycle = cy_counts.index[0] if len(cy_counts) > 0 else "N/A"
                top_cycle_n = cy_counts.values[0] if len(cy_counts) > 0 else 0
                w_day = day_counts.index[0] if len(day_counts) > 0 else "N/A"
                w_day_n = day_counts.values[0] if len(day_counts) > 0 else 0
                w_shift = shift_counts_bridge.index[0] if len(shift_counts_bridge) > 0 else "N/A"
                w_shift_n = shift_counts_bridge.values[0] if len(shift_counts_bridge) > 0 else 0

                df["Date"] = df["Last Updated Time"].dt.strftime("%d/%m")
                daily = df.groupby("Date").size()
                daily_lines = "\n".join([f"{d} - {n} lost" for d, n in daily.items()])

                # Shift breakdown — show all 3 shifts
                shift_lines = ""
                for s in SHIFT_ORDER:
                    n = shift_counts_bridge.get(s, 0)
                    pct = round(n / total * 100, 1) if total > 0 else 0
                    shift_lines += f"  {s}: {n} ({pct}%)\n"

                cluster_details = ""
                for cl_name, cl_n in cl_counts.head(3).items():
                    pct = round(cl_n / total * 100, 1)
                    top_aisles = df[df["Cluster"] == cl_name]["Aisle"].dropna().value_counts().head(3)
                    aisles_str = ", ".join([f"{a} ({n})" for a, n in top_aisles.items()])
                    cluster_details += f"  Cluster {cl_name}: {cl_n} ({pct}%) — Aisles: {aisles_str}\n"

                dsp_lines = "\n".join([f"  {d}: {n} ({round(n/total*100,1)}%)" for d, n in dsp_counts.head(3).items()])
                size_lines = "\n".join([f"  {s}: {n}" for s, n in sz_counts.items()])

                if "State" in df.columns:
                    state_lines = "\n".join([f"  {s}: {n}" for s, n in df["State"].dropna().value_counts().items()])
                else:
                    state_lines = "  (Not in export)"

                actions = []
                ac = lambda txt: actions.append(f"AC{len(actions)+1}: {txt}")

                if w_cluster_pct > 40:
                    ac(f"Dedicated PS to Cluster {w_cluster} full shift — {w_cluster_pct}% of losts here.")
                else:
                    ac(f"PS rotation between top clusters ({', '.join([str(c) for c in cl_counts.head(3).index])}).")

                if dsp_mult >= 2.0:
                    ac(f"DSP {w_dsp} stand-down with leadership — {dsp_mult}x average.")
                elif dsp_mult >= 1.5:
                    ac(f"DSP {w_dsp} process briefing — {dsp_mult}x average.")
                else:
                    ac("Station-wide process refresher — losts spread evenly across DSPs.")

                if w_size in ["Large Oversize", "Small Oversize"]:
                    ac(f"Oversize stow audit in Aisle {w_aisle} — {w_size} most commonly lost.")
                elif w_size == "Small":
                    ac(f"Small parcel stow briefing — {w_size_n} small parcels lost.")
                else:
                    ac(f"Stow quality walk in Cluster {w_cluster} for {w_size} parcels.")

                if avg_aisle > 0 and w_aisle_n >= avg_aisle * 3:
                    ac(f"Physical inspection of Aisle {w_aisle} — {round(w_aisle_n/avg_aisle,1)}x average.")
                elif avg_aisle > 0 and w_aisle_n >= avg_aisle * 2:
                    ac(f"Increased PS in Aisle {w_aisle} — {w_aisle_n} losts, above average.")
                else:
                    ac("Daily PS huddle to review losts by aisle.")

                if len(shift_counts_bridge) > 1 and w_shift_n > total * 0.5:
                    ac(f"{w_shift} shift 5-whys — {w_shift_n} losts ({round(w_shift_n/total*100,1)}% of total).")

                relo_n = sum(cy_counts.get(c, 0) for c in cy_counts.index if "RELO" in str(c))
                if relo_n > total * 0.15:
                    ac(f"RELO process review — {relo_n} losts from relocation cycles.")

                if len(daily) > 1 and w_day_n > total * 0.3:
                    ac(f"Review {w_day} staffing — {w_day_n} losts ({round(w_day_n/total*100,1)}% of total).")

                bridge = f"""Lost Parcels Bridge - DRM2
{date_range_text}

Lost (Total): {total}

Daily Breakdown:
{daily_lines}

Shift Breakdown:
{shift_lines}
RC1) Location:
{cluster_details}
RC2) DSP:
{dsp_lines}

RC3) Size:
{size_lines}

RC4) Status:
{state_lines}

{chr(10).join(actions)}
"""
                st.subheader("Draft Bridge")
                st.text_area("Edit:", value=bridge, height=450, key="bridge_txt")

                st.subheader("Enhance with Quick")
                prompt = f"""Write a Lost Parcels bridge for DRM2. Use RC1-4 for root causes, AC1-4 for actions. Be specific.

Data ({date_range_text}): Total={total}, Cluster={w_cluster} ({w_cluster_n}, {w_cluster_pct}%), Aisle={w_aisle} ({w_aisle_n}), DSP={w_dsp} ({w_dsp_n}, {dsp_mult}x avg), Size={w_size} ({w_size_n}), Cycle={top_cycle} ({top_cycle_n}), Day={w_day} ({w_day_n}), Shift={w_shift} ({w_shift_n})
Daily: {daily_lines}
Shifts: {shift_lines}
Clusters: {cluster_details}DSPs: {dsp_lines}

Generate specific actions a station manager would implement. Reference clusters, aisles, DSPs, sizes, days, shifts.
"""
                st.code(prompt, language="text")
                st.info("Copy icon (top-right) → Quick → Ctrl+V")


# =====================================================================
# MULTI-STATION COMPARE MODE
# =====================================================================
else:
    st.subheader("Upload Station Data (2–5 stations)")
    st.caption("One SCC export per station. Station name auto-detected from 'Station' column.")

    num = st.slider("Stations to compare:", 2, 5, 2, key="num_st")

    uploaded = {}
    cols = st.columns(num)
    for i in range(num):
        with cols[i]:
            f = st.file_uploader(f"Station {i+1}", type="csv", key=f"st_up_{i}")
            if f:
                uploaded[i] = f

    if len(uploaded) >= 2:
        stations = {}
        names = []
        for i, file in uploaded.items():
            temp = clean_data(pd.read_csv(file))
            name = get_station_name(temp, file.name)
            stations[name] = temp
            names.append(name)

        st.success(f"Loaded: {', '.join(names)}")

        # Summary metrics per station
        st.subheader("Station Summary")
        mcols = st.columns(len(names))
        for idx, name in enumerate(names):
            shift_known = stations[name][stations[name]["Shift"] != "Unknown"]["Shift"]
            worst_shift = safe_top(shift_known) if len(shift_known) > 0 else "N/A"
            mcols[idx].metric(name, f"{len(stations[name])} lost")
            mcols[idx].caption(f"Worst shift: {worst_shift}")

        for name in names:
            cl_ranked = stations[name]["Cluster"].dropna().value_counts()
            if len(cl_ranked) > 0:
                sweep_parts = [f"#{i+1} {cl} ({n})" for i, (cl, n) in enumerate(cl_ranked.head(3).items())]
                st.info(f"🧹 **{name} Sweep Priority:** {' → '.join(sweep_parts)}")

        # --- 8 TABS ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            ["Overview", "Location", "Rankings", "DSP & Cycle", "Shift", "Time", "Export", "Bridge"]
        )

        # TAB 1: OVERVIEW COMPARE
        with tab1:
            st.caption("Compare totals, size, and clusters across stations.")
            view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_ov")
            if view == "Chart":
                st.subheader("Total Lost by Station")
                fig, ax = plt.subplots(figsize=CHART_FULL)
                ax.bar(names, [len(stations[n]) for n in names], color=STATION_COLORS[:len(names)])
                ax.set_ylabel("Lost Parcels", fontsize=8)
                ax.set_title("Total Lost Parcels by Station", fontsize=9)
                ax.tick_params(labelsize=7)
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Size Breakdown by Station")
                fig2, ax2 = plt.subplots(figsize=CHART_WIDE)
                x = range(len(SIZE_ORDER))
                w = 0.8 / len(names)
                for idx, name in enumerate(names):
                    counts = [len(stations[name][stations[name]["Size Category"] == s]) for s in SIZE_ORDER]
                    offset = (idx - len(names)/2 + 0.5) * w
                    ax2.bar([xi + offset for xi in x], counts, w, label=name, color=STATION_COLORS[idx])
                ax2.set_xticks(x)
                ax2.set_xticklabels(SIZE_ORDER, fontsize=7)
                ax2.set_ylabel("Lost Parcels", fontsize=8)
                ax2.tick_params(labelsize=7)
                ax2.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig2)

                st.subheader("Clusters per Station")
                for idx, name in enumerate(names):
                    cl = stations[name]["Cluster"].dropna().value_counts()
                    if len(cl) > 0:
                        st.pyplot(plot_bar(cl, "Cluster", "Lost", f"{name} — Clusters",
                                           color=STATION_COLORS[idx], figsize=CHART_SMALL))
            else:
                st.subheader("Totals")
                st.dataframe(pd.DataFrame({"Station": names, "Total Lost": [len(stations[n]) for n in names]},
                                          index=range(1, len(names)+1)))
                st.subheader("Size Breakdown")
                sz = {n: [len(stations[n][stations[n]["Size Category"] == s]) for s in SIZE_ORDER] for n in names}
                st.dataframe(pd.DataFrame(sz, index=SIZE_ORDER))
                st.subheader("Clusters")
                for name in names:
                    st.write(f"**{name}:**")
                    cl = stations[name]["Cluster"].dropna().value_counts()
                    if len(cl) > 0:
                        st.dataframe(make_table(cl, "Cluster", "Lost Parcels"))

        # TAB 2: LOCATION COMPARE
        with tab2:
            st.caption("Drill into each station's cluster hotspots.")
            for idx, name in enumerate(names):
                st.subheader(f"📍 {name}")
                sdf = stations[name]
                clusters = sorted(sdf["Cluster"].dropna().unique())
                if clusters:
                    sel = st.selectbox(f"Cluster ({name}):", clusters, key=f"mc_cl_{idx}")
                    filt = sdf[sdf["Cluster"] == sel]
                    st.write(f"{len(filt)} parcels in Cluster {sel}")
                    ai = filt["Aisle"].dropna().value_counts().head(10)
                    if len(ai) > 0:
                        st.pyplot(plot_bar(ai, "Aisle", "Lost", f"{name} — Cluster {sel} Aisles",
                                           color=STATION_COLORS[idx], figsize=CHART_SMALL))
                    zn = filt["Sort Zone"].dropna().value_counts().head(10)
                    if len(zn) > 0:
                        st.pyplot(plot_bar(zn, "Sort Zone", "Lost", f"{name} — Cluster {sel} Zones",
                                           color=STATION_COLORS[idx], figsize=CHART_SMALL))
                else:
                    st.info(f"No cluster data for {name}.")
                st.markdown("---")

        # TAB 3: RANKINGS COMPARE
        with tab3:
            st.caption("Top 10 worst locations per station.")
            rank_by = st.selectbox("Rank by:", ["Sort Zone", "Aisle"], key="mc_rank")
            rank_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_rank_v")
            for idx, name in enumerate(names):
                data = stations[name][rank_by].dropna().value_counts().head(10)
                if len(data) > 0:
                    if rank_view == "Chart":
                        st.pyplot(plot_bar(data, "Lost", rank_by, f"{name} — Top 10 {rank_by}s",
                                           color=STATION_COLORS[idx], horizontal=True, figsize=CHART_SMALL))
                    else:
                        st.subheader(name)
                        st.dataframe(make_table(data, rank_by, "Lost Parcels"))

        # TAB 4: DSP & CYCLE COMPARE
        with tab4:
            st.caption("DSP and cycle comparison across stations.")
            dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_dsp_v")
            if dsp_view == "Chart":
                for idx, name in enumerate(names):
                    dsp = stations[name]["DSP Name"].dropna().value_counts().head(10)
                    if len(dsp) > 0:
                        st.pyplot(plot_dsp(dsp, f"{name} — Worst DSPs",
                                           color=STATION_COLORS[idx], figsize=CHART_FULL))
                st.subheader("Cycle Comparison")
                all_cycles = sorted(set().union(*[stations[n]["Assigned Cycle"].dropna().unique() for n in names]))
                if all_cycles:
                    fig, ax = plt.subplots(figsize=CHART_FULL)
                    x = range(len(all_cycles))
                    w = 0.8 / len(names)
                    for idx, name in enumerate(names):
                        cy = stations[name]["Assigned Cycle"].dropna().value_counts()
                        counts = [cy.get(c, 0) for c in all_cycles]
                        offset = (idx - len(names)/2 + 0.5) * w
                        ax.bar([xi + offset for xi in x], counts, w, label=name, color=STATION_COLORS[idx])
                    ax.set_xticks(x)
                    ax.set_xticklabels(all_cycles, fontsize=7)
                    ax.set_ylabel("Lost Parcels", fontsize=8)
                    ax.set_title("Cycle Comparison — All Stations", fontsize=9)
                    ax.tick_params(labelsize=7)
                    ax.legend(fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                for idx, name in enumerate(names):
                    st.subheader(name)
                    dsp = stations[name]["DSP Name"].dropna().value_counts()
                    if len(dsp) > 0:
                        st.write("**DSPs (alphabetical):**")
                        st.dataframe(make_table(dsp.sort_index(), "DSP", "Lost Parcels"))
                    cy = stations[name]["Assigned Cycle"].dropna().value_counts()
                    if len(cy) > 0:
                        st.write("**Cycles:**")
                        st.dataframe(make_table(cy, "Cycle", "Lost Parcels"))
                    st.markdown("---")

        # TAB 5: SHIFT COMPARE (dedicated tab)
        with tab5:
            st.caption("Shift accountability — per station leaderboards and parcel-level verification.")

            # Grouped comparison chart at the top
            st.subheader("⏱️ Shift Comparison — All Stations")
            st.caption("**Definitions:** " +
                       " | ".join([f"**{s}:** {d}" for s, d in SHIFT_DEFINITIONS.items()]))

            # Grouped bar chart (all stations, all 3 shifts)
            fig_sh, ax_sh = plt.subplots(figsize=CHART_FULL)
            x = range(len(SHIFT_ORDER))
            w = 0.8 / len(names)
            for idx, name in enumerate(names):
                shift_data = stations[name][stations[name]["Shift"] != "Unknown"]["Shift"].value_counts()
                counts = [shift_data.get(s, 0) for s in SHIFT_ORDER]  # 0 if no data for that shift
                offset = (idx - len(names)/2 + 0.5) * w
                ax_sh.bar([xi + offset for xi in x], counts, w, label=name, color=STATION_COLORS[idx])
            ax_sh.set_xticks(x)
            ax_sh.set_xticklabels(SHIFT_ORDER, fontsize=8)
            ax_sh.set_ylabel("Lost Parcels", fontsize=8)
            ax_sh.set_title("Lost by Shift — Station Comparison", fontsize=9)
            ax_sh.tick_params(labelsize=7)
            ax_sh.legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig_sh)

            st.markdown("---")

            # Full shift tab per station (with parcel tables)
            for idx, name in enumerate(names):
                st.subheader(f"📍 {name}")
                sdf = stations[name]
                dr = get_date_range(sdf) if "Last Updated Time" in sdf.columns else ""
                render_shift_tab(sdf, len(sdf), dr, key_prefix=f"mc_{idx}")
                st.markdown("---")

        # TAB 6: TIME COMPARE
        with tab6:
            st.caption("Day-of-week patterns across stations.")
            time_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_time_v")
            if time_view == "Chart":
                fig, ax = plt.subplots(figsize=CHART_FULL)
                for idx, name in enumerate(names):
                    if "Day of Week" in stations[name].columns:
                        dd = stations[name]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        ax.plot(dd.index, dd.values, marker="o", label=name,
                                color=STATION_COLORS[idx], linewidth=2, markersize=4)
                ax.set_ylabel("Lost Parcels", fontsize=8)
                ax.set_title("Lost by Day — Station Comparison", fontsize=9)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig)
                for idx, name in enumerate(names):
                    if "Day of Week" in stations[name].columns:
                        dd = stations[name]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        st.pyplot(plot_bar(dd, "Day", "Lost", f"{name} — by Day",
                                           color=STATION_COLORS[idx], figsize=CHART_SMALL))
            else:
                st.subheader("Day of Week Comparison")
                day_all = {}
                for name in names:
                    if "Day of Week" in stations[name].columns:
                        day_all[name] = stations[name]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0).values
                st.dataframe(pd.DataFrame(day_all, index=DAY_ORDER))

        # TAB 7: EXPORT COMPARE
        with tab7:
            st.caption("Download individual or combined station data.")
            st.subheader("Individual Downloads")
            for name in names:
                st.download_button(f"Download {name}", stations[name].to_csv(index=False),
                                   f"Lost_{name}_Cleaned.csv", "text/csv", key=f"dl_{name}")
            st.subheader("Combined Download")
            combined = pd.concat([stations[n].assign(Station_Name=n) for n in names], ignore_index=True)
            st.download_button("Download All Stations Combined", combined.to_csv(index=False),
                               "Lost_All_Stations_Combined.csv", "text/csv", key="dl_all")

        # TAB 8: BRIDGE COMPARE
        with tab8:
            st.caption("Cross-station comparison summary.")
            st.subheader("Station Comparison")
            comp = []
            for name in names:
                sdf = stations[name]
                shift_known = sdf[sdf["Shift"] != "Unknown"]["Shift"]
                comp.append({
                    "Station": name,
                    "Total Lost": len(sdf),
                    "Worst Cluster": safe_top(sdf["Cluster"]),
                    "Worst Aisle": safe_top(sdf["Aisle"]),
                    "Worst DSP": safe_top(sdf["DSP Name"]),
                    "Worst Shift": safe_top(shift_known) if len(shift_known) > 0 else "N/A",
                    "Most Lost Size": safe_top(sdf["Size Category"]),
                    "Top Cycle": safe_top(sdf["Assigned Cycle"])
                })
            st.dataframe(pd.DataFrame(comp, index=range(1, len(comp)+1)))

            st.subheader("Key Differences")
            best = min(names, key=lambda n: len(stations[n]))
            worst = max(names, key=lambda n: len(stations[n]))
            gap = len(stations[worst]) - len(stations[best])
            gap_pct = round(gap / len(stations[worst]) * 100, 1) if len(stations[worst]) > 0 else 0
            st.write(f"**Best:** {best} ({len(stations[best])} losts)")
            st.write(f"**Worst:** {worst} ({len(stations[worst])} losts)")
            st.write(f"**Gap:** {gap} parcels ({gap_pct}% difference)")

            st.subheader("Enhance with Quick")
            summaries = "\n".join([f"- {n}: {len(stations[n])} losts, worst cluster={safe_top(stations[n]['Cluster'])}, worst DSP={safe_top(stations[n]['DSP Name'])}, worst shift={safe_top(stations[n][stations[n]['Shift']!='Unknown']['Shift']) if len(stations[n][stations[n]['Shift']!='Unknown']) > 0 else 'N/A'}" for n in names])
            compare_prompt = f"""Compare these stations' lost parcel performance and suggest what {worst} can learn from {best}:

{summaries}

Best: {best} ({len(stations[best])} losts)
Worst: {worst} ({len(stations[worst])} losts)
Gap: {gap} parcels

Generate specific recommendations for {worst}. Cover cluster management, DSP accountability, size handling, shift performance, and cycles.
"""
            st.code(compare_prompt, language="text")
            st.info("Copy icon → Quick → Ctrl+V")

    elif len(uploaded) == 1:
        st.warning("Upload at least 2 files to compare.")
    else:
        st.info("Upload CSV files above to begin.")
