```python
import streamlit as st  # Streamlit — the framework that builds this web app
import pandas as pd  # Pandas — reads CSV files and lets us work with tables (dataframes)
import matplotlib.pyplot as plt  # Matplotlib — creates all the charts and graphs

# --- PAGE SETUP ---
st.set_page_config(page_title="DRM2 Lost Heatmap", page_icon="📦", layout="wide")  # wide layout, tab title, icon
st.title("📦 DRM2 Lost Parcel Heatmap")  # main page title
st.markdown("---")  # horizontal divider line

# --- CONSTANTS (edit these to change behaviour globally) ---
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]  # colours for multi-station charts
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]  # size tier order
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # weekday order
SHIFT_ORDER = ["NS", "AM", "PM"]  # shift order for charts/tables
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen"}  # one colour per shift

# Shift time windows (displayed so everyone knows the rules)
SHIFT_DEFINITIONS = {
    "NS": "00:00 – 08:59 (Night Sort — stow)",
    "AM": "09:00 – 13:59 (Pick, stage, dispatch)",
    "PM": "14:00 – 23:59 (Dispatch, RELO)"
}

# Maps each hour (0-23) to a shift
SHIFT_HOUR_MAP = {
    0: "NS", 1: "NS", 2: "NS", 3: "NS", 4: "NS",
    5: "NS", 6: "NS", 7: "NS", 8: "NS",
    9: "AM", 10: "AM", 11: "AM", 12: "AM", 13: "AM",
    14: "PM", 15: "PM", 16: "PM", 17: "PM", 18: "PM",
    19: "PM", 20: "PM", 21: "PM", 22: "PM", 23: "PM"
}

# Columns that contain personal data — removed on upload for GDPR
SENSITIVE_COLS = [
    "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
    "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
    "Payment Method", "District", "Scheduled Delivery End Time"
]

# Columns the app needs to work properly
REQUIRED_COLS = [
    "Tracking ID", "Sort Zone", "Aisle", "Cluster",
    "Package Length", "Package Width", "Package Height",
    "DSP Name", "Assigned Cycle", "Last Updated Time"
]

# Chart dimensions (width, height in inches) — kept small so page isn't cluttered
CHART = (7, 2.5)  # standard chart
CHART_SM = (6, 2)  # smaller chart for multi-station
DSP_MAX = 20  # max characters for DSP names on charts before truncating
LABEL_MAX = 25  # max characters for sort zone / location labels

# Columns to show in parcel detail tables (Reason included)
DETAIL_COLS = ["Tracking ID", "Cluster", "Aisle", "Sort Zone",
               "DSP Name", "Size Category", "Shift", "Reason"]


# =====================================================================
# HELPER FUNCTIONS (reusable building blocks)
# =====================================================================

def get_size(val):
    """Classify a parcel by its longest side (cm) into Amazon UK size tiers."""
    if pd.isna(val): return "Unknown"  # no measurement → unknown
    if val <= 35: return "Small"  # up to 35cm
    if val <= 45: return "Medium"  # 36-45cm
    if val <= 61: return "Small Oversize"  # 46-61cm
    return "Large Oversize"  # 62cm+


def classify_shift(hour):
    """Turn an hour (0-23) into a shift name. NaN → Unknown."""
    if pd.isna(hour): return "Unknown"  # no hour available
    return SHIFT_HOUR_MAP.get(int(hour), "Unknown")  # look up in the map


def clean_data(df):
    """Master cleaning function — runs everything needed on a raw SCC export."""
    # 1. Remove sensitive columns
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])

    # 2. Fix package dimensions (come as strings like "23.50 cm")
    for col in ["Package Length", "Package Width", "Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm", "").str.replace("cm", "")  # strip unit text
            df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to number, NaN if impossible

    # 3. Calculate longest side and assign size category
    dims = ["Package Length", "Package Width", "Package Height"]
    if all(c in df.columns for c in dims):
        df["Longest Side"] = df[dims].max(axis=1)  # longest of the 3 dimensions
    else:
        df["Longest Side"] = float("nan")  # dimensions missing
    df["Size Category"] = df["Longest Side"].apply(get_size)  # classify each parcel

    # 4. Parse timestamp and extract day of week
    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], errors="coerce")
        df["Day of Week"] = df["Last Updated Time"].dt.day_name()  # e.g. "Monday"

    # 5. Assign shift — primary: Dispatch Time hour, fallback: Assigned Cycle text
    if "Dispatch Time" in df.columns:
        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], errors="coerce")
        df["Shift"] = df["Dispatch Time"].dt.hour.apply(classify_shift)
    elif "Assigned Cycle" in df.columns:
        def cyc(c):
            """Fallback: guess shift from cycle name text."""
            if pd.isna(c): return "Unknown"
            u = str(c).upper()
            if "NS" in u or "NIGHT" in u: return "NS"
            if "PM" in u or "RELO" in u or "C2" in u: return "PM"
            if "AM" in u or "C1" in u: return "AM"
            return "Unknown"
        df["Shift"] = df["Assigned Cycle"].apply(cyc)
    else:
        df["Shift"] = "Unknown"  # neither column available

    return df


def get_station_name(df, filename):
    """Get station name from the Station column, or fall back to filename."""
    if "Station" in df.columns and len(df["Station"].dropna()) > 0:
        return df["Station"].dropna().iloc[0]  # first non-empty station value
    return filename.replace(".csv", "").replace("_", " ").strip()[:20]  # clean up filename


def get_date_range(df):
    """Return formatted date range string like '01 Jul 2026' or '01 Jul – 07 Jul 2026'."""
    if "Last Updated Time" not in df.columns: return ""
    valid = df["Last Updated Time"].dropna()
    if len(valid) == 0: return ""
    s = valid.min().strftime("%d %b %Y")  # start date
    e = valid.max().strftime("%d %b %Y")  # end date
    return s if s == e else f"{s} – {e}"  # single day or range


def safe_top(series):
    """Safely get the most common value in a series. Returns 'N/A' if empty."""
    c = series.dropna().value_counts()
    return c.index[0] if len(c) > 0 else "N/A"


def trunc(labels, max_len=LABEL_MAX):
    """Shorten labels for charts so they don't overlap. Adds '...' if cut."""
    return [str(l)[:max_len] + "..." if len(str(l)) > max_len else str(l) for l in labels]


def get_detail_cols(df, extra=None):
    """Get list of detail columns that exist in the dataframe."""
    base = list(DETAIL_COLS)  # copy the constant
    if extra:
        base = extra + [c for c in base if c not in extra]  # put extras first
    return [c for c in base if c in df.columns]  # only columns that exist


def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    """Horizontal bar chart — used for ALL rankings/locations/DSPs/reasons. Labels never overlap."""
    h = max(2, len(data) * 0.3)  # auto-scale height: taller if more bars
    fig, ax = plt.subplots(figsize=(figsize_width, h))
    labs = trunc(data.index, max_label)  # truncate long labels
    ax.barh(labs, data.values, color=color)  # horizontal bars
    ax.invert_yaxis()  # biggest value at top
    # Add values at end of each bar
    for i, v in enumerate(data.values):
        ax.text(v + 0.2, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def make_bar_vert(data, xl, yl, title, color="steelblue", figsize=CHART):
    """Vertical bar chart — used for Size, Cluster (short labels only)."""
    fig, ax = plt.subplots(figsize=figsize)
    labs = trunc(data.index, LABEL_MAX)
    ax.bar(labs, data.values, color=color)  # vertical bars
    # Add values above each bar
    for i, v in enumerate(data.values):
        ax.text(i, v + 0.2, str(int(v)), ha="center", fontsize=7)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def make_bar_shift(data, title):
    """Shift chart — always shows all 3 shifts (NS, AM, PM) even if 0."""
    data = data.reindex(SHIFT_ORDER, fill_value=0)  # ensure all 3 appear
    fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER],
                  color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])  # coloured by shift
    # Add values above bars
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.2, str(int(h)), ha="center", fontsize=7)
    ax.set_xlabel("Shift", fontsize=8)
    ax.set_ylabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def make_table(series, c1, c2):
    """Turn a value_counts series into a numbered display table."""
    t = series.reset_index()  # convert to dataframe
    t.columns = [c1, c2]  # rename columns
    t.index = range(1, len(t) + 1)  # number rows from 1
    return t


def shift_leaderboard(df, total):
    """Build shift leaderboard — always shows all 3 shifts, sorted worst first."""
    counts = df[df["Shift"] != "Unknown"]["Shift"].value_counts()  # count per shift
    rows = []
    for s in SHIFT_ORDER:
        n = int(counts.get(s, 0))  # parcels lost on this shift (0 if none)
        pct = round(n / total * 100, 1) if total > 0 else 0  # percentage
        rows.append({
            "Shift": s,
            "Lost Parcels": n,
            "% of Total": f"{pct}%",
            "Time Window": SHIFT_DEFINITIONS[s]  # show the hours so it's clear
        })
    rows.sort(key=lambda r: r["Lost Parcels"], reverse=True)  # worst first
    t = pd.DataFrame(rows)
    t.index = range(1, len(t) + 1)  # number from 1
    return t


def render_shift_tab(df, total, dr, key_prefix=""):
    """Render the full Shift Rankings tab content."""
    # Instructions
    st.info("💡 **What's here:** See which shift loses the most parcels. "
            "Expand a shift to see every lost parcel with its dispatch time, cluster, aisle, DSP, and **reason**.")

    # Show shift definitions so everyone knows the rules
    cols = st.columns(3)
    for i, (s, d) in enumerate(SHIFT_DEFINITIONS.items()):
        cols[i].markdown(f"**{s}:** {d}")
    st.markdown("---")

    # Leaderboard table (only 3 rows so give it full width and height)
    st.subheader("🏆 Shift Leaderboard")
    lb = shift_leaderboard(df, total)
    st.dataframe(lb, use_container_width=True, height=150)  # big enough to see all 3

    # Bar chart
    shift_data = df[df["Shift"] != "Unknown"]["Shift"].value_counts()
    if len(shift_data) > 0:
        st.pyplot(make_bar_shift(shift_data, f"Lost by Shift ({dr})"))
    st.markdown("---")

    # Reason breakdown per shift — shows ALL parcels, not just ones with a reason
    if "Reason" in df.columns:
        st.subheader("❓ Top Reasons by Shift")
        st.caption("Select a shift to see why every parcel was lost.")
        shift_pick = st.selectbox("Select Shift:", SHIFT_ORDER, key=f"{key_prefix}reason_shift_pick")
        s_df = df[df["Shift"] == shift_pick].copy()  # ALL parcels on this shift
        if len(s_df) > 0:
            s_df["Reason"] = s_df["Reason"].fillna("No Reason Recorded")  # fill blanks so nothing is hidden
            st.markdown(f"**{shift_pick}** — **{len(s_df)} parcels total**")
            # Count table showing every reason (ranked most common first)
            rc = s_df["Reason"].value_counts()
            tbl = rc.reset_index()
            tbl.columns = ["Reason", "Count"]
            tbl.index = range(1, len(tbl) + 1)
            st.dataframe(tbl, use_container_width=True)
            # Expandable list per reason showing every parcel
            st.caption("Expand a reason below to see every parcel:")
            for reason in rc.index:
                reason_df = s_df[s_df["Reason"] == reason]
                show_cols = [c for c in ["Tracking ID", "Cluster", "Aisle",
                             "Sort Zone", "DSP Name", "Size Category"] if c in s_df.columns]
                with st.expander(f"**{reason}** — {len(reason_df)} parcels"):
                    out = reason_df[show_cols].sort_values("DSP Name").reset_index(drop=True)
                    out.index = range(1, len(out) + 1)
                    st.dataframe(out, use_container_width=True)
        else:
            st.info(f"No parcels on {shift_pick} shift.")
        st.markdown("---")

    # Per-shift parcel detail (expandable — keeps page clean)
    st.subheader("📦 Parcels Per Shift")
    st.caption("Expand a shift below to see every parcel for verification.")
    for row in lb.itertuples():
        s_name = row.Shift  # shift name
        count = row._2  # lost parcels count
        pct_str = row._3  # percentage string
        shift_df = df[df["Shift"] == s_name].copy()  # filter to this shift
        with st.expander(f"**{s_name}** — {count} parcels ({pct_str})"):
            if count == 0:
                st.success(f"✅ No parcels lost on {s_name} shift.")
            else:
                # Show columns that exist, sorted by DSP alphabetically
                cols_show = [c for c in ["Tracking ID", "Dispatch Time", "Cluster",
                             "Aisle", "Sort Zone", "DSP Name", "Size Category", "Reason"] if c in df.columns]
                if "Dispatch Time" in cols_show:
                    shift_df["Dispatch Time"] = shift_df["Dispatch Time"].dt.strftime("%d/%m/%Y %H:%M")
                out = shift_df[cols_show].sort_values("DSP Name").reset_index(drop=True)  # alphabetical by DSP
                out.index = range(1, len(out) + 1)
                st.dataframe(out, use_container_width=True)

    # Warning if some parcels couldn't be assigned
    unk = len(df[df["Shift"] == "Unknown"])
    if unk > 0:
        st.warning(f"⚠️ {unk} parcels couldn't be assigned to a shift (no Dispatch Time). Excluded above.")


# =====================================================================
# MODE TOGGLE (top of page — single vs multi station)
# =====================================================================
mode = st.radio("Mode:", ["Single Station", "Multi-Station Compare"], horizontal=True, key="mode")

# --- HOW TO GET YOUR DATA (always available, collapsed by default) ---
with st.expander("📖 How to get your data (click if you need help)"):
    st.markdown("""
    **Step 1 — Get Tracker IDs from PerfectMile:**
    1. Open PerfectMile → Concessions Control Tower
    2. Go to the **L&U** tab → **Lost** bucket
    3. Copy all Tracker IDs

    **Step 2 — Export from SCC:**
    1. Open [SCC](https://logistics.amazon.co.uk/station/dashboard/outboundAMZL)
    2. Paste Tracker IDs into the search
    3. Apply any filters you need (zone, aisle, cluster, DSP, cycle)
    4. Click **Export → CSV**

    **Step 3 — Upload:**
    - Single Station → upload one CSV below
    - Multi-Station Compare → upload one CSV per station

    **Required columns:** Tracking ID, Sort Zone, Aisle, Cluster, Package Length/Width/Height, DSP Name, Assigned Cycle, Last Updated Time

    **Optional columns (recommended):** Reason, Dispatch Time, State
    """)


# =====================================================================
# SINGLE STATION MODE
# =====================================================================
if mode == "Single Station":

    uploaded_file = st.file_uploader("Upload your SCC export (.csv)", type="csv")  # file upload widget

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # read the CSV into a dataframe

        # Warn about sensitive columns being removed
        found = [c for c in SENSITIVE_COLS if c in df.columns]
        if found:
            st.warning(f"🔒 Sensitive columns auto-removed: {', '.join(found)}")

        df = clean_data(df)  # run full cleaning pipeline

        # Check required columns exist
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}. Check your SCC export.")
        elif len(df) == 0:
            st.warning("⚠️ File has no data rows.")
        else:
            st.success(f"✅ Loaded — **{len(df)} lost parcels**")
            dr = get_date_range(df)  # date range for chart titles

            # --- SUMMARY METRICS (always visible) ---
            st.subheader(f"Quick Summary ({dr})")
            c1, c2, c3, c4, c5 = st.columns(5)  # 5 metric boxes side by side
            c1.metric("Total Lost", len(df))
            c2.metric("Worst Cluster", safe_top(df["Cluster"]))
            c3.metric("Worst Aisle", safe_top(df["Aisle"]))
            c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])
            sk = df[df["Shift"] != "Unknown"]["Shift"]  # only parcels with known shift
            c5.metric("Worst Shift", safe_top(sk) if len(sk) > 0 else "N/A")

            # --- CLUSTER PRIORITY (which clusters to focus on first) ---
            cl_r = df["Cluster"].dropna().value_counts()  # count per cluster
            if len(cl_r) > 0:
                parts = []
                for i, (cl, n) in enumerate(cl_r.head(3).items()):
                    pct = round(n / len(df) * 100, 1)
                    parts.append(f"#{i+1} Cluster {cl} — {n} parcels ({pct}%)")
                st.info(f"🎯 **Cluster Priority (highest losts first):** " + " → ".join(parts))

            # --- TABS ---
            t1, t2, t3, t4, t5, t6, t7 = st.tabs([
                "📊 Summary",
                "📍 Lost Locations",
                "🚚 DSP",
                "⏱️ Shift Rankings",
                "📅 Day of Week",
                "💾 Export",
                "📋 Bridge"
            ])

            # ==============================================================
            # TAB 1: SUMMARY
            # ==============================================================
            with t1:
                st.info("💡 **What's here:** High-level breakdown by **size**, **cluster**, and "
                        "**reason** (why each parcel was lost). See the big picture before drilling down.")

                view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="sum_v")

                if view == "Chart":
                    # Size breakdown chart (vertical — labels are short)
                    sc = df["Size Category"].value_counts()
                    if len(sc) > 0:
                        colors = ["green", "orange", "red", "darkred", "grey"][:len(sc)]
                        st.pyplot(make_bar_vert(sc, "Size", "Lost Parcels",
                                               f"Lost by Size ({dr})", color=colors))

                    # Cluster breakdown (horizontal — handles any number of clusters)
                    cc = df["Cluster"].dropna().value_counts()
                    if len(cc) > 0:
                        st.pyplot(make_bar_horiz(cc, f"Lost by Cluster ({dr})"))

                    # Reason breakdown (horizontal — reason strings can be long)
                    if "Reason" in df.columns:
                        rc = df["Reason"].dropna().value_counts()
                        if len(rc) > 0:
                            st.pyplot(make_bar_horiz(rc, f"Lost by Reason ({dr})",
                                                    color="firebrick"))
                else:
                    # Size table
                    st.subheader("Size Breakdown")
                    st.dataframe(make_table(df["Size Category"].value_counts(), "Size", "Lost Parcels"))

                    # Cluster × Size pivot table
                    st.subheader("Cluster × Size")
                    pivot = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0)
                    pivot["Total"] = pivot.sum(axis=1)  # add total column
                    st.dataframe(pivot)

                    # Reason table
                    if "Reason" in df.columns:
                        st.subheader("Reason Breakdown")
                        st.dataframe(make_table(df["Reason"].dropna().value_counts(),
                                               "Reason", "Lost Parcels"),
                                     use_container_width=True)

            # ==============================================================
            # TAB 2: LOST LOCATIONS (merged Location + Rankings)
            # ==============================================================
            with t2:
                st.info("💡 **What's here:** Find exactly **where** parcels are being lost. "
                        "Choose a category (Cluster, Aisle, or Sort Zone) to see the **Top 10 worst** "
                        "locations, then pick a specific cluster to drill into its aisles and zones.")

                # --- TOP 10 RANKINGS SECTION ---
                st.subheader("🏆 Top 10 Worst Locations")
                rank_by = st.selectbox("Rank by:", ["Cluster", "Aisle", "Sort Zone"], key="rb")
                rank_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="rv")

                rank_data = df[rank_by].dropna().value_counts().head(10)  # top 10 worst
                if len(rank_data) > 0:
                    if rank_view == "Chart":
                        # Always horizontal — prevents Sort Zone labels from overlapping
                        st.pyplot(make_bar_horiz(rank_data,
                                                f"Top 10 {rank_by}s ({dr})",
                                                color="darkred"))
                    else:
                        st.dataframe(make_table(rank_data, rank_by, "Lost Parcels"))

                st.markdown("---")

                # --- DRILL-DOWN SECTION ---
                st.subheader("🔍 Cluster Drill-Down")
                st.caption("Pick a cluster to see which aisles and zones within it have the most losts.")
                clusters = sorted(df["Cluster"].dropna().unique())  # all unique clusters
                if clusters:
                    sel_cluster = st.selectbox("Select Cluster:", clusters, key="cl_sel")
                    filtered = df[df["Cluster"] == sel_cluster]  # only parcels from this cluster
                    st.write(f"**{len(filtered)} parcels** in Cluster {sel_cluster}")

                    drill_by = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="drill")
                    drill_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dv")

                    drill_data = filtered[drill_by].dropna().value_counts()
                    if len(drill_data) > 0:
                        if drill_view == "Chart":
                            # Horizontal bars — Sort Zones can be long, this prevents overlap
                            st.pyplot(make_bar_horiz(drill_data,
                                                    f"Cluster {sel_cluster} — {drill_by}s",
                                                    color="steelblue"))
                        else:
                            st.dataframe(make_table(drill_data, drill_by, "Lost Parcels"))

                    # Parcel detail table (grouped by DSP alphabetically, includes Reason)
                    with st.expander(f"📦 All parcels in Cluster {sel_cluster}"):
                        show_cols = get_detail_cols(filtered, extra=["Tracking ID", "Aisle", "Sort Zone"])
                        detail = filtered[show_cols].sort_values("DSP Name").reset_index(drop=True)
                        detail.index = range(1, len(detail) + 1)
                        st.dataframe(detail, use_container_width=True)

            # ==============================================================
            # TAB 3: DSP
            # ==============================================================
            with t3:
                st.info("💡 **What's here:** See which DSPs (delivery service partners) are losing "
                        "the most parcels. Chart view ranks **worst-first**. Table view is **alphabetical** "
                        "so you can quickly find a specific DSP.")

                dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dsp_v")
                dsp_data = df["DSP Name"].dropna().value_counts()  # count per DSP
                cycle_data = df["Assigned Cycle"].dropna().value_counts()  # count per cycle

                if dsp_view == "Chart":
                    if len(dsp_data) > 0:
                        # Horizontal bars — DSP names are long
                        st.pyplot(make_bar_horiz(dsp_data, f"Lost by DSP ({dr})",
                                                color="orange", max_label=DSP_MAX))
                    if len(cycle_data) > 0:
                        st.pyplot(make_bar_horiz(cycle_data, f"Lost by Cycle ({dr})",
                                                color="purple"))
                else:
                    if len(dsp_data) > 0:
                        st.subheader("DSP (A–Z)")
                        # Alphabetical table
                        dsp_alpha = dsp_data.sort_index()  # sort alphabetically
                        st.dataframe(make_table(dsp_alpha, "DSP", "Lost Parcels"),
                                     use_container_width=True)

                    if len(cycle_data) > 0:
                        st.subheader("Cycle")
                        st.dataframe(make_table(cycle_data, "Cycle", "Lost Parcels"))

                # Expandable: full parcel list grouped by DSP (includes Reason)
                with st.expander("📦 All parcels grouped by DSP (alphabetical)"):
                    show_cols = get_detail_cols(df, extra=["Tracking ID", "DSP Name"])
                    all_by_dsp = df[show_cols].sort_values("DSP Name").reset_index(drop=True)
                    all_by_dsp.index = range(1, len(all_by_dsp) + 1)
                    st.dataframe(all_by_dsp, use_container_width=True)

            # ==============================================================
            # TAB 4: SHIFT RANKINGS
            # ==============================================================
            with t4:
                render_shift_tab(df, len(df), dr)

            # ==============================================================
            # TAB 5: DAY OF WEEK
            # ==============================================================
            with t5:
                st.info("💡 **What's here:** See which **days of the week** have the most lost parcels. "
                        "The line graph shows the trend across the week with values on each point. "
                        "Expand 'Parcel Details' to see every lost parcel for a specific day.")

                day_data = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                day_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="day_v")

                if day_view == "Chart":
                    # Line graph with values shown on each point
                    fig, ax = plt.subplots(figsize=CHART)
                    ax.plot(day_data.index, day_data.values, marker="o",
                            color="green", linewidth=2, markersize=6)
                    # Add actual values above each data point
                    for i, (day, val) in enumerate(day_data.items()):
                        ax.annotate(str(int(val)),  # the number
                                    xy=(i, val),  # position of the point
                                    xytext=(0, 8),  # offset 8 pixels above
                                    textcoords="offset points",
                                    ha="center", fontsize=8, fontweight="bold")
                    ax.set_xlabel("Day", fontsize=8)
                    ax.set_ylabel("Lost Parcels", fontsize=8)
                    ax.set_title(f"Lost by Day of Week ({dr})", fontsize=9)
                    ax.tick_params(labelsize=7)
                    plt.xticks(rotation=0, ha="center")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.dataframe(make_table(day_data, "Day", "Lost Parcels"))

                # Expandable: parcels for a specific day (includes Reason)
                with st.expander("🔍 Parcel Details by Day"):
                    avail_days = [d for d in DAY_ORDER if d in df["Day of Week"].values]
                    if avail_days:
                        sel_day = st.selectbox("Select day:", avail_days, key="day_sel")
                        day_df = df[df["Day of Week"] == sel_day]
                        st.write(f"**{len(day_df)} parcels** on {sel_day}")
                        show_cols = get_detail_cols(day_df)
                        out = day_df[show_cols].sort_values("DSP Name").reset_index(drop=True)
                        out.index = range(1, len(out) + 1)
                        st.dataframe(out, use_container_width=True)

            # ==============================================================
            # TAB 6: EXPORT
            # ==============================================================
            with t6:
                st.info("💡 **What's here:** Download the cleaned dataset as a CSV. "
                        "Sensitive columns have been removed and size/shift columns have been added.")
                st.download_button("⬇️ Download Cleaned CSV", df.to_csv(index=False),
                                   "Lost_Parcels_Cleaned.csv", "text/csv")

            # ==============================================================
            # TAB 7: BRIDGE
            # ==============================================================
            with t7:
                st.info("💡 **What's here:** An auto-generated lost parcels bridge with root causes "
                        "and actions. Edit it directly below, or copy the prompt into Quick for an "
                        "AI-enhanced version.")

                total = len(df)

                # Gather all the stats needed for the bridge
                cl_c = df["Cluster"].dropna().value_counts()
                ai_c = df["Aisle"].dropna().value_counts()
                dsp_c = df["DSP Name"].dropna().value_counts()
                sz_c = df["Size Category"].value_counts()
                cy_c = df["Assigned Cycle"].dropna().value_counts()
                day_c = df["Day of Week"].dropna().value_counts()
                sh_c = df[df["Shift"] != "Unknown"]["Shift"].value_counts()
                # Reason counts
                rs_c = df["Reason"].dropna().value_counts() if "Reason" in df.columns else pd.Series(dtype="int64")

                # Extract worst values
                wc = cl_c.index[0] if len(cl_c) > 0 else "N/A"
                wc_n = int(cl_c.values[0]) if len(cl_c) > 0 else 0
                wc_p = round(wc_n / total * 100, 1) if total > 0 else 0
                wa = ai_c.index[0] if len(ai_c) > 0 else "N/A"
                wa_n = int(ai_c.values[0]) if len(ai_c) > 0 else 0
                avg_a = ai_c.mean() if len(ai_c) > 0 else 1
                wd = dsp_c.index[0] if len(dsp_c) > 0 else "N/A"
                wd_n = int(dsp_c.values[0]) if len(dsp_c) > 0 else 0
                avg_d = dsp_c.mean() if len(dsp_c) > 0 else 1
                dm = round(wd_n / avg_d, 1) if avg_d > 0 else 1.0
                ws = sz_c.index[0] if len(sz_c) > 0 else "N/A"
                ws_n = int(sz_c.values[0]) if len(sz_c) > 0 else 0
                wday = day_c.index[0] if len(day_c) > 0 else "N/A"
                wday_n = int(day_c.values[0]) if len(day_c) > 0 else 0
                wsh = sh_c.index[0] if len(sh_c) > 0 else "N/A"
                wsh_n = int(sh_c.values[0]) if len(sh_c) > 0 else 0
                wr = rs_c.index[0] if len(rs_c) > 0 else "N/A"
                wr_n = int(rs_c.values[0]) if len(rs_c) > 0 else 0

                # Daily breakdown
                df["Date"] = df["Last Updated Time"].dt.strftime("%d/%m")
                daily = df.groupby("Date").size()
                dl = "\n".join([f"  {d}: {n} lost" for d, n in daily.items()])

                # Shift breakdown
                sl = "\n".join([f"  {s}: {int(sh_c.get(s,0))} ({round(int(sh_c.get(s,0))/total*100,1)}%)"
                                for s in SHIFT_ORDER])

                # Top 3 clusters with their top 3 aisles
                cdet = ""
                for cn, cv in cl_c.head(3).items():
                    ta = df[df["Cluster"] == cn]["Aisle"].dropna().value_counts().head(3)
                    al = ", ".join([f"{a} ({n})" for a, n in ta.items()])
                    cdet += f"  Cluster {cn}: {cv} ({round(int(cv)/total*100,1)}%) — {al}\n"

                # Top DSPs
                dlines = "\n".join([f"  {d}: {n} ({round(int(n)/total*100,1)}%)"
                                    for d, n in dsp_c.head(3).items()])

                # Size breakdown
                slines = "\n".join([f"  {s}: {n}" for s, n in sz_c.items()])

                # Reason breakdown
                if len(rs_c) > 0:
                    rlines = "\n".join([f"  {r}: {n} ({round(int(n)/total*100,1)}%)"
                                        for r, n in rs_c.head(5).items()])
                else:
                    rlines = "  (Not in export)"

                # State breakdown (if column exists)
                if "State" in df.columns:
                    stlines = "\n".join([f"  {s}: {n}" for s, n in
                                         df["State"].dropna().value_counts().items()])
                else:
                    stlines = "  (Not in export)"

                # Generate dynamic actions based on patterns
                acts = []
                def ac(t): acts.append(f"AC{len(acts)+1}: {t}")

                if wc_p > 40:
                    ac(f"Dedicated PS to Cluster {wc} — {wc_p}% of losts concentrated here.")
                else:
                    ac(f"PS rotation between top clusters ({', '.join([str(c) for c in cl_c.head(3).index])}).")

                if dm >= 2:
                    ac(f"Stand-down meeting with DSP {wd} leadership — {dm}x the station average.")
                elif dm >= 1.5:
                    ac(f"Process briefing for DSP {wd} — {dm}x the station average.")
                else:
                    ac("Station-wide process refresher — losts spread across DSPs evenly.")

                if ws in ["Large Oversize", "Small Oversize"]:
                    ac(f"Oversize stow audit in Aisle {wa}.")
                elif ws == "Small":
                    ac(f"Small parcel stow briefing — {ws_n} small parcels lost.")
                else:
                    ac(f"Stow walk Cluster {wc} focusing on {ws} parcels.")

                if avg_a > 0 and wa_n >= avg_a * 3:
                    ac(f"Physical inspection of Aisle {wa} — {round(wa_n/avg_a,1)}x the average.")
                elif avg_a > 0 and wa_n >= avg_a * 2:
                    ac(f"Increase PS presence in Aisle {wa}.")
                else:
                    ac("Daily PS huddle with aisle focus rotation.")

                if len(sh_c) > 1 and wsh_n > total * 0.5:
                    ac(f"5-whys session for {wsh} shift — {wsh_n} losts ({round(wsh_n/total*100,1)}%).")

                # Reason-based action
                if len(rs_c) > 0 and wr_n > total * 0.3:
                    ac(f"Process deep-dive on '{wr}' — {wr_n} parcels ({round(wr_n/total*100,1)}% of total).")

                relo = sum(int(cy_c.get(c, 0)) for c in cy_c.index if "RELO" in str(c).upper())
                if relo > total * 0.15:
                    ac(f"RELO process review — {relo} parcels lost during RELO.")

                if len(daily) > 1 and wday_n > total * 0.3:
                    ac(f"Staffing review for {wday} — {wday_n} losts ({round(wday_n/total*100,1)}%).")

                # Build the bridge text
                bridge = f"""Lost Parcels Bridge — DRM2
{dr}

LOST (Total): {total}

DAILY BREAKDOWN:
{dl}

SHIFT BREAKDOWN:
{sl}

RC1) LOCATION:
{cdet}
RC2) DSP:
{dlines}

RC3) SIZE:
{slines}

RC4) REASON:
{rlines}

RC5) STATUS:
{stlines}

ACTIONS:
{chr(10).join(acts)}
"""
                # Editable text area
                st.text_area("✏️ Edit bridge below:", value=bridge, height=400, key="bridge_edit")

                # Quick enhancement prompt
                st.subheader("🤖 Enhance with Quick")
                prompt = (
                    f"Write a professional Lost Parcels bridge for DRM2 station. "
                    f"Include RC1-RC5 root causes and AC1-AC5+ specific actions.\n\n"
                    f"Data ({dr}): Total={total}, Worst Cluster={wc} ({wc_n}, {wc_p}%), "
                    f"Worst Aisle={wa} ({wa_n}), Worst DSP={wd} ({wd_n}, {dm}x avg), "
                    f"Worst Size={ws} ({ws_n}), Worst Day={wday} ({wday_n}), "
                    f"Worst Shift={wsh} ({wsh_n}), Top Reason={wr} ({wr_n})\n\n"
                    f"Daily: {dl}\nShifts: {sl}\nClusters:\n{cdet}DSPs:\n{dlines}\n"
                    f"Reasons:\n{rlines}\n\n"
                    f"Generate specific, actionable recommendations referencing exact clusters, "
                    f"aisles, DSPs, sizes, days, shifts, and loss reasons."
                )
                st.code(prompt, language="text")  # code block has built-in copy icon
                st.caption("📋 Click the copy icon → open Quick → paste → get AI-enhanced bridge")


# =====================================================================
# MULTI-STATION COMPARE MODE
# =====================================================================
else:
    st.subheader("Upload Station Data")
    st.caption("Upload one SCC export per station (2–5 stations).")
    num = st.slider("How many stations?", 2, 5, 2, key="num_stations")  # pick number of stations

    # Create one uploader per station
    uploaded = {}
    cols = st.columns(num)
    for i in range(num):
        with cols[i]:
            f = st.file_uploader(f"Station {i+1}", type="csv", key=f"up_{i}")
            if f:
                uploaded[i] = f

    if len(uploaded) >= 2:
        # Load and clean each station's data
        stations = {}  # {name: dataframe}
        names = []  # ordered list of station names
        for i, file in uploaded.items():
            tmp = clean_data(pd.read_csv(file))  # read + clean
            name = get_station_name(tmp, file.name)  # auto-detect name
            stations[name] = tmp
            names.append(name)

        st.success(f"✅ Loaded: **{', '.join(names)}**")

        # Station summary metrics
        st.subheader("Station Overview")
        mc = st.columns(len(names))
        for i, n in enumerate(names):
            sk = stations[n][stations[n]["Shift"] != "Unknown"]["Shift"]
            mc[i].metric(n, f"{len(stations[n])} lost")
            mc[i].caption(f"Worst shift: {safe_top(sk) if len(sk) > 0 else 'N/A'}")

        # Cluster priorities per station
        for n in names:
            cr = stations[n]["Cluster"].dropna().value_counts()
            if len(cr) > 0:
                parts = []
                for i, (c, v) in enumerate(cr.head(3).items()):
                    pct = round(v / len(stations[n]) * 100, 1)
                    parts.append(f"#{i+1} Cluster {c} — {v} ({pct}%)")
                st.info(f"🎯 **{n} priority:** {' → '.join(parts)}")

        # Tabs
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📊 Summary",
            "📍 Lost Locations",
            "🚚 DSP",
            "⏱️ Shift Rankings",
            "📅 Day of Week",
            "💾 Export",
            "📋 Bridge"
        ])

        # TAB 1: SUMMARY
        with t1:
            st.info("💡 **What's here:** Compare total lost parcels, size breakdowns, and "
                    "**loss reasons** across all stations.")
            view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_sum_v")

            if view == "Chart":
                # Total comparison bar chart (vertical — station names are short)
                fig, ax = plt.subplots(figsize=CHART)
                bars = ax.bar(names, [len(stations[n]) for n in names],
                              color=STATION_COLORS[:len(names)])
                for b in bars:
                    h = b.get_height()
                    ax.text(b.get_x() + b.get_width()/2, h + 0.3, str(int(h)),
                            ha="center", fontsize=8, fontweight="bold")
                ax.set_ylabel("Lost Parcels", fontsize=8)
                ax.set_title("Total Lost by Station", fontsize=9)
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)

                # Grouped size comparison
                fig2, ax2 = plt.subplots(figsize=(8, 2.5))
                x = range(len(SIZE_ORDER))
                w = 0.8 / len(names)
                for i, n in enumerate(names):
                    cts = [len(stations[n][stations[n]["Size Category"] == s]) for s in SIZE_ORDER]
                    offset = (i - len(names)/2 + 0.5) * w
                    ax2.bar([xi + offset for xi in x], cts, w,
                            label=n, color=STATION_COLORS[i])
                ax2.set_xticks(x)
                ax2.set_xticklabels(SIZE_ORDER, fontsize=7)
                ax2.set_ylabel("Lost", fontsize=8)
                ax2.set_title("Size Comparison", fontsize=9)
                ax2.legend(fontsize=7)
                ax2.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig2)

                # Reason comparison per station (expandable)
                has_reason = any("Reason" in stations[n].columns for n in names)
                if has_reason:
                    st.markdown("---")
                    st.subheader("❓ Loss Reasons by Station")
                    for i, n in enumerate(names):
                        if "Reason" in stations[n].columns:
                            rc = stations[n]["Reason"].dropna().value_counts().head(5)
                            if len(rc) > 0:
                                st.pyplot(make_bar_horiz(rc,
                                                        f"{n} — Top Reasons",
                                                        color=STATION_COLORS[i],
                                                        figsize_width=6))
            else:
                st.dataframe(pd.DataFrame({
                    "Station": names,
                    "Total Lost": [len(stations[n]) for n in names]
                }, index=range(1, len(names)+1)))

                # Reason table per station
                has_reason = any("Reason" in stations[n].columns for n in names)
                if has_reason:
                    st.subheader("❓ Loss Reasons")
                    for n in names:
                        if "Reason" in stations[n].columns:
                            rc = stations[n]["Reason"].dropna().value_counts()
                            if len(rc) > 0:
                                with st.expander(f"{n} — Reasons"):
                                    st.dataframe(make_table(rc, "Reason", "Lost Parcels"),
                                                 use_container_width=True)

        # TAB 2: LOST LOCATIONS
        with t2:
            st.info("💡 **What's here:** Compare worst locations across stations. "
                    "Pick a category to see Top 10 rankings, then expand a station to drill down.")

            rank_by = st.selectbox("Rank by:", ["Cluster", "Aisle", "Sort Zone"], key="mc_rb")
            rank_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_rv")

            for i, n in enumerate(names):
                d = stations[n][rank_by].dropna().value_counts().head(10)
                if len(d) > 0:
                    if rank_view == "Chart":
                        # Horizontal bars — no overlap regardless of label length
                        st.pyplot(make_bar_horiz(d,
                                                f"{n} — Top 10 {rank_by}s",
                                                color=STATION_COLORS[i],
                                                figsize_width=6))
                    else:
                        st.subheader(n)
                        st.dataframe(make_table(d, rank_by, "Lost Parcels"))

            st.markdown("---")
            st.subheader("🔍 Station Drill-Down")
            for i, n in enumerate(names):
                with st.expander(f"📍 {n} — Cluster Detail", expanded=False):
                    sdf = stations[n]
                    cls = sorted(sdf["Cluster"].dropna().unique())
                    if cls:
                        sel = st.selectbox(f"Cluster ({n}):", cls, key=f"mc_cl_{i}")
                        flt = sdf[sdf["Cluster"] == sel]
                        st.write(f"**{len(flt)} parcels** in Cluster {sel}")
                        ai = flt["Aisle"].dropna().value_counts().head(10)
                        if len(ai) > 0:
                            st.pyplot(make_bar_horiz(ai,
                                                    f"{n} — Cluster {sel} Aisles",
                                                    color=STATION_COLORS[i],
                                                    figsize_width=6))

        # TAB 3: DSP
        with t3:
            st.info("💡 **What's here:** Compare DSP performance across stations. "
                    "Chart = worst-first ranking. Table = alphabetical for quick lookup.")
            dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_dsp_v")

            if dsp_view == "Chart":
                for i, n in enumerate(names):
                    dsp = stations[n]["DSP Name"].dropna().value_counts().head(10)
                    if len(dsp) > 0:
                        st.pyplot(make_bar_horiz(dsp, f"{n} — Worst DSPs",
                                                color=STATION_COLORS[i], max_label=DSP_MAX))
            else:
                for i, n in enumerate(names):
                    with st.expander(f"🚚 {n} — DSP List (A–Z)", expanded=False):
                        dsp = stations[n]["DSP Name"].dropna().value_counts().sort_index()
                        if len(dsp) > 0:
                            st.dataframe(make_table(dsp, "DSP", "Lost Parcels"),
                                         use_container_width=True)

        # TAB 4: SHIFT RANKINGS
        with t4:
            st.info("💡 **What's here:** Compare which shifts lose the most parcels at each station.")

            # Grouped shift comparison chart
            fig_sh, ax_sh = plt.subplots(figsize=CHART)
            x = range(len(SHIFT_ORDER))
            w = 0.8 / len(names)
            for i, n in enumerate(names):
                sd = stations[n][stations[n]["Shift"] != "Unknown"]["Shift"].value_counts()
                cts = [sd.get(s, 0) for s in SHIFT_ORDER]
                offset = (i - len(names)/2 + 0.5) * w
                bars = ax_sh.bar([xi + offset for xi in x], cts, w,
                                 label=n, color=STATION_COLORS[i])
                # Add values on bars
                for b in bars:
                    h = b.get_height()
                    if h > 0:
                        ax_sh.text(b.get_x() + b.get_width()/2, h + 0.2,
                                   str(int(h)), ha="center", fontsize=6)
            ax_sh.set_xticks(x)
            ax_sh.set_xticklabels(SHIFT_ORDER, fontsize=8)
            ax_sh.set_ylabel("Lost Parcels", fontsize=8)
            ax_sh.set_title("Shift Comparison", fontsize=9)
            ax_sh.tick_params(labelsize=7)
            ax_sh.legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig_sh)

            st.markdown("---")
            for i, n in enumerate(names):
                with st.expander(f"⏱️ {n} — Full Shift Detail", expanded=False):
                    render_shift_tab(stations[n], len(stations[n]),
                                     get_date_range(stations[n]),
                                     key_prefix=f"mc_{i}_")

        # TAB 5: DAY OF WEEK
        with t5:
            st.info("💡 **What's here:** Overlaid line graph comparing daily patterns across stations. "
                    "Values shown on each data point so you can read exact numbers.")

            fig, ax = plt.subplots(figsize=CHART)
            for i, n in enumerate(names):
                if "Day of Week" in stations[n].columns:
                    dd = stations[n]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                    ax.plot(dd.index, dd.values, marker="o", label=n,
                            color=STATION_COLORS[i], linewidth=2, markersize=5)
                    # Add values on each point
                    for j, (day, val) in enumerate(dd.items()):
                        ax.annotate(str(int(val)), xy=(j, val), xytext=(0, 8),
                                    textcoords="offset points", ha="center",
                                    fontsize=7, color=STATION_COLORS[i])
            ax.set_ylabel("Lost Parcels", fontsize=8)
            ax.set_title("Lost by Day of Week", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        # TAB 6: EXPORT
        with t6:
            st.info("💡 **What's here:** Download cleaned data per station or as one combined file.")
            for n in names:
                st.download_button(f"⬇️ Download {n}", stations[n].to_csv(index=False),
                                   f"Lost_{n}.csv", "text/csv", key=f"dl_{n}")
            combined = pd.concat([stations[n].assign(Station_Name=n) for n in names], ignore_index=True)
            st.download_button("⬇️ Download All Stations (Combined)", combined.to_csv(index=False),
                               "Lost_Combined.csv", "text/csv", key="dl_all")

        # TAB 7: BRIDGE
        with t7:
            st.info("💡 **What's here:** Comparison table + Quick prompt for a cross-station bridge. "
                    "See which station is worst and what it can learn from the best.")

            # Comparison table (now includes top reason)
            comp = []
            for n in names:
                sdf = stations[n]
                sk = sdf[sdf["Shift"] != "Unknown"]["Shift"]
                top_reason = safe_top(sdf["Reason"]) if "Reason" in sdf.columns else "N/A"
                comp.append({
                    "Station": n,
                    "Total Lost": len(sdf),
                    "Worst Cluster": safe_top(sdf["Cluster"]),
                    "Worst Aisle": safe_top(sdf["Aisle"]),
                    "Worst DSP": safe_top(sdf["DSP Name"]),
                    "Worst Shift": safe_top(sk) if len(sk) > 0 else "N/A",
                    "Top Reason": top_reason
                })
            st.dataframe(pd.DataFrame(comp, index=range(1, len(comp)+1)),
                         use_container_width=True)

            # Best vs worst
            best = min(names, key=lambda n: len(stations[n]))
            worst = max(names, key=lambda n: len(stations[n]))
            gap = len(stations[worst]) - len(stations[best])
            st.write(f"**Best:** {best} ({len(stations[best])}) | "
                     f"**Worst:** {worst} ({len(stations[worst])}) | **Gap:** {gap}")

            # Quick prompt (includes reasons)
            summ = "\n".join([
                f"- {n}: {len(stations[n])} losts, worst cluster={safe_top(stations[n]['Cluster'])}, "
                f"worst DSP={safe_top(stations[n]['DSP Name'])}, "
                f"top reason={safe_top(stations[n]['Reason']) if 'Reason' in stations[n].columns else 'N/A'}"
                for n in names
            ])
            st.code(f"Compare these stations and recommend what {worst} can learn from {best}:\n{summ}\n"
                    f"Best={best} ({len(stations[best])}), Worst={worst} ({len(stations[worst])}), Gap={gap}\n"
                    f"Include analysis of loss reasons and recommend targeted process improvements.",
                    language="text")
            st.caption("📋 Click the copy icon → open Quick → paste → get AI comparison")

    elif len(uploaded) == 1:
        st.warning("⚠️ Upload at least 2 station files to compare.")
    else:
        st.info("👆 Upload your CSV files above to get started.")
```
