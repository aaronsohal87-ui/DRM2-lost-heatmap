import streamlit as st  # web app framework
import pandas as pd  # data manipulation (dataframes)
import matplotlib.pyplot as plt  # chart generation

# --- PAGE CONFIG ---
st.set_page_config(page_title="SCC Lost Heatmap", page_icon="📦", layout="wide")  # wide layout, browser tab title
st.title("SCC Lost Parcel Heatmap")  # main heading
st.markdown("---")  # horizontal divider

# --- CONSTANTS ---
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]  # one colour per station (up to 5)
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]  # standard size categories
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # week order
SENSITIVE_COLS = [  # columns to auto-remove (personal data)
    "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
    "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
    "Payment Method", "District", "Scheduled Delivery End Time"
]
REQUIRED_COLS = [  # columns the app needs to function
    "Tracking ID", "Sort Zone", "Aisle", "Cluster",
    "Package Length", "Package Width", "Package Height",
    "DSP Name", "Assigned Cycle", "Last Updated Time"
]


# --- HELPER FUNCTIONS ---
def get_size(longest):
    """Classify a parcel by its longest side into Amazon UK size tiers"""
    if pd.isna(longest):
        return "Unknown"  # no dimension data
    elif longest <= 35:
        return "Small"
    elif longest <= 45:
        return "Medium"
    elif longest <= 61:
        return "Small Oversize"
    else:
        return "Large Oversize"


def clean_data(df):
    """Clean one station's data: remove sensitive cols, fix dimensions, add size + day"""
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])  # remove sensitive columns

    # Fix package dimensions (remove "cm" text, convert to numbers)
    for col in ["Package Length", "Package Width", "Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm", "").str.replace("cm", "")  # strip unit text
            df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to float, NaN if invalid

    # Calculate longest side and size category
    dim_cols = ["Package Length", "Package Width", "Package Height"]
    if all(c in df.columns for c in dim_cols):
        df["Longest Side"] = df[dim_cols].max(axis=1)  # longest of the 3 dimensions
    else:
        df["Longest Side"] = float("nan")  # no dimensions available

    df["Size Category"] = df["Longest Side"].apply(get_size)  # apply size classification

    # Parse timestamps and extract day of week
    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], errors="coerce")  # parse date
        df["Day of Week"] = df["Last Updated Time"].dt.day_name()  # e.g. "Monday"

    return df


def get_station_name(df, filename):
    """Get station name from Station column, or fall back to filename"""
    if "Station" in df.columns and len(df["Station"].dropna()) > 0:
        return df["Station"].dropna().iloc[0]  # use first non-null station value
    return filename.replace(".csv", "").replace("_", " ").strip()[:20]  # clean up filename as fallback


def get_date_range(df):
    """Return formatted date range string from the data"""
    start = df["Last Updated Time"].min().strftime("%d %b %Y")  # earliest date
    end = df["Last Updated Time"].max().strftime("%d %b %Y")  # latest date
    return start if start == end else f"{start} - {end}"  # single day or range


def safe_top(series, n=1):
    """Safely get top value(s) from a series, returns 'N/A' if empty"""
    counts = series.dropna().value_counts()  # count occurrences, ignore NaN
    if len(counts) == 0:
        return "N/A" if n == 1 else pd.Series(dtype="object")  # empty fallback
    return counts.index[0] if n == 1 else counts.head(n)  # top 1 or top N


def plot_bar(data, xlabel, ylabel, title, color="steelblue", horizontal=False, figsize=(9, 3.5)):
    """Reusable bar chart function — smaller sizing, no rotated labels"""
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        ax.barh(data.index, data.values, color=color)  # horizontal bars
        ax.invert_yaxis()  # highest at top
    else:
        ax.bar(data.index, data.values, color=color)  # vertical bars
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=0, ha="center")  # always horizontal labels
    plt.tight_layout()  # prevent label cutoff
    return fig


def make_table(series, col1_name, col2_name):
    """Convert a value_counts series into a numbered dataframe for display"""
    tbl = series.reset_index()  # convert index to column
    tbl.columns = [col1_name, col2_name]  # rename columns
    tbl.index = range(1, len(tbl) + 1)  # start numbering from 1
    return tbl


# --- MODE TOGGLE ---
mode = st.radio("Mode:", ["Single Station", "Multi-Station Compare"], horizontal=True, key="mode_toggle")


# =====================================================================
# SINGLE STATION MODE
# =====================================================================
if mode == "Single Station":

    uploaded_file = st.file_uploader("Upload SCC export (.csv)", type="csv")  # file upload widget

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # read CSV into dataframe

        # Show warning if sensitive columns were found
        found_sensitive = [c for c in SENSITIVE_COLS if c in df.columns]
        if found_sensitive:
            st.warning(f"Sensitive columns found and removed: {', '.join(found_sensitive)}")

        df = clean_data(df)  # apply all cleaning steps

        # Check required columns exist
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.info("Check your SCC export includes these fields, then re-upload.")
        elif len(df) == 0:
            st.warning("File has no data rows.")
        else:
            st.success(f"Data loaded — {len(df)} packages ready.")

            date_range_text = get_date_range(df)  # e.g. "01 Jul 2025 - 07 Jul 2025"

            # --- SUMMARY METRICS ---
            st.subheader(f"Quick Summary ({date_range_text})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Lost", len(df))
            c2.metric("Worst Cluster", safe_top(df["Cluster"]))
            c3.metric("Worst Aisle", safe_top(df["Aisle"]))
            c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])  # truncate long DSP names

            if len(df) < 5:
                st.info("Small dataset — consider uploading a full week.")

            # --- TABS ---
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                ["Overview", "Location", "Rankings", "DSP & Cycle", "Time", "Export", "Bridge"]
            )

            # TAB 1: OVERVIEW
            with tab1:
                st.caption("Size breakdown and cluster summary.")
                view = st.radio("Display:", ["Table", "Chart"], horizontal=True, key="ov_view")

                if view == "Table":
                    st.subheader("Size Breakdown")
                    st.write(df["Size Category"].value_counts())  # simple count per size
                    st.subheader("Cluster × Size")
                    tbl = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0)  # pivot table
                    tbl["Total"] = tbl.sum(axis=1)  # add total column
                    st.dataframe(tbl)
                else:
                    # Size bar chart
                    size_counts = df["Size Category"].value_counts()
                    if len(size_counts) > 0:
                        colors = ["green", "orange", "red", "darkred", "grey"][:len(size_counts)]
                        fig = plot_bar(size_counts, "Size Category", "Lost Parcels",
                                       f"Lost by Size ({date_range_text})", color=colors)
                        st.pyplot(fig)

                    # Cluster bar chart
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
                    filt = df[df["Cluster"] == sel_cluster]  # filter to selected cluster
                    st.write(f"{len(filt)} parcels in Cluster {sel_cluster}")

                    view_by = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="view_by")
                    loc_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="loc_view")
                    loc_data = filt[view_by].dropna().value_counts()

                    if loc_view == "Chart":
                        if len(loc_data) > 0:
                            st.pyplot(plot_bar(loc_data, view_by, "Lost Parcels",
                                               f"Cluster {sel_cluster} by {view_by}", figsize=(10, 3.5)))

                        # Size filter within cluster
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
                                           f"Top 10 {rank_by}s ({date_range_text})", color="darkred", horizontal=True))
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
                        st.pyplot(plot_bar(dsp_data, "DSP", "Lost Parcels",
                                           f"Lost by DSP ({date_range_text})", color="orange"))
                    if len(cycle_data) > 0:
                        st.pyplot(plot_bar(cycle_data, "Cycle", "Lost Parcels",
                                           f"Lost by Cycle ({date_range_text})", color="purple"))
                else:
                    if len(dsp_data) > 0:
                        st.subheader("DSP")
                        st.dataframe(make_table(dsp_data, "DSP", "Lost Parcels"))
                    if len(cycle_data) > 0:
                        st.subheader("Cycle")
                        st.dataframe(make_table(cycle_data, "Cycle", "Lost Parcels"))

            # TAB 5: TIME
            with tab5:
                st.caption("Day-of-week patterns and tracking ID lookup.")
                day_data = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)  # force Mon-Sun order
                time_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="time_view")

                if time_view == "Chart":
                    st.pyplot(plot_bar(day_data, "Day", "Lost Parcels",
                                       f"Lost by Day ({date_range_text})", color="green"))
                else:
                    st.dataframe(make_table(day_data, "Day", "Lost Parcels"))

                # Day drill-down
                st.subheader("Parcel Details by Day")
                available_days = [d for d in DAY_ORDER if d in df["Day of Week"].values]  # only days with data
                if available_days:
                    sel_day = st.selectbox("Day:", available_days, key="day_sel")
                    day_df = df[df["Day of Week"] == sel_day]  # filter to selected day
                    st.write(f"{len(day_df)} parcels on {sel_day}")
                    show_cols = [c for c in ["Tracking ID", "Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"] if c in df.columns]
                    st.dataframe(day_df[show_cols])  # show relevant columns only

            # TAB 6: EXPORT
            with tab6:
                st.caption("Download cleaned data.")
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "Lost_Parcels_Cleaned.csv", "text/csv")  # one-click download

            # TAB 7: BRIDGE
            with tab7:
                st.caption("Auto-generated bridge with dynamic action plans.")
                total = len(df)

                # Gather key stats
                cl_counts = df["Cluster"].dropna().value_counts()
                ai_counts = df["Aisle"].dropna().value_counts()
                dsp_counts = df["DSP Name"].dropna().value_counts()
                sz_counts = df["Size Category"].value_counts()
                cy_counts = df["Assigned Cycle"].dropna().value_counts()
                day_counts = df["Day of Week"].dropna().value_counts()

                # Safe stat extraction
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

                # Daily breakdown
                df["Date"] = df["Last Updated Time"].dt.strftime("%d/%m")  # short date format
                daily = df.groupby("Date").size()  # count per day
                daily_lines = "\n".join([f"{d} - {n} lost" for d, n in daily.items()])

                # Top 3 clusters with their top 3 aisles
                cluster_details = ""
                for cl_name, cl_n in cl_counts.head(3).items():
                    pct = round(cl_n / total * 100, 1)
                    top_aisles = df[df["Cluster"] == cl_name]["Aisle"].dropna().value_counts().head(3)
                    aisles_str = ", ".join([f"{a} ({n})" for a, n in top_aisles.items()])
                    cluster_details += f"  Cluster {cl_name}: {cl_n} ({pct}%) — Aisles: {aisles_str}\n"

                dsp_lines = "\n".join([f"  {d}: {n} ({round(n/total*100,1)}%)" for d, n in dsp_counts.head(3).items()])
                size_lines = "\n".join([f"  {s}: {n}" for s, n in sz_counts.items()])

                # State breakdown (if column exists)
                if "State" in df.columns:
                    state_lines = "\n".join([f"  {s}: {n}" for s, n in df["State"].dropna().value_counts().items()])
                else:
                    state_lines = "  (Not in export)"

                # --- DYNAMIC ACTIONS ---
                actions = []
                ac = lambda txt: actions.append(f"AC{len(actions)+1}: {txt}")  # shorthand to add action

                # Cluster concentration
                if w_cluster_pct > 40:
                    ac(f"Dedicated PS to Cluster {w_cluster} full shift — {w_cluster_pct}% of losts here.")
                else:
                    ac(f"PS rotation between top clusters ({', '.join([str(c) for c in cl_counts.head(3).index])}).")

                # DSP performance
                if dsp_mult >= 2.0:
                    ac(f"DSP {w_dsp} stand-down with leadership — {dsp_mult}x average.")
                elif dsp_mult >= 1.5:
                    ac(f"DSP {w_dsp} process briefing — {dsp_mult}x average.")
                else:
                    ac("Station-wide process refresher — losts spread evenly across DSPs.")

                # Size pattern
                if w_size in ["Large Oversize", "Small Oversize"]:
                    ac(f"Oversize stow audit in Aisle {w_aisle} — {w_size} most commonly lost.")
                elif w_size == "Small":
                    ac(f"Small parcel stow briefing — {w_size_n} small parcels lost.")
                else:
                    ac(f"Stow quality walk in Cluster {w_cluster} for {w_size} parcels.")

                # Aisle anomaly
                if avg_aisle > 0 and w_aisle_n >= avg_aisle * 3:
                    ac(f"Physical inspection of Aisle {w_aisle} — {round(w_aisle_n/avg_aisle,1)}x average.")
                elif avg_aisle > 0 and w_aisle_n >= avg_aisle * 2:
                    ac(f"Increased PS in Aisle {w_aisle} — {w_aisle_n} losts, above average.")
                else:
                    ac("Daily PS huddle to review losts by aisle.")

                # RELO check
                relo_n = sum(cy_counts.get(c, 0) for c in cy_counts.index if "RELO" in str(c))
                if relo_n > total * 0.15:
                    ac(f"RELO process review — {relo_n} losts from relocation cycles.")

                # Day spike
                if len(daily) > 1 and w_day_n > total * 0.3:
                    ac(f"Review {w_day} staffing — {w_day_n} losts ({round(w_day_n/total*100,1)}% of total).")

                # --- BUILD BRIDGE TEXT ---
                bridge = f"""Lost Parcels Bridge - DRM2
{date_range_text}

Lost (Total): {total}

Daily Breakdown:
{daily_lines}

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

                # Quick prompt
                st.subheader("Enhance with Quick")
                prompt = f"""Write a Lost Parcels bridge for DRM2. Use RC1-4 for root causes, AC1-4 for actions. Be specific.

Data ({date_range_text}): Total={total}, Cluster={w_cluster} ({w_cluster_n}, {w_cluster_pct}%), Aisle={w_aisle} ({w_aisle_n}), DSP={w_dsp} ({w_dsp_n}, {dsp_mult}x avg), Size={w_size} ({w_size_n}), Cycle={top_cycle} ({top_cycle_n}), Day={w_day} ({w_day_n})
Daily: {daily_lines}
Clusters: {cluster_details}DSPs: {dsp_lines}

Generate specific actions a station manager would implement. Reference clusters, aisles, DSPs, sizes, days.
"""
                st.code(prompt, language="text")
                st.info("Copy icon (top-right) → Quick → Ctrl+V")


# =====================================================================
# MULTI-STATION COMPARE MODE
# =====================================================================
else:
    st.subheader("Upload Station Data (2–5 stations)")
    st.caption("One SCC export per station. Station name auto-detected from 'Station' column.")

    num = st.slider("Stations to compare:", 2, 5, 2, key="num_st")  # slider for number of stations

    # Separate uploaders in columns
    uploaded = {}
    cols = st.columns(num)
    for i in range(num):
        with cols[i]:
            f = st.file_uploader(f"Station {i+1}", type="csv", key=f"st_up_{i}")
            if f:
                uploaded[i] = f

    # Process if at least 2 files uploaded
    if len(uploaded) >= 2:
        stations = {}  # dict of {name: dataframe}
        names = []  # ordered list of station names

        for i, file in uploaded.items():
            temp = clean_data(pd.read_csv(file))  # read + clean
            name = get_station_name(temp, file.name)  # detect station name
            stations[name] = temp
            names.append(name)

        st.success(f"Loaded: {', '.join(names)}")

        # Summary metrics row
        st.subheader("Station Summary")
        mcols = st.columns(len(names))
        for idx, name in enumerate(names):
            mcols[idx].metric(name, f"{len(stations[name])} lost")

        # --- TABS ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["Overview", "Location", "Rankings", "DSP & Cycle", "Time", "Export", "Bridge"]
        )

        # TAB 1: OVERVIEW COMPARE
        with tab1:
            st.caption("Compare totals, size, and clusters across stations.")
            view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_ov")

            if view == "Chart":
                # Total losts bar
                st.subheader("Total Lost by Station")
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.bar(names, [len(stations[n]) for n in names], color=STATION_COLORS[:len(names)])
                ax.set_ylabel("Lost Parcels")
                ax.set_title("Total Lost Parcels by Station")
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig)

                # Grouped size breakdown
                st.subheader("Size Breakdown by Station")
                fig2, ax2 = plt.subplots(figsize=(9, 3.5))
                x = range(len(SIZE_ORDER))
                w = 0.8 / len(names)  # bar width based on station count
                for idx, name in enumerate(names):
                    counts = [len(stations[name][stations[name]["Size Category"] == s]) for s in SIZE_ORDER]
                    offset = (idx - len(names)/2 + 0.5) * w  # centre bars around tick
                    ax2.bar([xi + offset for xi in x], counts, w, label=name, color=STATION_COLORS[idx])
                ax2.set_xticks(x)
                ax2.set_xticklabels(SIZE_ORDER)
                ax2.set_ylabel("Lost Parcels")
                ax2.legend()
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig2)

                # Cluster charts per station
                st.subheader("Clusters per Station")
                for idx, name in enumerate(names):
                    cl = stations[name]["Cluster"].dropna().value_counts()
                    if len(cl) > 0:
                        st.pyplot(plot_bar(cl, "Cluster", "Lost", f"{name} — Clusters",
                                           color=STATION_COLORS[idx], figsize=(8, 2.5)))
            else:
                # Table view
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

                    # Aisle chart
                    ai = filt["Aisle"].dropna().value_counts().head(10)
                    if len(ai) > 0:
                        st.pyplot(plot_bar(ai, "Aisle", "Lost", f"{name} — Cluster {sel} Aisles",
                                           color=STATION_COLORS[idx], figsize=(9, 3)))
                    # Zone chart
                    zn = filt["Sort Zone"].dropna().value_counts().head(10)
                    if len(zn) > 0:
                        st.pyplot(plot_bar(zn, "Sort Zone", "Lost", f"{name} — Cluster {sel} Zones",
                                           color=STATION_COLORS[idx], figsize=(9, 3)))
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
                                           color=STATION_COLORS[idx], horizontal=True, figsize=(9, 3)))
                    else:
                        st.subheader(name)
                        st.dataframe(make_table(data, rank_by, "Lost Parcels"))

        # TAB 4: DSP & CYCLE COMPARE
        with tab4:
            st.caption("DSP and cycle comparison across stations.")
            dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_dsp_v")

            if dsp_view == "Chart":
                # DSP per station
                for idx, name in enumerate(names):
                    dsp = stations[name]["DSP Name"].dropna().value_counts().head(10)
                    if len(dsp) > 0:
                        st.pyplot(plot_bar(dsp, "DSP", "Lost", f"{name} — Worst DSPs",
                                           color=STATION_COLORS[idx]))

                # Grouped cycle chart
                st.subheader("Cycle Comparison")
                all_cycles = sorted(set().union(*[stations[n]["Assigned Cycle"].dropna().unique() for n in names]))
                if all_cycles:
                    fig, ax = plt.subplots(figsize=(9, 3.5))
                    x = range(len(all_cycles))
                    w = 0.8 / len(names)
                    for idx, name in enumerate(names):
                        cy = stations[name]["Assigned Cycle"].dropna().value_counts()
                        counts = [cy.get(c, 0) for c in all_cycles]
                        offset = (idx - len(names)/2 + 0.5) * w
                        ax.bar([xi + offset for xi in x], counts, w, label=name, color=STATION_COLORS[idx])
                    ax.set_xticks(x)
                    ax.set_xticklabels(all_cycles)
                    ax.set_ylabel("Lost Parcels")
                    ax.set_title("Cycle Comparison — All Stations")
                    ax.legend()
                    plt.xticks(rotation=0, ha="center")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                for idx, name in enumerate(names):
                    st.subheader(name)
                    dsp = stations[name]["DSP Name"].dropna().value_counts()
                    if len(dsp) > 0:
                        st.write("**DSPs:**")
                        st.dataframe(make_table(dsp, "DSP", "Lost Parcels"))
                    cy = stations[name]["Assigned Cycle"].dropna().value_counts()
                    if len(cy) > 0:
                        st.write("**Cycles:**")
                        st.dataframe(make_table(cy, "Cycle", "Lost Parcels"))
                    st.markdown("---")

        # TAB 5: TIME COMPARE
        with tab5:
            st.caption("Day-of-week patterns across stations.")
            time_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_time_v")

            if time_view == "Chart":
                # Overlaid line chart (all stations on one graph)
                st.subheader("Day of Week — All Stations")
                fig, ax = plt.subplots(figsize=(9, 3.5))
                for idx, name in enumerate(names):
                    if "Day of Week" in stations[name].columns:
                        dd = stations[name]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        ax.plot(dd.index, dd.values, marker="o", label=name,
                                color=STATION_COLORS[idx], linewidth=2)  # line per station
                ax.set_ylabel("Lost Parcels")
                ax.set_title("Lost by Day — Station Comparison")
                ax.legend()
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig)

                # Individual bar charts
                for idx, name in enumerate(names):
                    if "Day of Week" in stations[name].columns:
                        dd = stations[name]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        st.pyplot(plot_bar(dd, "Day", "Lost", f"{name} — by Day",
                                           color=STATION_COLORS[idx], figsize=(8, 2.5)))
            else:
                st.subheader("Day of Week Comparison")
                day_all = {}
                for name in names:
                    if "Day of Week" in stations[name].columns:
                        day_all[name] = stations[name]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0).values
                st.dataframe(pd.DataFrame(day_all, index=DAY_ORDER))  # one column per station

        # TAB 6: EXPORT COMPARE
        with tab6:
            st.caption("Download individual or combined station data.")

            st.subheader("Individual Downloads")
            for name in names:
                st.download_button(f"Download {name}", stations[name].to_csv(index=False),
                                   f"Lost_{name}_Cleaned.csv", "text/csv", key=f"dl_{name}")

            st.subheader("Combined Download")
            combined = pd.concat([stations[n].assign(Station_Name=n) for n in names], ignore_index=True)
            st.download_button("Download All Stations Combined", combined.to_csv(index=False),
                               "Lost_All_Stations_Combined.csv", "text/csv", key="dl_all")

        # TAB 7: BRIDGE COMPARE
        with tab7:
            st.caption("Cross-station comparison summary.")

            # Comparison table
            st.subheader("Station Comparison")
            comp = []
            for name in names:
                sdf = stations[name]
                comp.append({
                    "Station": name,
                    "Total Lost": len(sdf),
                    "Worst Cluster": safe_top(sdf["Cluster"]),
                    "Worst Aisle": safe_top(sdf["Aisle"]),
                    "Worst DSP": safe_top(sdf["DSP Name"]),
                    "Most Lost Size": safe_top(sdf["Size Category"]),
                    "Top Cycle": safe_top(sdf["Assigned Cycle"])
                })
            st.dataframe(pd.DataFrame(comp, index=range(1, len(comp)+1)))

            # Best vs worst
            st.subheader("Key Differences")
            best = min(names, key=lambda n: len(stations[n]))  # fewest losts
            worst = max(names, key=lambda n: len(stations[n]))  # most losts
            gap = len(stations[worst]) - len(stations[best])
            gap_pct = round(gap / len(stations[worst]) * 100, 1) if len(stations[worst]) > 0 else 0

            st.write(f"**Best:** {best} ({len(stations[best])} losts)")
            st.write(f"**Worst:** {worst} ({len(stations[worst])} losts)")
            st.write(f"**Gap:** {gap} parcels ({gap_pct}% difference)")

            # Quick prompt for comparison
            st.subheader("Enhance with Quick")
            summaries = "\n".join([f"- {n}: {len(stations[n])} losts, worst cluster={safe_top(stations[n]['Cluster'])}, worst DSP={safe_top(stations[n]['DSP Name'])}" for n in names])
            compare_prompt = f"""Compare these stations' lost parcel performance and suggest what {worst} can learn from {best}:

{summaries}

Best: {best} ({len(stations[best])} losts)
Worst: {worst} ({len(stations[worst])} losts)
Gap: {gap} parcels

Generate specific recommendations for {worst} based on what {best} does differently. Cover cluster management, DSP accountability, size handling, and cycles.
"""
            st.code(compare_prompt, language="text")
            st.info("Copy icon → Quick → Ctrl+V")

    elif len(uploaded) == 1:
        st.warning("Upload at least 2 files to compare.")  # need minimum 2
    else:
        st.info("Upload CSV files above to begin.")  # no files yet
