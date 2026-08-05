import streamlit as st  # Streamlit — builds the web app interface
import pandas as pd  # Pandas — reads/manipulates the CSV data as tables (dataframes)
import matplotlib.pyplot as plt  # Matplotlib — generates all the charts/graphs

# --- PAGE SETUP ---
# layout="wide" makes the app fill the full browser width instead of a narrow column
st.set_page_config(page_title="SCC Lost Heatmap", page_icon="📦", layout="wide")
st.title("SCC Lost Parcel Heatmap")  # big heading at the top of the page
st.markdown("---")  # draws a horizontal line as a visual separator

# --- CONSTANTS (values that never change, defined once at the top for easy editing) ---
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]  # each station gets a unique colour
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]  # Amazon UK parcel size tiers
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # forces Mon→Sun order

# These columns contain personal/sensitive data and must be auto-removed on upload
SENSITIVE_COLS = [
    "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
    "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
    "Payment Method", "District", "Scheduled Delivery End Time"
]

# These columns MUST exist in the CSV or the app can't work
REQUIRED_COLS = [
    "Tracking ID", "Sort Zone", "Aisle", "Cluster",
    "Package Length", "Package Width", "Package Height",
    "DSP Name", "Assigned Cycle", "Last Updated Time"
]

# --- CHART SIZE CONSTANTS (change these to resize ALL charts at once) ---
CHART_FULL = (7, 2.5)  # standard chart size (width, height in inches)
CHART_SMALL = (6, 2)  # smaller chart for per-station breakdowns
CHART_WIDE = (8, 2.5)  # slightly wider for charts with many x-axis labels


# --- HELPER FUNCTIONS (reusable blocks of code) ---

def get_size(longest):
    """
    Takes the longest side of a parcel (in cm) and returns its size category.
    Based on Amazon UK size tiers. Returns "Unknown" if no measurement available.
    """
    if pd.isna(longest):  # pd.isna checks if the value is missing/NaN
        return "Unknown"
    elif longest <= 35:
        return "Small"
    elif longest <= 45:
        return "Medium"
    elif longest <= 61:
        return "Small Oversize"
    else:
        return "Large Oversize"


def clean_data(df):
    """
    Takes a raw dataframe from CSV and cleans it:
    1. Removes sensitive columns (personal data)
    2. Fixes package dimension columns (removes "cm" text, converts to numbers)
    3. Calculates longest side and assigns size category
    4. Parses timestamps and extracts day of week
    Returns the cleaned dataframe.
    """
    # Step 1: Drop any sensitive columns that exist in this file
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])

    # Step 2: Clean dimension columns — SCC exports them as text like "23.50 cm"
    for col in ["Package Length", "Package Width", "Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm", "").str.replace("cm", "")  # remove "cm" text
            df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to number; "coerce" means invalid → NaN

    # Step 3: Find the longest of the 3 dimensions, then classify size
    dim_cols = ["Package Length", "Package Width", "Package Height"]
    if all(c in df.columns for c in dim_cols):  # check all 3 columns exist
        df["Longest Side"] = df[dim_cols].max(axis=1)  # max across columns for each row
    else:
        df["Longest Side"] = float("nan")  # no dimension data available

    df["Size Category"] = df["Longest Side"].apply(get_size)  # run get_size() on every row

    # Step 4: Parse the timestamp column and extract day names
    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], errors="coerce")  # text → datetime
        df["Day of Week"] = df["Last Updated Time"].dt.day_name()  # e.g. "Monday", "Tuesday"

    return df  # return the cleaned version


def get_station_name(df, filename):
    """
    Tries to get the station name from the "Station" column in the data.
    If that column doesn't exist or is empty, falls back to using the filename.
    """
    if "Station" in df.columns and len(df["Station"].dropna()) > 0:
        return df["Station"].dropna().iloc[0]  # grab the first non-empty station value
    # Fallback: clean up the filename to use as station name
    return filename.replace(".csv", "").replace("_", " ").strip()[:20]


def get_date_range(df):
    """
    Looks at the earliest and latest dates in the data.
    Returns a string like "01 Jul 2025 - 07 Jul 2025" or just "01 Jul 2025" if single day.
    """
    start = df["Last Updated Time"].min().strftime("%d %b %Y")  # earliest date formatted
    end = df["Last Updated Time"].max().strftime("%d %b %Y")  # latest date formatted
    return start if start == end else f"{start} - {end}"  # single day vs range


def safe_top(series, n=1):
    """
    Safely gets the most common value from a column.
    Returns "N/A" if the column is empty (prevents crashes).
    n=1 returns just the top value; n>1 returns the top N as a series.
    """
    counts = series.dropna().value_counts()  # count how many times each value appears
    if len(counts) == 0:
        return "N/A" if n == 1 else pd.Series(dtype="object")  # nothing to return
    return counts.index[0] if n == 1 else counts.head(n)  # most frequent value


def plot_bar(data, xlabel, ylabel, title, color="steelblue", horizontal=False, figsize=CHART_FULL):
    """
    Creates a bar chart from a pandas Series (index = labels, values = bar heights).
    - horizontal=True makes it a horizontal bar chart (good for rankings)
    - figsize controls width and height in inches
    - All labels are always horizontal (rotation=0)
    Returns the figure object ready for st.pyplot().
    """
    fig, ax = plt.subplots(figsize=figsize)  # create figure + axes at specified size
    if horizontal:
        ax.barh(data.index, data.values, color=color)  # horizontal bars
        ax.invert_yaxis()  # put highest value at the top
    else:
        ax.bar(data.index, data.values, color=color)  # vertical bars
    ax.set_xlabel(xlabel, fontsize=8)  # x-axis label, small font
    ax.set_ylabel(ylabel, fontsize=8)  # y-axis label, small font
    ax.set_title(title, fontsize=9)  # chart title, slightly larger
    ax.tick_params(labelsize=7)  # make tick labels smaller so they fit
    plt.xticks(rotation=0, ha="center")  # keep all labels horizontal
    plt.tight_layout()  # auto-adjust spacing so nothing gets cut off
    return fig  # return the figure (caller passes to st.pyplot)


def make_table(series, col1_name, col2_name):
    """
    Converts a value_counts() result into a clean numbered table.
    e.g. turns {Cluster A: 50, Cluster B: 30} into a dataframe with rank 1, 2, 3...
    """
    tbl = series.reset_index()  # move index (category names) into a column
    tbl.columns = [col1_name, col2_name]  # rename columns to something readable
    tbl.index = range(1, len(tbl) + 1)  # number rows starting from 1 (not 0)
    return tbl


# --- MODE TOGGLE (appears at the very top of the app) ---
# User chooses between analysing one station or comparing multiple
mode = st.radio("Mode:", ["Single Station", "Multi-Station Compare"], horizontal=True, key="mode_toggle")


# =====================================================================
# SINGLE STATION MODE — one CSV upload, full analysis
# =====================================================================
if mode == "Single Station":

    # File uploader widget — only accepts .csv files
    uploaded_file = st.file_uploader("Upload SCC export (.csv)", type="csv")

    if uploaded_file is not None:  # only run if user has uploaded something
        df = pd.read_csv(uploaded_file)  # read the CSV into a pandas dataframe

        # Warn user if sensitive columns were found (they're auto-removed anyway)
        found_sensitive = [c for c in SENSITIVE_COLS if c in df.columns]
        if found_sensitive:
            st.warning(f"Sensitive columns found and removed: {', '.join(found_sensitive)}")

        df = clean_data(df)  # apply all cleaning (dimensions, sizes, dates)

        # Check all required columns exist
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")  # show what's missing
            st.info("Check your SCC export includes these fields, then re-upload.")
        elif len(df) == 0:
            st.warning("File has no data rows.")  # empty file check
        else:
            st.success(f"Data loaded — {len(df)} packages ready.")  # confirmation message

            date_range_text = get_date_range(df)  # e.g. "01 Jul 2025 - 07 Jul 2025"

            # --- SUMMARY METRICS (4 boxes always visible at the top) ---
            st.subheader(f"Quick Summary ({date_range_text})")
            c1, c2, c3, c4 = st.columns(4)  # 4 equal-width columns
            c1.metric("Total Lost", len(df))  # total number of rows = total lost parcels
            c2.metric("Worst Cluster", safe_top(df["Cluster"]))  # cluster with most losts
            c3.metric("Worst Aisle", safe_top(df["Aisle"]))  # aisle with most losts
            c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])  # DSP with most losts (truncated)

            if len(df) < 5:
                st.info("Small dataset — consider uploading a full week.")  # warn about tiny samples

            # --- TABS (the 7 main sections of the app) ---
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                ["Overview", "Location", "Rankings", "DSP & Cycle", "Time", "Export", "Bridge"]
            )

            # ==============================================================
            # TAB 1: OVERVIEW — size breakdown + cluster summary
            # ==============================================================
            with tab1:
                st.caption("Size breakdown and cluster summary.")  # small description text
                view = st.radio("Display:", ["Table", "Chart"], horizontal=True, key="ov_view")

                if view == "Table":
                    st.subheader("Size Breakdown")
                    st.write(df["Size Category"].value_counts())  # count per size category

                    st.subheader("Cluster × Size")
                    # Pivot table: rows=Cluster, columns=Size, values=count
                    tbl = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0)
                    tbl["Total"] = tbl.sum(axis=1)  # add a Total column
                    st.dataframe(tbl)  # display as interactive table
                else:
                    # Bar chart of size categories
                    size_counts = df["Size Category"].value_counts()
                    if len(size_counts) > 0:
                        colors = ["green", "orange", "red", "darkred", "grey"][:len(size_counts)]
                        fig = plot_bar(size_counts, "Size Category", "Lost Parcels",
                                       f"Lost by Size ({date_range_text})", color=colors)
                        st.pyplot(fig)

                    # Bar chart of clusters
                    cl_counts = df["Cluster"].dropna().value_counts()
                    if len(cl_counts) > 0:
                        st.pyplot(plot_bar(cl_counts, "Cluster", "Lost Parcels",
                                           f"Lost by Cluster ({date_range_text})"))

            # ==============================================================
            # TAB 2: LOCATION — drill into a specific cluster
            # ==============================================================
            with tab2:
                st.caption("Drill into a cluster to see aisle/zone hotspots.")
                clusters = sorted(df["Cluster"].dropna().unique())  # get unique clusters, sorted

                if clusters:
                    # Dropdown to pick which cluster to inspect
                    sel_cluster = st.selectbox("Cluster:", clusters, key="cl_sel")
                    filt = df[df["Cluster"] == sel_cluster]  # filter dataframe to just that cluster
                    st.write(f"{len(filt)} parcels in Cluster {sel_cluster}")

                    # Choose whether to view by Aisle or Sort Zone
                    view_by = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="view_by")
                    loc_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="loc_view")
                    loc_data = filt[view_by].dropna().value_counts()  # count per aisle/zone

                    if loc_view == "Chart":
                        if len(loc_data) > 0:
                            st.pyplot(plot_bar(loc_data, view_by, "Lost Parcels",
                                               f"Cluster {sel_cluster} by {view_by}", figsize=CHART_WIDE))

                        # Sub-filter: look at a specific size within this cluster
                        st.subheader(f"Size by Aisle in Cluster {sel_cluster}")
                        sel_size = st.selectbox("Size:", sorted(df["Size Category"].dropna().unique()), key="sz_sel")
                        size_filt = filt[filt["Size Category"] == sel_size]  # filter by size too
                        st.write(f"{len(size_filt)} '{sel_size}' parcels")
                        sz_data = size_filt["Aisle"].dropna().value_counts()
                        if len(sz_data) > 0:
                            st.pyplot(plot_bar(sz_data, "Aisle", "Lost Parcels",
                                               f"'{sel_size}' in Cluster {sel_cluster}", color="red"))
                        else:
                            st.info(f"No '{sel_size}' parcels in this cluster.")
                    else:
                        # Table view
                        if len(loc_data) > 0:
                            st.dataframe(make_table(loc_data, "Location", "Lost Parcels"))
                        st.subheader("Size Breakdown")
                        sz_tbl = filt.groupby(["Aisle", "Size Category"]).size().unstack(fill_value=0)
                        sz_tbl["Total"] = sz_tbl.sum(axis=1)
                        st.dataframe(sz_tbl)
                else:
                    st.info("No cluster data available.")

            # ==============================================================
            # TAB 3: RANKINGS — top 10 worst locations
            # ==============================================================
            with tab3:
                st.caption("Top 10 worst locations.")
                rank_by = st.selectbox("Rank by:", ["Sort Zone", "Aisle"], key="rank_sel")
                rank_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="rank_view")
                rank_data = df[rank_by].dropna().value_counts().head(10)  # top 10 only

                if len(rank_data) > 0:
                    if rank_view == "Chart":
                        # Horizontal bar chart — best for rankings
                        st.pyplot(plot_bar(rank_data, "Lost Parcels", rank_by,
                                           f"Top 10 {rank_by}s ({date_range_text})",
                                           color="darkred", horizontal=True))
                    else:
                        st.dataframe(make_table(rank_data, rank_by, "Lost Parcels"))
                else:
                    st.info("No data available.")

            # ==============================================================
            # TAB 4: DSP & CYCLE — which DSPs lose most, which cycles
            # ==============================================================
            with tab4:
                st.caption("DSP performance and cycle distribution.")
                dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dsp_view")
                dsp_data = df["DSP Name"].dropna().value_counts()  # count losts per DSP
                cycle_data = df["Assigned Cycle"].dropna().value_counts()  # count losts per cycle

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

            # ==============================================================
            # TAB 5: TIME — day-of-week analysis + tracking ID lookup
            # ==============================================================
            with tab5:
                st.caption("Day-of-week patterns and tracking ID lookup.")
                # Reindex forces Mon-Sun order even if some days have 0 losts
                day_data = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                time_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="time_view")

                if time_view == "Chart":
                    st.pyplot(plot_bar(day_data, "Day", "Lost Parcels",
                                       f"Lost by Day ({date_range_text})", color="green"))
                else:
                    st.dataframe(make_table(day_data, "Day", "Lost Parcels"))

                # Drill-down: select a day and see individual tracking IDs
                st.subheader("Parcel Details by Day")
                available_days = [d for d in DAY_ORDER if d in df["Day of Week"].values]  # only show days with data
                if available_days:
                    sel_day = st.selectbox("Day:", available_days, key="day_sel")
                    day_df = df[df["Day of Week"] == sel_day]  # filter to that day
                    st.write(f"{len(day_df)} parcels on {sel_day}")
                    # Only show useful columns in the table
                    show_cols = [c for c in ["Tracking ID", "Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"] if c in df.columns]
                    st.dataframe(day_df[show_cols])

            # ==============================================================
            # TAB 6: EXPORT — download cleaned data
            # ==============================================================
            with tab6:
                st.caption("Download cleaned data.")
                # Creates a download button that saves the cleaned dataframe as CSV
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "Lost_Parcels_Cleaned.csv", "text/csv")

            # ==============================================================
            # TAB 7: BRIDGE — auto-generated report with dynamic actions
            # ==============================================================
            with tab7:
                st.caption("Auto-generated bridge with dynamic action plans.")
                total = len(df)  # total number of lost parcels

                # --- GATHER ALL KEY STATS ---
                cl_counts = df["Cluster"].dropna().value_counts()  # losts per cluster
                ai_counts = df["Aisle"].dropna().value_counts()  # losts per aisle
                dsp_counts = df["DSP Name"].dropna().value_counts()  # losts per DSP
                sz_counts = df["Size Category"].value_counts()  # losts per size
                cy_counts = df["Assigned Cycle"].dropna().value_counts()  # losts per cycle
                day_counts = df["Day of Week"].dropna().value_counts()  # losts per day

                # Safely extract worst values (won't crash if data is empty)
                w_cluster = cl_counts.index[0] if len(cl_counts) > 0 else "N/A"
                w_cluster_n = cl_counts.values[0] if len(cl_counts) > 0 else 0
                w_cluster_pct = round(w_cluster_n / total * 100, 1) if total > 0 else 0

                w_aisle = ai_counts.index[0] if len(ai_counts) > 0 else "N/A"
                w_aisle_n = ai_counts.values[0] if len(ai_counts) > 0 else 0
                avg_aisle = ai_counts.mean() if len(ai_counts) > 0 else 1  # average losts per aisle

                w_dsp = dsp_counts.index[0] if len(dsp_counts) > 0 else "N/A"
                w_dsp_n = dsp_counts.values[0] if len(dsp_counts) > 0 else 0
                avg_dsp = dsp_counts.mean() if len(dsp_counts) > 0 else 1  # average losts per DSP
                dsp_mult = round(w_dsp_n / avg_dsp, 1) if avg_dsp > 0 else 1.0  # how many x above average

                w_size = sz_counts.index[0] if len(sz_counts) > 0 else "N/A"
                w_size_n = sz_counts.values[0] if len(sz_counts) > 0 else 0
                top_cycle = cy_counts.index[0] if len(cy_counts) > 0 else "N/A"
                top_cycle_n = cy_counts.values[0] if len(cy_counts) > 0 else 0
                w_day = day_counts.index[0] if len(day_counts) > 0 else "N/A"
                w_day_n = day_counts.values[0] if len(day_counts) > 0 else 0

                # --- DAILY BREAKDOWN (how many losts per calendar date) ---
                df["Date"] = df["Last Updated Time"].dt.strftime("%d/%m")  # format as DD/MM
                daily = df.groupby("Date").size()  # count per date
                daily_lines = "\n".join([f"{d} - {n} lost" for d, n in daily.items()])

                # --- TOP 3 CLUSTERS WITH THEIR TOP 3 AISLES ---
                cluster_details = ""
                for cl_name, cl_n in cl_counts.head(3).items():  # loop through top 3 clusters
                    pct = round(cl_n / total * 100, 1)  # percentage of total
                    # Find the top 3 aisles within this cluster
                    top_aisles = df[df["Cluster"] == cl_name]["Aisle"].dropna().value_counts().head(3)
                    aisles_str = ", ".join([f"{a} ({n})" for a, n in top_aisles.items()])
                    cluster_details += f"  Cluster {cl_name}: {cl_n} ({pct}%) — Aisles: {aisles_str}\n"

                # Top 3 DSPs formatted
                dsp_lines = "\n".join([f"  {d}: {n} ({round(n/total*100,1)}%)" for d, n in dsp_counts.head(3).items()])
                # All sizes formatted
                size_lines = "\n".join([f"  {s}: {n}" for s, n in sz_counts.items()])

                # State breakdown (only if the column exists in the export)
                if "State" in df.columns:
                    state_lines = "\n".join([f"  {s}: {n}" for s, n in df["State"].dropna().value_counts().items()])
                else:
                    state_lines = "  (Not in export)"

                # --- DYNAMIC ACTION PLANS (change based on patterns in the data) ---
                actions = []
                ac = lambda txt: actions.append(f"AC{len(actions)+1}: {txt}")  # shorthand: auto-numbers actions

                # Pattern 1: Is one cluster holding >40% of all losts?
                if w_cluster_pct > 40:
                    ac(f"Dedicated PS to Cluster {w_cluster} full shift — {w_cluster_pct}% of losts here.")
                else:
                    ac(f"PS rotation between top clusters ({', '.join([str(c) for c in cl_counts.head(3).index])}).")

                # Pattern 2: Is one DSP significantly worse than average?
                if dsp_mult >= 2.0:
                    ac(f"DSP {w_dsp} stand-down with leadership — {dsp_mult}x average.")
                elif dsp_mult >= 1.5:
                    ac(f"DSP {w_dsp} process briefing — {dsp_mult}x average.")
                else:
                    ac("Station-wide process refresher — losts spread evenly across DSPs.")

                # Pattern 3: What size is causing the most issues?
                if w_size in ["Large Oversize", "Small Oversize"]:
                    ac(f"Oversize stow audit in Aisle {w_aisle} — {w_size} most commonly lost.")
                elif w_size == "Small":
                    ac(f"Small parcel stow briefing — {w_size_n} small parcels lost.")
                else:
                    ac(f"Stow quality walk in Cluster {w_cluster} for {w_size} parcels.")

                # Pattern 4: Is one aisle way above average? (3x = needs physical inspection)
                if avg_aisle > 0 and w_aisle_n >= avg_aisle * 3:
                    ac(f"Physical inspection of Aisle {w_aisle} — {round(w_aisle_n/avg_aisle,1)}x average.")
                elif avg_aisle > 0 and w_aisle_n >= avg_aisle * 2:
                    ac(f"Increased PS in Aisle {w_aisle} — {w_aisle_n} losts, above average.")
                else:
                    ac("Daily PS huddle to review losts by aisle.")

                # Pattern 5: Are RELO cycles contributing more than 15%?
                relo_n = sum(cy_counts.get(c, 0) for c in cy_counts.index if "RELO" in str(c))
                if relo_n > total * 0.15:
                    ac(f"RELO process review — {relo_n} losts from relocation cycles.")

                # Pattern 6: Is one day of the week responsible for >30% of all losts?
                if len(daily) > 1 and w_day_n > total * 0.3:
                    ac(f"Review {w_day} staffing — {w_day_n} losts ({round(w_day_n/total*100,1)}% of total).")

                # --- ASSEMBLE THE FULL BRIDGE TEXT ---
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
                # Display as editable text box
                st.subheader("Draft Bridge")
                st.text_area("Edit:", value=bridge, height=400, key="bridge_txt")

                # --- ENHANCE WITH QUICK (copy-paste prompt) ---
                st.subheader("Enhance with Quick")
                prompt = f"""Write a Lost Parcels bridge for DRM2. Use RC1-4 for root causes, AC1-4 for actions. Be specific.

Data ({date_range_text}): Total={total}, Cluster={w_cluster} ({w_cluster_n}, {w_cluster_pct}%), Aisle={w_aisle} ({w_aisle_n}), DSP={w_dsp} ({w_dsp_n}, {dsp_mult}x avg), Size={w_size} ({w_size_n}), Cycle={top_cycle} ({top_cycle_n}), Day={w_day} ({w_day_n})
Daily: {daily_lines}
Clusters: {cluster_details}DSPs: {dsp_lines}

Generate specific actions a station manager would implement. Reference clusters, aisles, DSPs, sizes, days.
"""
                st.code(prompt, language="text")  # st.code has a built-in copy icon
                st.info("Copy icon (top-right) → Quick → Ctrl+V")


# =====================================================================
# MULTI-STATION COMPARE MODE — upload 2-5 CSVs, see side-by-side
# =====================================================================
else:
    st.subheader("Upload Station Data (2–5 stations)")
    st.caption("One SCC export per station. Station name auto-detected from 'Station' column.")

    # Slider lets user pick how many stations they want to compare
    num = st.slider("Stations to compare:", 2, 5, 2, key="num_st")

    # Create separate file uploaders arranged in columns
    uploaded = {}  # will store {index: file} for each uploaded file
    cols = st.columns(num)  # create N equal columns
    for i in range(num):
        with cols[i]:  # put each uploader in its own column
            f = st.file_uploader(f"Station {i+1}", type="csv", key=f"st_up_{i}")
            if f:
                uploaded[i] = f  # store the file if user uploaded one

    # Only proceed if at least 2 files are uploaded
    if len(uploaded) >= 2:
        stations = {}  # dict of {station_name: cleaned_dataframe}
        names = []  # ordered list of station names

        # Read and clean each uploaded file
        for i, file in uploaded.items():
            temp = clean_data(pd.read_csv(file))  # read CSV + clean
            name = get_station_name(temp, file.name)  # detect station name
            stations[name] = temp  # store with station name as key
            names.append(name)  # track order

        st.success(f"Loaded: {', '.join(names)}")  # confirmation

        # Summary metrics — one box per station showing total losts
        st.subheader("Station Summary")
        mcols = st.columns(len(names))
        for idx, name in enumerate(names):
            mcols[idx].metric(name, f"{len(stations[name])} lost")

        # --- TABS (same 7 as single station, but with comparison views) ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["Overview", "Location", "Rankings", "DSP & Cycle", "Time", "Export", "Bridge"]
        )

        # ==============================================================
        # TAB 1: OVERVIEW COMPARE — totals, sizes, clusters side by side
        # ==============================================================
        with tab1:
            st.caption("Compare totals, size, and clusters across stations.")
            view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_ov")

            if view == "Chart":
                # Bar chart: total losts per station
                st.subheader("Total Lost by Station")
                fig, ax = plt.subplots(figsize=CHART_FULL)
                ax.bar(names, [len(stations[n]) for n in names], color=STATION_COLORS[:len(names)])
                ax.set_ylabel("Lost Parcels", fontsize=8)
                ax.set_title("Total Lost Parcels by Station", fontsize=9)
                ax.tick_params(labelsize=7)
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig)

                # Grouped bar chart: size breakdown per station
                st.subheader("Size Breakdown by Station")
                fig2, ax2 = plt.subplots(figsize=CHART_WIDE)
                x = range(len(SIZE_ORDER))  # positions on x-axis
                w = 0.8 / len(names)  # bar width depends on how many stations
                for idx, name in enumerate(names):
                    # Count how many parcels of each size for this station
                    counts = [len(stations[name][stations[name]["Size Category"] == s]) for s in SIZE_ORDER]
                    offset = (idx - len(names)/2 + 0.5) * w  # offset bars so they sit side by side
                    ax2.bar([xi + offset for xi in x], counts, w, label=name, color=STATION_COLORS[idx])
                ax2.set_xticks(x)
                ax2.set_xticklabels(SIZE_ORDER, fontsize=7)
                ax2.set_ylabel("Lost Parcels", fontsize=8)
                ax2.tick_params(labelsize=7)
                ax2.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig2)

                # Individual cluster charts per station
                st.subheader("Clusters per Station")
                for idx, name in enumerate(names):
                    cl = stations[name]["Cluster"].dropna().value_counts()
                    if len(cl) > 0:
                        st.pyplot(plot_bar(cl, "Cluster", "Lost", f"{name} — Clusters",
                                           color=STATION_COLORS[idx], figsize=CHART_SMALL))
            else:
                # Table views
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

        # ==============================================================
        # TAB 2: LOCATION COMPARE — cluster drill-down per station
        # ==============================================================
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

                    # Aisle breakdown for this cluster
                    ai = filt["Aisle"].dropna().value_counts().head(10)
                    if len(ai) > 0:
                        st.pyplot(plot_bar(ai, "Aisle", "Lost", f"{name} — Cluster {sel} Aisles",
                                           color=STATION_COLORS[idx], figsize=CHART_SMALL))
                    # Sort Zone breakdown for this cluster
                    zn = filt["Sort Zone"].dropna().value_counts().head(10)
                    if len(zn) > 0:
                        st.pyplot(plot_bar(zn, "Sort Zone", "Lost", f"{name} — Cluster {sel} Zones",
                                           color=STATION_COLORS[idx], figsize=CHART_SMALL))
                else:
                    st.info(f"No cluster data for {name}.")
                st.markdown("---")  # divider between stations

        # ==============================================================
        # TAB 3: RANKINGS COMPARE — top 10 per station
        # ==============================================================
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

        # ==============================================================
        # TAB 4: DSP & CYCLE COMPARE
        # ==============================================================
        with tab4:
            st.caption("DSP and cycle comparison across stations.")
            dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_dsp_v")

            if dsp_view == "Chart":
                # DSP charts per station
                for idx, name in enumerate(names):
                    dsp = stations[name]["DSP Name"].dropna().value_counts().head(10)
                    if len(dsp) > 0:
                        st.pyplot(plot_bar(dsp, "DSP", "Lost", f"{name} — Worst DSPs",
                                           color=STATION_COLORS[idx]))

                # Grouped cycle comparison (all stations on one chart)
                st.subheader("Cycle Comparison")
                # Get all unique cycles across all stations
                all_cycles = sorted(set().union(*[stations[n]["Assigned Cycle"].dropna().unique() for n in names]))
                if all_cycles:
                    fig, ax = plt.subplots(figsize=CHART_FULL)
                    x = range(len(all_cycles))
                    w = 0.8 / len(names)
                    for idx, name in enumerate(names):
                        cy = stations[name]["Assigned Cycle"].dropna().value_counts()
                        counts = [cy.get(c, 0) for c in all_cycles]  # 0 if cycle doesn't exist for this station
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
                        st.write("**DSPs:**")
                        st.dataframe(make_table(dsp, "DSP", "Lost Parcels"))
                    cy = stations[name]["Assigned Cycle"].dropna().value_counts()
                    if len(cy) > 0:
                        st.write("**Cycles:**")
                        st.dataframe(make_table(cy, "Cycle", "Lost Parcels"))
                    st.markdown("---")

        # ==============================================================
        # TAB 5: TIME COMPARE — day-of-week overlaid line + individual bars
        # ==============================================================
        with tab5:
            st.caption("Day-of-week patterns across stations.")
            time_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="mc_time_v")

            if time_view == "Chart":
                # Overlaid line chart — all stations on one graph for easy comparison
                st.subheader("Day of Week — All Stations")
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

                # Individual day bar charts per station
                for idx, name in enumerate(names):
                    if "Day of Week" in stations[name].columns:
                        dd = stations[name]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        st.pyplot(plot_bar(dd, "Day", "Lost", f"{name} — by Day",
                                           color=STATION_COLORS[idx], figsize=CHART_SMALL))
            else:
                # Table: one column per station, rows = days
                st.subheader("Day of Week Comparison")
                day_all = {}
                for name in names:
                    if "Day of Week" in stations[name].columns:
                        day_all[name] = stations[name]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0).values
                st.dataframe(pd.DataFrame(day_all, index=DAY_ORDER))

        # ==============================================================
        # TAB 6: EXPORT COMPARE — download per station or combined
        # ==============================================================
        with tab6:
            st.caption("Download individual or combined station data.")

            st.subheader("Individual Downloads")
            for name in names:
                st.download_button(f"Download {name}", stations[name].to_csv(index=False),
                                   f"Lost_{name}_Cleaned.csv", "text/csv", key=f"dl_{name}")

            st.subheader("Combined Download")
            # Merge all stations into one file with a "Station_Name" column to tell them apart
            combined = pd.concat([stations[n].assign(Station_Name=n) for n in names], ignore_index=True)
            st.download_button("Download All Stations Combined", combined.to_csv(index=False),
                               "Lost_All_Stations_Combined.csv", "text/csv", key="dl_all")

        # ==============================================================
        # TAB 7: BRIDGE COMPARE — summary table + best vs worst analysis
        # ==============================================================
        with tab7:
            st.caption("Cross-station comparison summary.")

            # Build comparison table with key stats per station
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

            # Identify best and worst performing stations
            st.subheader("Key Differences")
            best = min(names, key=lambda n: len(stations[n]))  # station with fewest losts
            worst = max(names, key=lambda n: len(stations[n]))  # station with most losts
            gap = len(stations[worst]) - len(stations[best])  # difference in count
            gap_pct = round(gap / len(stations[worst]) * 100, 1) if len(stations[worst]) > 0 else 0

            st.write(f"**Best:** {best} ({len(stations[best])} losts)")
            st.write(f"**Worst:** {worst} ({len(stations[worst])} losts)")
            st.write(f"**Gap:** {gap} parcels ({gap_pct}% difference)")

            # Quick prompt for AI-enhanced comparison
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
        st.warning("Upload at least 2 files to compare.")
    else:
        st.info("Upload CSV files above to begin.")
