import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration — must be the very first Streamlit command
# layout="wide" uses full screen width instead of narrow centered column
# page_icon shows in the browser tab
st.set_page_config(page_title="SCC Lost Heatmap", page_icon="📦", layout="wide")

st.title("SCC Lost Parcel Heatmap")
st.markdown("---")  # horizontal divider line

# --- FILE UPLOAD ---
# type="csv" restricts the upload button to only accept CSV files
uploaded_file = st.file_uploader("Upload SCC export as a .csv file please to generate the heatmap", type="csv")

# Everything below only runs if a file has been uploaded
# Without this check, the app would crash trying to process nothing
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # reads CSV into a DataFrame (table)

    # ====================================================================
    # SENSITIVE DATA CHECK
    # Automatically detects and removes columns containing personal info
    # This protects privacy since the app is publicly hosted
    # ====================================================================
    sensitive_columns = [
        "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
        "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
        "Payment Method", "District", "Scheduled Delivery End Time"
    ]

    # Loop through sensitive list — keep only ones that actually exist in the uploaded file
    found_sensitive = [col for col in sensitive_columns if col in df.columns]

    if found_sensitive:
        st.warning(f"Sensitive Information Column Titles Uploaded: {', '.join(found_sensitive)}")
        st.info("These columns have been automatically removed")
        df = df.drop(columns=found_sensitive)  # permanently removes from the data

    # ====================================================================
    # COLUMN VALIDATION
    # Checks all required columns exist before processing
    # Tells user exactly what's missing so they can fix their SCC export
    # ====================================================================
    required_columns = [
        "Tracking ID", "Sort Zone", "Aisle", "Cluster",
        "Package Length", "Package Width", "Package Height",
        "DSP Name", "Assigned Cycle", "Last Updated Time"
    ]
    # Find any required columns NOT present in the uploaded file
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error("Missing required columns:")
        for col in missing:
            st.write(f"   - {col}")
        st.info("Please check your SCC export filters include these fields, then re-upload.")
    else:
        st.success(f"Data loaded - {df.shape[0]} packages ready for analysis.")

        # Check for empty dataset
        if len(df) == 0:
            st.warning("The uploaded file has no data rows. Please check your export.")
        else:

            # ================================================================
            # DATA CLEANING - PACKAGE SIZE GROUPING
            # SCC exports dimensions as text ("23.50 cm") or sometimes as numbers
            # We need them as numbers to calculate size categories
            # ================================================================
            for col in ["Package Length", "Package Width", "Package Height"]:
                if df[col].dtype == "object":  # if column contains text
                    df[col] = df[col].str.replace(" cm", "").str.replace("cm", "").astype(float)
                else:
                    df[col] = df[col].astype(float)  # already a number, just ensure it's float

            # For each row, find the longest of the 3 dimensions
            # axis=1 means "look across columns" (not down rows)
            df["Longest Side"] = df[["Package Length", "Package Width", "Package Height"]].max(axis=1)

            # Function to assign Amazon UK size tier based on longest dimension
            # pd.isna() check prevents crash if a dimension value is missing
            def get_size(longest):
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

            # .apply() runs the get_size function on every single row
            df["Size Category"] = df["Longest Side"].apply(get_size)

            # ================================================================
            # DATE RANGE CALCULATION
            # Used in chart titles so users know what time period they're looking at
            # errors="coerce" converts any badly formatted dates to NaN instead of crashing
            # ================================================================
            df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], errors="coerce")
            start_date = df["Last Updated Time"].min().strftime("%d %b %Y")
            end_date = df["Last Updated Time"].max().strftime("%d %b %Y")

            # Single day check — prevents ugly "27 Jul 2026 - 27 Jul 2026" titles
            if start_date == end_date:
                date_range_text = f"{start_date}"
            else:
                date_range_text = f"{start_date} - {end_date}"

            # ================================================================
            # SUMMARY STATS - Always visible at top (like PSE dashboard KPIs)
            # Uses st.columns() for side-by-side layout
            # Uses st.metric() for big number display cards
            # ================================================================
            st.subheader(f"Quick Summary ({date_range_text})")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Lost", len(df))

            # Safe access for each stat — shows "N/A" if data is empty/missing
            if len(df["Cluster"].dropna()) > 0:
                col2.metric("Worst Cluster", df["Cluster"].value_counts().index[0])
            else:
                col2.metric("Worst Cluster", "N/A")

            if len(df["Aisle"].dropna()) > 0:
                col3.metric("Worst Aisle", df["Aisle"].value_counts().index[0])
            else:
                col3.metric("Worst Aisle", "N/A")

            if len(df["DSP Name"].dropna()) > 0:
                top_dsp_name = df["DSP Name"].dropna().value_counts().index[0]
                col4.metric("Top DSP", top_dsp_name[:15])  # [:15] prevents long names overflowing
            else:
                col4.metric("Top DSP", "N/A")

            # Small dataset warning — analysis is less meaningful with few data points
            if len(df) < 5:
                st.info("Small dataset — analysis may be less meaningful. Consider uploading a full week for better insights.")

            # ================================================================
            # TABS - Organises all content into clickable sections
            # Prevents overwhelming the user with one long scrolling page
            # ================================================================
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                ["Overview", "Location", "Rankings", "DSP & Cycle", "Time", "Export", "Bridge"]
            )

            # ================================================================
            # TAB 1: OVERVIEW
            # Size breakdown and cluster summary — high level picture
            # ================================================================
            with tab1:
                st.caption("Size breakdown of lost parcels and summary by cluster.")

                # One radio toggle controls everything in this tab
                overview_display = st.radio("Display as:", ["Table", "Chart"], horizontal=True, key="overview_display")

                if overview_display == "Table":
                    st.subheader("Lost Parcel Size Breakdown")
                    st.write(df["Size Category"].value_counts())

                    st.subheader("Lost Parcels by Cluster and Size")
                    # groupby + unstack creates a pivot table (like a cross-tab in Excel)
                    # rows = clusters, columns = size categories, values = count
                    summary_tbl = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0)
                    summary_tbl["Total"] = summary_tbl.sum(axis=1)  # row totals
                    st.dataframe(summary_tbl)
                else:
                    st.subheader("Lost Parcels by Size")
                    size_counts = df["Size Category"].value_counts()
                    if len(size_counts) > 0:
                        fig_ov, ax_ov = plt.subplots(figsize=(8, 4))
                        # Color list adapts to however many size categories exist
                        colors = ["green", "orange", "red", "darkred", "grey"][:len(size_counts)]
                        ax_ov.bar(size_counts.index, size_counts.values, color=colors)
                        ax_ov.set_xlabel("Size Category")
                        ax_ov.set_ylabel("Lost Parcels")
                        ax_ov.set_title(f"Lost Parcels by Size ({date_range_text})")
                        plt.xticks(rotation=0, ha="center")
                        plt.tight_layout()  # prevents labels being cut off at edges
                        st.pyplot(fig_ov)

                    st.subheader("Lost Parcels by Cluster")
                    cluster_counts = df["Cluster"].dropna().value_counts()
                    if len(cluster_counts) > 0:
                        fig_cl, ax_cl = plt.subplots(figsize=(8, 4))
                        ax_cl.bar(cluster_counts.index, cluster_counts.values, color="steelblue")
                        ax_cl.set_xlabel("Cluster")
                        ax_cl.set_ylabel("Lost Parcels")
                        ax_cl.set_title(f"Lost Parcels by Cluster ({date_range_text})")
                        plt.xticks(rotation=0, ha="center")
                        plt.tight_layout()
                        st.pyplot(fig_cl)

            # ================================================================
            # TAB 2: LOCATION
            # Drill-down: user picks a cluster → sees aisles/zones within it
            # Plus size-by-zone analysis for spotting problem areas
            # ================================================================
            with tab2:
                st.caption("Drill into a cluster to see which aisles or zones have the most losts. Filter by package size to spot problem areas.")

                clusters_available = sorted(df["Cluster"].dropna().unique())

                if len(clusters_available) > 0:
                    # Dropdown listing all clusters — sorted alphabetically
                    selected_cluster = st.selectbox("Select Cluster:", clusters_available, key="cluster_select")

                    # Filter entire table to only rows matching selected cluster
                    # This is like applying a filter in Excel
                    filtered_df = df[df["Cluster"] == selected_cluster]
                    st.write(f"Showing {len(filtered_df)} lost parcels in Cluster {selected_cluster}")

                    # Second dropdown — choose detail level within the cluster
                    view_detail = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="view_select")

                    # Toggle for this entire tab
                    location_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="location_display")

                    # Count lost parcels per location in the selected cluster
                    chart_data = filtered_df[view_detail].dropna().value_counts()

                    if location_display == "Chart":
                        if len(chart_data) > 0:
                            fig, ax = plt.subplots(figsize=(14, 5))
                            ax.bar(chart_data.index, chart_data.values)
                            ax.set_xlabel(view_detail)
                            ax.set_ylabel("Lost Parcels")
                            ax.set_title(f"Lost Parcels in Cluster {selected_cluster} by {view_detail}")
                            # Sort Zone names are long — angle them. Aisle names are short — keep flat
                            if view_detail == "Sort Zone":
                                plt.xticks(rotation=45, ha="right")
                            else:
                                plt.xticks(rotation=0, ha="center")
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.info("No location data available for this cluster.")

                        # Size by zone — which size parcels go missing where
                        st.subheader(f"Package Size by Aisle in Cluster {selected_cluster}")
                        selected_size = st.selectbox("Select Size Category:", sorted(df["Size Category"].dropna().unique()), key="size_select")
                        size_df = filtered_df[filtered_df["Size Category"] == selected_size]
                        st.write(f"{len(size_df)} '{selected_size}' parcels lost in Cluster {selected_cluster}")
                        size_zone_data = size_df["Aisle"].dropna().value_counts()

                        if len(size_zone_data) > 0:
                            fig6, ax6 = plt.subplots(figsize=(12, 5))
                            ax6.bar(size_zone_data.index, size_zone_data.values, color="red")
                            ax6.set_xlabel("Aisle")
                            ax6.set_ylabel("Lost Parcels")
                            ax6.set_title(f"'{selected_size}' Parcels Lost by Aisle in Cluster {selected_cluster}")
                            plt.xticks(rotation=0, ha="center")
                            plt.tight_layout()
                            st.pyplot(fig6)
                        else:
                            st.info(f"No '{selected_size}' parcels lost in Cluster {selected_cluster}")
                    else:
                        # Table view — shows ranked list and size pivot
                        if len(chart_data) > 0:
                            st.subheader(f"Lost Parcels by {view_detail}")
                            location_table = chart_data.reset_index()
                            location_table.columns = ["Location", "Lost Parcels"]
                            location_table.index = range(1, len(location_table) + 1)  # rank from 1
                            st.dataframe(location_table)

                        st.subheader(f"Package Size Breakdown in Cluster {selected_cluster}")
                        cluster_size_tbl = filtered_df.groupby(["Aisle", "Size Category"]).size().unstack(fill_value=0)
                        cluster_size_tbl["Total"] = cluster_size_tbl.sum(axis=1)
                        st.dataframe(cluster_size_tbl)
                else:
                    st.info("No cluster data available.")

            # ================================================================
            # TAB 3: RANKINGS
            # Top 10 worst locations — quick priority list for action
            # ================================================================
            with tab3:
                st.caption("See the worst performing locations ranked by number of lost parcels.")

                rank_view = st.selectbox("Rank by:", ["Sort Zone", "Aisle"], key="rank_select")
                rank_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="rank_display")

                # .head(10) takes only the top 10 — if fewer than 10 exist, shows all
                rank_data = df[rank_view].dropna().value_counts().head(10)

                if len(rank_data) > 0:
                    if rank_display == "Chart":
                        fig8, ax8 = plt.subplots(figsize=(12, 5))
                        # .barh() = horizontal bars — better for ranked lists with long labels
                        ax8.barh(rank_data.index, rank_data.values, color="darkred")
                        ax8.set_xlabel("Lost Parcels")
                        ax8.set_ylabel(rank_view)
                        ax8.set_title(f"Top 10 {rank_view}s with Most Lost Parcels ({date_range_text})")
                        ax8.invert_yaxis()  # #1 worst at the top, reads like a leaderboard
                        plt.tight_layout()
                        st.pyplot(fig8)
                    else:
                        rank_table = rank_data.reset_index()
                        rank_table.columns = ["Location", "Lost Parcels"]
                        rank_table.index = range(1, len(rank_table) + 1)  # rank from 1
                        st.dataframe(rank_table)
                else:
                    st.info("No location data available for ranking.")

            # ================================================================
            # TAB 4: DSP & CYCLE
            # Which DSPs lose most + dispatch cycle comparison
            # ================================================================
            with tab4:
                st.caption("See which DSPs lose the most parcels and compare performance across dispatch cycles.")

                dsp_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="dsp_display")

                dsp_data = df["DSP Name"].dropna().value_counts()
                cycle_data = df["Assigned Cycle"].dropna().value_counts()

                if dsp_display == "Chart":
                    st.subheader(f"Lost Parcels by DSP ({date_range_text})")
                    if len(dsp_data) > 0:
                        fig2, ax2 = plt.subplots(figsize=(12, 5))
                        ax2.bar(dsp_data.index, dsp_data.values, color="orange")
                        ax2.set_xlabel("DSP")
                        ax2.set_ylabel("Lost Parcels")
                        ax2.set_title(f"Lost Parcels by DSP ({date_range_text})")
                        plt.xticks(rotation=45, ha="right")  # angled — DSP names are long
                        plt.tight_layout()
                        st.pyplot(fig2)
                    else:
                        st.info("No DSP data available.")

                    st.subheader(f"Lost Parcels by Cycle ({date_range_text})")
                    if len(cycle_data) > 0:
                        fig4, ax4 = plt.subplots(figsize=(10, 5))
                        ax4.bar(cycle_data.index, cycle_data.values, color="purple")
                        ax4.set_xlabel("Cycle")
                        ax4.set_ylabel("Lost Parcels")
                        ax4.set_title(f"Lost Parcels by Cycle ({date_range_text})")
                        plt.xticks(rotation=0, ha="center")
                        plt.tight_layout()
                        st.pyplot(fig4)
                    else:
                        st.info("No cycle data available.")
                else:
                    st.subheader(f"Lost Parcels by DSP ({date_range_text})")
                    if len(dsp_data) > 0:
                        dsp_table = dsp_data.reset_index()
                        dsp_table.columns = ["DSP", "Lost Parcels"]
                        dsp_table.index = range(1, len(dsp_table) + 1)
                        st.dataframe(dsp_table)
                    else:
                        st.info("No DSP data available.")

                    st.subheader(f"Lost Parcels by Cycle ({date_range_text})")
                    if len(cycle_data) > 0:
                        cycle_table = cycle_data.reset_index()
                        cycle_table.columns = ["Cycle", "Lost Parcels"]
                        cycle_table.index = range(1, len(cycle_table) + 1)
                        st.dataframe(cycle_table)
                    else:
                        st.info("No cycle data available.")

            # ================================================================
            # TAB 5: TIME
            # Day of week patterns + tracking ID lookup for investigation
            # ================================================================
            with tab5:
                st.caption("See which days of the week have the most losts. Select a day to view individual tracking IDs.")

                # .dt.day_name() extracts weekday name from timestamp (e.g. "Saturday")
                df["Day of Week"] = df["Last Updated Time"].dt.day_name()
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                # .reindex() forces Mon-Sun order; fill_value=0 shows 0 for days with no losts
                day_data = df["Day of Week"].dropna().value_counts().reindex(day_order, fill_value=0)

                time_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="time_display")

                if time_display == "Chart":
                    st.subheader(f"Lost Parcels by Day of Week ({date_range_text})")
                    fig3, ax3 = plt.subplots(figsize=(10, 5))
                    ax3.bar(day_data.index, day_data.values, color="green")
                    ax3.set_xlabel("Day of Week")
                    ax3.set_ylabel("Lost Parcels")
                    ax3.set_title(f"Lost Parcels by Day of Week ({date_range_text})")
                    plt.xticks(rotation=0, ha="center")
                    plt.tight_layout()
                    st.pyplot(fig3)
                else:
                    st.subheader(f"Lost Parcels by Day of Week ({date_range_text})")
                    day_table = day_data.reset_index()
                    day_table.columns = ["Day", "Lost Parcels"]
                    day_table.index = range(1, len(day_table) + 1)
                    st.dataframe(day_table)

                # Tracking ID lookup — for investigating specific packages
                st.subheader("Lost Parcel Details by Day")
                # Only show days that have data — prevents selecting empty days
                available_days = [d for d in day_order if d in df["Day of Week"].values]
                if len(available_days) > 0:
                    selected_day = st.selectbox("Select Day:", available_days, key="day_select")
                    day_df = df[df["Day of Week"] == selected_day]
                    st.write(f"{len(day_df)} parcels lost on {selected_day}")
                    # Only show columns that exist in the data (defensive)
                    display_cols = [col for col in ["Tracking ID", "Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"] if col in df.columns]
                    st.dataframe(day_df[display_cols])
                else:
                    st.info("No day data available.")

            # ================================================================
            # TAB 6: EXPORT
            # Download cleaned data as CSV — useful for sharing or further analysis
            # ================================================================
            with tab6:
                st.caption("Download the cleaned data with sensitive information removed and size categories added.")

                st.subheader("Export Data")
                # .to_csv(index=False) converts table to CSV text without row numbers
                csv = df.to_csv(index=False)

                st.download_button(
                    label="Download cleaned data as CSV",
                    data=csv,
                    file_name="Lost_Parcels_Cleaned.csv",
                    mime="text/csv"  # tells browser this is a CSV file
                )

            # ================================================================
            # TAB 7: BRIDGE
            # Auto-generates a lost parcels bridge with DYNAMIC action plans
            # Actions change based on actual patterns detected in the data
            # ================================================================
            with tab7:
                st.caption("Auto-generates a lost parcels bridge draft. Copy the Quick prompt for AI-enhanced action plans.")

                # --- CALCULATE KEY STATS (all with safe access) ---
                total_lost = len(df)

                # Cluster stats — safe access in case column is all NaN
                cluster_counts = df["Cluster"].dropna().value_counts()
                if len(cluster_counts) > 0:
                    worst_cluster = cluster_counts.index[0]
                    worst_cluster_count = cluster_counts.values[0]
                    worst_cluster_pct = round(worst_cluster_count / total_lost * 100, 1)
                else:
                    worst_cluster = "N/A"
                    worst_cluster_count = 0
                    worst_cluster_pct = 0

                # Aisle stats
                aisle_counts = df["Aisle"].dropna().value_counts()
                if len(aisle_counts) > 0:
                    worst_aisle = aisle_counts.index[0]
                    worst_aisle_count = aisle_counts.values[0]
                    avg_aisle_count = aisle_counts.mean()
                else:
                    worst_aisle = "N/A"
                    worst_aisle_count = 0
                    avg_aisle_count = 1

                # DSP stats
                dsp_counts = df["DSP Name"].dropna().value_counts()
                if len(dsp_counts) > 0:
                    worst_dsp = dsp_counts.index[0]
                    worst_dsp_count = dsp_counts.values[0]
                    avg_dsp_count = dsp_counts.mean()
                    worst_dsp_multiple = round(worst_dsp_count / avg_dsp_count, 1) if avg_dsp_count > 0 else 1.0
                else:
                    worst_dsp = "N/A"
                    worst_dsp_count = 0
                    worst_dsp_multiple = 1.0

                # Size stats
                size_counts_bridge = df["Size Category"].value_counts()
                worst_size = size_counts_bridge.index[0] if len(size_counts_bridge) > 0 else "N/A"
                worst_size_count = size_counts_bridge.values[0] if len(size_counts_bridge) > 0 else 0

                # Cycle stats
                cycle_counts_bridge = df["Assigned Cycle"].dropna().value_counts()
                top_cycle = cycle_counts_bridge.index[0] if len(cycle_counts_bridge) > 0 else "N/A"
                top_cycle_count = cycle_counts_bridge.values[0] if len(cycle_counts_bridge) > 0 else 0

                # Day stats
                df["Day of Week"] = df["Last Updated Time"].dt.day_name()
                day_counts_bridge = df["Day of Week"].dropna().value_counts()
                worst_day = day_counts_bridge.index[0] if len(day_counts_bridge) > 0 else "N/A"
                worst_day_count = day_counts_bridge.values[0] if len(day_counts_bridge) > 0 else 0

                # --- DAILY BREAKDOWN (like PS Effectiveness bridge has) ---
                df["Date"] = df["Last Updated Time"].dt.strftime("%d/%m")
                daily_counts = df.groupby("Date").size()
                daily_lines = "\n".join([f"{date} - {count} lost" for date, count in daily_counts.items()])

                # --- TOP 3 CLUSTERS WITH TOP 3 AISLES EACH ---
                top_clusters = cluster_counts.head(3)
                cluster_details = ""
                for cluster_name, cluster_count in top_clusters.items():
                    cluster_pct = round(cluster_count / total_lost * 100, 1)
                    cluster_aisles = df[df["Cluster"] == cluster_name]["Aisle"].dropna().value_counts().head(3)
                    aisle_list = ", ".join([f"{aisle} ({count})" for aisle, count in cluster_aisles.items()])
                    cluster_details += f"  Cluster {cluster_name}: {cluster_count} parcels ({cluster_pct}%) — Top aisles: {aisle_list}\n"

                # --- TOP 3 DSPs ---
                top_dsps = dsp_counts.head(3)
                dsp_lines = "\n".join([f"  {dsp}: {count} parcels ({round(count/total_lost*100,1)}%)" for dsp, count in top_dsps.items()])

                # --- SIZE BREAKDOWN ---
                size_lines = "\n".join([f"  {size}: {count}" for size, count in size_counts_bridge.items()])

                # --- STATE BREAKDOWN (only if column exists in the data) ---
                if "State" in df.columns:
                    state_breakdown = df["State"].dropna().value_counts()
                    state_lines = "\n".join([f"  {state}: {count}" for state, count in state_breakdown.items()])
                else:
                    state_lines = "  (State data not included in export)"

                # ============================================================
                # DYNAMIC ACTION PLANS
                # These change based on the actual patterns in the data
                # Different conditions trigger different recommended actions
                # ============================================================
                actions = []

                # PATTERN 1: One cluster dominates (>40% of losts)
                if worst_cluster_pct > 40:
                    actions.append(f"AC{len(actions)+1}: Dedicated PS resource assigned to Cluster {worst_cluster} for full shift coverage — {worst_cluster_pct}% of all losts concentrated here.")
                else:
                    actions.append(f"AC{len(actions)+1}: PS resource to rotate between top clusters ({', '.join([str(c) for c in cluster_counts.head(3).index])}) — losts are spread across multiple areas.")

                # PATTERN 2: One DSP significantly above average (>2x)
                if worst_dsp_multiple >= 2.0:
                    actions.append(f"AC{len(actions)+1}: DSP {worst_dsp} to attend stand-down meeting with station leadership — {worst_dsp_multiple}x station average for losts. Review pick-and-stage compliance.")
                elif worst_dsp_multiple >= 1.5:
                    actions.append(f"AC{len(actions)+1}: DSP {worst_dsp} to be briefed on correct pick-and-stage process — currently {worst_dsp_multiple}x station average.")
                else:
                    actions.append(f"AC{len(actions)+1}: Station-wide process refresher during AM standup — losts spread evenly across DSPs, indicating a systemic process gap rather than individual DSP issue.")

                # PATTERN 3: Size-specific issue
                if worst_size == "Large Oversize" or worst_size == "Small Oversize":
                    actions.append(f"AC{len(actions)+1}: OS to conduct oversize stow audit in Aisle {worst_aisle} — verify shelf capacity and stow compliance for {worst_size} items which are most commonly lost.")
                elif worst_size == "Small":
                    actions.append(f"AC{len(actions)+1}: AA briefing on small parcel stow procedure — ensuring items placed fully inside bins and not balanced on edges. {worst_size_count} small parcels lost this period.")
                else:
                    actions.append(f"AC{len(actions)+1}: OS to walk Cluster {worst_cluster} during sortation checking stow quality for {worst_size} parcels — {worst_size_count} lost this period.")

                # PATTERN 4: One aisle significantly above average (>3x)
                if worst_aisle_count >= avg_aisle_count * 3:
                    actions.append(f"AC{len(actions)+1}: Physical inspection of Aisle {worst_aisle} requested — {worst_aisle_count} losts ({round(worst_aisle_count/avg_aisle_count, 1)}x average). Check for structural issues, overcrowding, or incorrect labelling.")
                elif worst_aisle_count >= avg_aisle_count * 2:
                    actions.append(f"AC{len(actions)+1}: Increased PS presence in Aisle {worst_aisle} during pick — {worst_aisle_count} losts, significantly above station average.")
                else:
                    actions.append(f"AC{len(actions)+1}: Daily PS huddle to review previous day's losts by aisle and assign targeted investigations based on volume and repeat locations.")

                # PATTERN 5: RELO cycle contributing significantly
                if "RELO" in str(top_cycle) or (len(cycle_counts_bridge) > 1 and "RELO" in str(cycle_counts_bridge.index[1] if len(cycle_counts_bridge) > 1 else "")):
                    relo_count = sum(cycle_counts_bridge[c] for c in cycle_counts_bridge.index if "RELO" in str(c))
                    if relo_count > total_lost * 0.15:
                        actions.append(f"AC{len(actions)+1}: RELO process review — {relo_count} losts from relocation cycles. Ensure packages scanned correctly before transfer and manifests reconciled at destination.")

                # PATTERN 6: One day significantly worse
                if worst_day_count > total_lost * 0.3 and len(daily_counts) > 1:
                    actions.append(f"AC{len(actions)+1}: Review {worst_day} staffing levels vs package volume — {worst_day_count} losts ({round(worst_day_count/total_lost*100,1)}% of total). Potential understaffing driving process shortcuts.")

                # Format all actions into text
                actions_text = "\n".join(actions)

                # --- BUILD FULL BRIDGE TEXT ---
                bridge_text = f"""Lost Parcels Bridge - DRM2
{date_range_text}

Lost (Total): {total_lost}

Daily Breakdown:
{daily_lines}

RC1) Location Concentration:
{cluster_details}
RC2) DSP Performance:
{dsp_lines}

RC3) Package Size:
{size_lines}

RC4) Package Status:
{state_lines}

{actions_text}
"""

                # --- DISPLAY ---
                st.subheader("Draft Bridge")
                st.text_area("Edit as needed:", value=bridge_text, height=500)

                # --- ENHANCE WITH QUICK ---
                st.subheader("Enhance with Quick")
                st.write("For unique, AI-generated action plans, copy the prompt below and paste into Quick online.")

                quick_prompt = f"""Write me a Lost Parcels bridge for DRM2 in the same style as a PS Effectiveness bridge. Use RC1, RC2, RC3, RC4 for root causes and AC1, AC2, AC3, AC4 for action plans. Make actions specific, realistic, and different from generic templates.

Data ({date_range_text}):
- Total lost: {total_lost}
- Worst cluster: {worst_cluster} ({worst_cluster_count} parcels, {worst_cluster_pct}%)
- Worst aisle: {worst_aisle} ({worst_aisle_count} parcels)
- Worst DSP: {worst_dsp} ({worst_dsp_count} parcels, {worst_dsp_multiple}x average)
- Most common size lost: {worst_size} ({worst_size_count} parcels)
- Top cycle: {top_cycle} ({top_cycle_count} parcels)
- Worst day: {worst_day} ({worst_day_count} parcels)
- Daily breakdown: {daily_lines}

Top 3 clusters:
{cluster_details}
Top 3 DSPs:
{dsp_lines}

Generate realistic, specific action plans that a station manager would actually implement. Reference specific clusters, aisles, DSPs, sizes, and days from the data. Keep the tone professional and concise. Do NOT use generic actions — make each one specific to these patterns.
"""

                st.code(quick_prompt, language="text")  # has built-in copy icon in top-right
                st.info("Click the copy icon (top-right corner above) → Open Quick → Ctrl+V")
