import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("SCC Lost Parcel Heatmap")

uploaded_file = st.file_uploader("Upload SCC export as a .csv file please to generate the heatmap", type="csv") # only accepts .csv files

if uploaded_file is not None: # only runs if user has uploaded a file
    df = pd.read_csv(uploaded_file) # reads the CSV into a table

    # Sensitive Data Check --------------------------------------------
    sensitive_columns = ["Last Scan By", "Driver Id", "Holder Name", "City", "Postal", "Province", "Ordering Order ID", "Order Amount", "Receivable Amount", "Payment Method", "District", "Scheduled Delivery End Time"]

    found_sensitive = [col for col in sensitive_columns if col in df.columns] # finds which sensitive columns exist in uploaded file

    if found_sensitive:
        st.warning(f"Sensitive Information Column Titles Uploaded: {', '.join(found_sensitive)}") # yellow warning
        st.info("These columns have been automatically removed") # blue info box
        df = df.drop(columns=found_sensitive) # deletes sensitive columns

    # Column Validation -----------------------------------------------
    required_columns = ["Tracking ID", "Sort Zone", "Aisle", "Cluster", "Package Length", "Package Width", "Package Height", "DSP Name", "Assigned Cycle", "Last Updated Time"]
    missing = [col for col in required_columns if col not in df.columns] # finds which required columns are missing

    if missing:
        st.error("Missing required columns:") # red error box
        for col in missing:
            st.write(f"   - {col}")
        st.info("Please check your SCC export filters include these fields, then re-upload.")
    else:
        st.success(f"Data loaded - {df.shape[0]} packages ready for analysis.") # green success

        # Parcel Size Grouping ------------------------------------------
        df["Package Length"] = df["Package Length"].str.replace(" cm", "").astype(float) # removes " cm" and converts to number
        df["Package Width"] = df["Package Width"].str.replace(" cm", "").astype(float)
        df["Package Height"] = df["Package Height"].str.replace(" cm", "").astype(float)

        df["Longest Side"] = df[["Package Length", "Package Width", "Package Height"]].max(axis=1) # picks largest dimension per row

        # assigns size label based on longest dimension (Amazon UK size tiers)
        def get_size(longest):
            if longest <= 35:
                return "Small"
            elif longest <= 45:
                return "Medium"
            elif longest <= 61:
                return "Small Oversize"
            else:
                return "Large Oversize"

        df["Size Category"] = df["Longest Side"].apply(get_size) # runs get_size on every row

        # Date range for chart titles -----------------------------------
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"]) # converts text to date/time
        start_date = df["Last Updated Time"].min().strftime("%d %b %Y") # earliest date
        end_date = df["Last Updated Time"].max().strftime("%d %b %Y") # latest date

        # Summary Stats -------------------------------------------------
        st.subheader(f"Quick Summary ({start_date} - {end_date})")

        col1, col2, col3, col4 = st.columns(4) # 4 boxes side by side
        col1.metric("Total Lost", len(df)) # total packages
        col2.metric("Worst Cluster", df["Cluster"].value_counts().index[0]) # cluster with most losts
        col3.metric("Worst Aisle", df["Aisle"].value_counts().index[0]) # aisle with most losts
        top_dsp_name = df["DSP Name"].dropna().value_counts().index[0] # DSP with most losts
        col4.metric("Worst DSP", top_dsp_name[:15]) # first 15 chars to prevent overflow

        # Tabs ----------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Location", "Rankings", "DSP & Cycle", "Time", "Export", "Bridge"])

        # TAB 1: OVERVIEW -----------------------------------------------
        with tab1:
            st.caption("Size breakdown of lost parcels and summary by cluster.")

            overview_display = st.radio("Display as:", ["Table", "Chart"], horizontal=True, key="overview_display") # toggle for whole tab

            if overview_display == "Table":
                st.subheader("Lost Parcel Size Breakdown")
                st.write(df["Size Category"].value_counts()) # count per size

                st.subheader("Lost Parcels by Cluster and Size")
                summary_tbl = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0) # pivot table: clusters vs sizes
                summary_tbl["Total"] = summary_tbl.sum(axis=1) # adds total column
                st.dataframe(summary_tbl)
            else:
                st.subheader("Lost Parcels by Size")
                size_counts = df["Size Category"].value_counts()
                fig_ov, ax_ov = plt.subplots(figsize=(8, 4))
                ax_ov.bar(size_counts.index, size_counts.values, color=["green", "orange", "red", "darkred"])
                ax_ov.set_xlabel("Size Category")
                ax_ov.set_ylabel("Lost Parcels")
                ax_ov.set_title(f"Lost Parcels by Size ({start_date} - {end_date})")
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig_ov)

                st.subheader("Lost Parcels by Cluster")
                cluster_counts = df["Cluster"].value_counts()
                fig_cl, ax_cl = plt.subplots(figsize=(8, 4))
                ax_cl.bar(cluster_counts.index, cluster_counts.values, color="steelblue")
                ax_cl.set_xlabel("Cluster")
                ax_cl.set_ylabel("Lost Parcels")
                ax_cl.set_title(f"Lost Parcels by Cluster ({start_date} - {end_date})")
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig_cl)

        # TAB 2: LOCATION -----------------------------------------------
        with tab2:
            st.caption("Drill into a cluster to see which aisles or zones have the most losts. Filter by package size to spot problem areas.")

            selected_cluster = st.selectbox("Select Cluster:", sorted(df["Cluster"].dropna().unique()), key="cluster_select") # dropdown of clusters
            filtered_df = df[df["Cluster"] == selected_cluster] # filters to selected cluster only
            st.write(f"Showing {len(filtered_df)} lost parcels in Cluster {selected_cluster}")

            view_detail = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="view_select") # pick detail level
            location_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="location_display") # toggle for whole tab

            chart_data = filtered_df[view_detail].value_counts() # counts losts per location in that cluster

            if location_display == "Chart":
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.bar(chart_data.index, chart_data.values)
                ax.set_xlabel(view_detail)
                ax.set_ylabel("Lost Parcels")
                ax.set_title(f"Lost Parcels in Cluster {selected_cluster} by {view_detail}")
                if view_detail == "Sort Zone":
                    plt.xticks(rotation=45, ha="right") # angled for long names
                else:
                    plt.xticks(rotation=0, ha="center") # horizontal for short names
                plt.tight_layout()
                st.pyplot(fig)

                # Size by zone chart
                st.subheader(f"Package Size by Aisle in Cluster {selected_cluster}")
                selected_size = st.selectbox("Select Size Category:", sorted(df["Size Category"].dropna().unique()), key="size_select")
                size_df = filtered_df[filtered_df["Size Category"] == selected_size] # filters to that size + cluster
                st.write(f"{len(size_df)} '{selected_size}' parcels lost in Cluster {selected_cluster}")
                size_zone_data = size_df["Aisle"].value_counts()

                if len(size_zone_data) > 0: # only draw if there's data
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
                st.subheader(f"Lost Parcels by {view_detail}")
                location_table = chart_data.reset_index().rename(columns={view_detail: "Location", "count": "Lost Parcels"})
                location_table.index = range(1, len(location_table) + 1) # ranking starts from 1
                st.dataframe(location_table)

                st.subheader(f"Package Size Breakdown in Cluster {selected_cluster}")
                cluster_size_tbl = filtered_df.groupby(["Aisle", "Size Category"]).size().unstack(fill_value=0) # pivot: aisles vs sizes
                cluster_size_tbl["Total"] = cluster_size_tbl.sum(axis=1)
                st.dataframe(cluster_size_tbl)

        # TAB 3: RANKINGS -----------------------------------------------
        with tab3:
            st.caption("See the worst performing locations ranked by number of lost parcels.")

            rank_view = st.selectbox("Rank by:", ["Sort Zone", "Aisle"], key="rank_select")
            rank_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="rank_display") # toggle for whole tab

            rank_data = df[rank_view].value_counts().head(10) # top 10 worst locations

            if rank_display == "Chart":
                fig8, ax8 = plt.subplots(figsize=(12, 5))
                ax8.barh(rank_data.index, rank_data.values, color="darkred") # horizontal bars for rankings
                ax8.set_xlabel("Lost Parcels")
                ax8.set_ylabel(rank_view)
                ax8.set_title(f"Top 10 {rank_view}s with Most Lost Parcels ({start_date} - {end_date})")
                ax8.invert_yaxis() # worst at top
                plt.tight_layout()
                st.pyplot(fig8)
            else:
                rank_table = rank_data.reset_index().rename(columns={rank_view: "Location", "count": "Lost Parcels"})
                rank_table.index = range(1, len(rank_table) + 1) # ranking starts from 1
                st.dataframe(rank_table)

        # TAB 4: DSP & CYCLE --------------------------------------------
        with tab4:
            st.caption("See which DSPs lose the most parcels and compare performance across dispatch cycles.")

            dsp_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="dsp_display") # toggle for whole tab

            if dsp_display == "Chart":
                st.subheader(f"Lost Parcels by DSP ({start_date} - {end_date})")
                dsp_data = df["DSP Name"].dropna().value_counts() # count losts per DSP
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.bar(dsp_data.index, dsp_data.values, color="orange")
                ax2.set_xlabel("DSP")
                ax2.set_ylabel("Lost Parcels")
                ax2.set_title(f"Lost Parcels by DSP ({start_date} - {end_date})")
                plt.xticks(rotation=45, ha="right") # angled for long DSP names
                plt.tight_layout()
                st.pyplot(fig2)

                st.subheader(f"Lost Parcels by Cycle ({start_date} - {end_date})")
                cycle_data = df["Assigned Cycle"].dropna().value_counts() # count losts per cycle
                fig4, ax4 = plt.subplots(figsize=(10, 5))
                ax4.bar(cycle_data.index, cycle_data.values, color="purple")
                ax4.set_xlabel("Cycle")
                ax4.set_ylabel("Lost Parcels")
                ax4.set_title(f"Lost Parcels by Cycle ({start_date} - {end_date})")
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig4)
            else:
                st.subheader(f"Lost Parcels by DSP ({start_date} - {end_date})")
                dsp_data = df["DSP Name"].dropna().value_counts()
                dsp_table = dsp_data.reset_index().rename(columns={"DSP Name": "DSP", "count": "Lost Parcels"})
                dsp_table.index = range(1, len(dsp_table) + 1) # ranking starts from 1
                st.dataframe(dsp_table)

                st.subheader(f"Lost Parcels by Cycle ({start_date} - {end_date})")
                cycle_data = df["Assigned Cycle"].dropna().value_counts()
                cycle_table = cycle_data.reset_index().rename(columns={"Assigned Cycle": "Cycle", "count": "Lost Parcels"})
                cycle_table.index = range(1, len(cycle_table) + 1) # ranking starts from 1
                st.dataframe(cycle_table)

        # TAB 5: TIME ---------------------------------------------------
        with tab5:
            st.caption("See which days of the week have the most losts. Select a day to view individual tracking IDs.")

            df["Day of Week"] = df["Last Updated Time"].dt.day_name() # extracts day name from date
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_data = df["Day of Week"].value_counts().reindex(day_order, fill_value=0) # counts per day in Mon-Sun order

            time_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="time_display") # toggle for whole tab

            if time_display == "Chart":
                st.subheader("Lost Parcels by Day of Week")
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                ax3.bar(day_data.index, day_data.values, color="green")
                ax3.set_xlabel("Day of Week")
                ax3.set_ylabel("Lost Parcels")
                ax3.set_title(f"Lost Parcels by Day of Week ({start_date} - {end_date})")
                plt.xticks(rotation=0, ha="center")
                plt.tight_layout()
                st.pyplot(fig3)
            else:
                st.subheader("Lost Parcels by Day of Week")
                day_table = day_data.reset_index().rename(columns={"Day of Week": "Day", "count": "Lost Parcels"})
                day_table.index = range(1, len(day_table) + 1) # ranking starts from 1
                st.dataframe(day_table)

            # Tracking ID lookup by day
            st.subheader("Lost Parcel Details by Day")
            selected_day = st.selectbox("Select Day:", day_order, key="day_select") # dropdown to pick a day
            day_df = df[df["Day of Week"] == selected_day] # filters to that day only
            st.write(f"{len(day_df)} parcels lost on {selected_day}")
            st.dataframe(day_df[["Tracking ID", "Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"]]) # key columns for investigation

        # TAB 6: EXPORT -------------------------------------------------
        with tab6:
            st.caption("Download the cleaned data with sensitive information removed and size categories added.")

            st.subheader("Export Data")
            csv = df.to_csv(index=False) # converts table to CSV text for download

            st.download_button(
                label="Download cleaned data as CSV", # button text
                data=csv, # the CSV content
                file_name="Lost_Parcels_Cleaned.csv", # name of downloaded file
                mime="text/csv" # tells browser it's a CSV
            ) 

                # TAB 7: BRIDGE -------------------------------------------------
        with tab7:
            st.caption("Auto-generates a lost parcels bridge draft. Copy the Quick prompt for AI-enhanced action plans.")

            # --- CALCULATE KEY STATS ---
            total_lost = len(df)
            worst_cluster = df["Cluster"].value_counts().index[0]
            worst_cluster_count = df["Cluster"].value_counts().values[0]
            worst_cluster_pct = round(worst_cluster_count / total_lost * 100, 1)

            worst_aisle = df["Aisle"].value_counts().index[0]
            worst_aisle_count = df["Aisle"].value_counts().values[0]

            worst_dsp = df["DSP Name"].dropna().value_counts().index[0]
            worst_dsp_count = df["DSP Name"].dropna().value_counts().values[0]
            avg_dsp_count = df["DSP Name"].dropna().value_counts().mean()
            worst_dsp_multiple = round(worst_dsp_count / avg_dsp_count, 1)

            worst_size = df["Size Category"].value_counts().index[0]
            worst_size_count = df["Size Category"].value_counts().values[0]

            top_cycle = df["Assigned Cycle"].dropna().value_counts().index[0]
            top_cycle_count = df["Assigned Cycle"].dropna().value_counts().values[0]

            # --- GENERATE DRAFT BRIDGE ---
            bridge_text = f"""Lost Parcels Bridge - DRM2
{start_date} - {end_date}

Lost (Total): {total_lost}

RC1) Cluster {worst_cluster}: {worst_cluster_count} parcels ({worst_cluster_pct}% of all losts) — Worst aisle: {worst_aisle} ({worst_aisle_count} parcels)
RC2) DSP {worst_dsp}: {worst_dsp_count} parcels ({worst_dsp_multiple}x station average)
RC3) {worst_size} parcels: {worst_size_count} lost — most common size category

AC1: Additional Problem Solver assigned to Cluster {worst_cluster} during {top_cycle} to investigate and resolve issues in Aisle {worst_aisle} which accounts for the highest concentration of losts.
AC2: DSP {worst_dsp} to be briefed on correct pick-and-stage process — currently {worst_dsp_multiple}x station average for lost parcels this period.
AC3: OS to conduct stow audit in Cluster {worst_cluster} focusing on {worst_size} parcel handling to ensure correct placement and reduce misplacement.
AC4: Daily PS huddle to review previous day's losts by cluster and assign targeted investigations based on volume and repeat locations.
"""

            st.subheader("Draft Bridge")
            st.text_area("Edit as needed:", value=bridge_text, height=350) # editable text box

            # --- ENHANCE WITH QUICK ---
            st.subheader("Enhance with Quick")
            st.write("For better action plans, copy the prompt below and paste into Quick online.")

            quick_prompt = f"""Write me a Lost Parcels bridge for DRM2 in the same style as a PS Effectiveness bridge. Use RC1, RC2, RC3 for root causes and AC1, AC2, AC3, AC4 for action plans. Make actions specific, realistic, and different from last week.

Data ({start_date} - {end_date}):
- Total lost: {total_lost}
- Worst cluster: {worst_cluster} ({worst_cluster_count} parcels, {worst_cluster_pct}%)
- Worst aisle: {worst_aisle} ({worst_aisle_count} parcels)
- Worst DSP: {worst_dsp} ({worst_dsp_count} parcels, {worst_dsp_multiple}x average)
- Most common size lost: {worst_size} ({worst_size_count} parcels)
- Top cycle: {top_cycle} ({top_cycle_count} parcels)

Generate realistic, specific action plans that a station manager would actually implement. Keep the tone professional and concise.
"""

            st.code(quick_prompt, language="text") # has copy icon in top-right corner
            st.info("Click the copy icon (top-right corner above) → Open Quick → Ctrl+V")
