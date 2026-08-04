import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("SCC Lost Parcel Heatmap")

uploaded_file = st.file_uploader("Upload SCC export as a .csv file please to generate the heatmap", type="csv") # says what filetype is uploaded

if uploaded_file is not None: # prevents app from crashing without an uploaded file
    df = pd.read_csv(uploaded_file) # scans and reads csv file

    # Sensitive Data Check --------------------------------------------

    sensitive_columns = ["Last Scan By", "Driver Id", "Holder Name", "City", "Postal", "Province", "Ordering Order ID", "Order Amount", "Receivable Amount", "Payment Method", "District", "Scheduled Delivery End Time"]

    found_sensitive = [col for col in sensitive_columns if col in df.columns] # go through sensitive column titles and check which ones actually exist in uploaded file

    if found_sensitive:
        st.warning(f"Sensitive Information Column Titles Uploaded: {', '.join(found_sensitive)}") # shows yellow warning box with variables possible in text
        st.info("These columns have been automatically removed") # shows box telling user columns been removed
        df = df.drop(columns=found_sensitive) # removes sensitive information

    required_columns = ["Tracking ID", "Sort Zone", "Aisle", "Cluster", "Package Length", "Package Width", "Package Height", "DSP Name", "Assigned Cycle", "Last Updated Time"] # required columns
    missing = [col for col in required_columns if col not in df.columns] # finds columns not_in_file

    if missing:
        st.error("Missing required columns:")
        for col in missing:
            st.write(f"   - {col}")
        st.info("Please check your SCC export filters include these fields, then re-upload.") # user_message failure_and must upload more
    else:
        st.success(f"Data loaded - {df.shape[0]} packages ready for analysis.") # user_message success

        # Parcel Size Grouping ------------------------------------------

        df["Package Length"] = df["Package Length"].str.replace(" cm", "").astype(float) # removes " cm" text and converts to number
        df["Package Width"] = df["Package Width"].str.replace(" cm", "").astype(float) # removes " cm" text and converts to number
        df["Package Height"] = df["Package Height"].str.replace(" cm", "").astype(float) # removes " cm" text and converts to number

        df["Longest Side"] = df[["Package Length", "Package Width", "Package Height"]].max(axis=1) # picks the largest dimension from all 3

        def get_size(longest):
            if longest <= 35:
                return "Small"
            elif longest <= 45:
                return "Medium"
            elif longest <= 61:
                return "Small Oversize"
            else:
                return "Large Oversize"

        df["Size Category"] = df["Longest Side"].apply(get_size) # runs get_size on every row and saves result as new column

        # Date range for titles
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"]) # converts text to date/time
        start_date = df["Last Updated Time"].min().strftime("%d %b %Y") # earliest date formatted nicely
        end_date = df["Last Updated Time"].max().strftime("%d %b %Y") # latest date formatted nicely

        # Summary Stats -------------------------------------------------

        st.subheader(f"Quick Summary ({start_date} - {end_date})")

        col1, col2, col3, col4 = st.columns(4) # creates 4 columns side by side

        col1.metric("Total Lost", len(df)) # total number of lost parcels
        col2.metric("Worst Cluster", df["Cluster"].value_counts().index[0]) # cluster with most losts
        col3.metric("Worst Aisle", df["Aisle"].value_counts().index[0]) # aisle with most losts
        top_dsp_name = df["DSP Name"].dropna().value_counts().index[0] # gets DSP with most losts
        col4.metric("Top DSP", top_dsp_name[:15]) # shows first 15 characters to stop it getting cut off

        # Tabs ----------------------------------------------------------

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Location", "Rankings", "DSP & Cycle", "Time", "Export"]) # creates clickable tabs

        # TAB 1: OVERVIEW -----------------------------------------------
        with tab1:
            st.caption("Size breakdown of lost parcels and summary by cluster.") # tells user what this tab does

            overview_display = st.radio("Display as:", ["Table", "Chart"], horizontal=True, key="overview_display") # one toggle controls whole tab

            if overview_display == "Table":
                st.subheader("Lost Parcel Size Breakdown")
                st.write(df["Size Category"].value_counts()) # counts how many packages are in size category

                st.subheader("Lost Parcels by Cluster and Size")
                summary_tbl = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0) # counts parcels by cluster AND size
                summary_tbl["Total"] = summary_tbl.sum(axis=1) # adds a Total column at the end
                st.dataframe(summary_tbl) # displays the table in the app
            else:
                st.subheader("Lost Parcels by Size")
                size_counts = df["Size Category"].value_counts() # counts per size
                fig_ov, ax_ov = plt.subplots(figsize=(8, 4)) # creates chart canvas
                ax_ov.bar(size_counts.index, size_counts.values, color=["green", "orange", "red", "darkred"]) # bars coloured by severity
                ax_ov.set_xlabel("Size Category") # labels x-axis
                ax_ov.set_ylabel("Lost Parcels") # labels y-axis
                ax_ov.set_title(f"Lost Parcels by Size ({start_date} - {end_date})") # chart title
                plt.xticks(rotation=0, ha="center") # keeps labels horizontal
                plt.tight_layout() # stops labels getting cut off
                st.pyplot(fig_ov) # displays chart

                st.subheader("Lost Parcels by Cluster")
                cluster_counts = df["Cluster"].value_counts() # counts per cluster
                fig_cl, ax_cl = plt.subplots(figsize=(8, 4)) # creates chart canvas
                ax_cl.bar(cluster_counts.index, cluster_counts.values, color="steelblue") # draws bars
                ax_cl.set_xlabel("Cluster") # labels x-axis
                ax_cl.set_ylabel("Lost Parcels") # labels y-axis
                ax_cl.set_title(f"Lost Parcels by Cluster ({start_date} - {end_date})") # chart title
                plt.xticks(rotation=0, ha="center") # keeps labels horizontal
                plt.tight_layout() # stops labels getting cut off
                st.pyplot(fig_cl) # displays chart

        # TAB 2: LOCATION -----------------------------------------------
        with tab2:
            st.caption("Drill into a cluster to see which aisles or zones have the most losts. Filter by package size to spot problem areas.") # tells user what this tab does

            selected_cluster = st.selectbox("Select Cluster:", sorted(df["Cluster"].dropna().unique()), key="cluster_select") # dropdown of every unique cluster

            filtered_df = df[df["Cluster"] == selected_cluster] # filters table to only rows matching selected cluster

            st.write(f"Showing {len(filtered_df)} lost parcels in Cluster {selected_cluster}") # tells user how many packages in that cluster

            view_detail = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="view_select") # dropdown to choose aisle or sort zone view

            location_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="location_display") # one toggle controls whole tab

            chart_data = filtered_df[view_detail].value_counts() # counts lost parcels per aisle/zone in that cluster

            if location_display == "Chart":
                # Location breakdown chart
                fig, ax = plt.subplots(figsize=(14, 5)) # creates chart canvas
                ax.bar(chart_data.index, chart_data.values) # draws bars
                ax.set_xlabel(view_detail) # labels x-axis
                ax.set_ylabel("Lost Parcels") # labels y-axis
                ax.set_title(f"Lost Parcels in Cluster {selected_cluster} by {view_detail}") # chart title
                if view_detail == "Sort Zone":
                    plt.xticks(rotation=45, ha="right") # rotate labels for long names
                else:
                    plt.xticks(rotation=0, ha="center") # horizontal for short names
                plt.tight_layout() # stops labels getting cut off
                st.pyplot(fig) # displays chart

                # Size by zone chart
                st.subheader(f"Package Size by Aisle in Cluster {selected_cluster}")
                selected_size = st.selectbox("Select Size Category:", sorted(df["Size Category"].dropna().unique()), key="size_select") # dropdown to pick a size
                size_df = filtered_df[filtered_df["Size Category"] == selected_size] # filters to that size in selected cluster
                st.write(f"{len(size_df)} '{selected_size}' parcels lost in Cluster {selected_cluster}")
                size_zone_data = size_df["Aisle"].value_counts() # counts which aisles lose that size most

                if len(size_zone_data) > 0: # only draw if there's data
                    fig6, ax6 = plt.subplots(figsize=(12, 5)) # creates chart canvas
                    ax6.bar(size_zone_data.index, size_zone_data.values, color="red") # draws bars in red
                    ax6.set_xlabel("Aisle") # labels x-axis
                    ax6.set_ylabel("Lost Parcels") # labels y-axis
                    ax6.set_title(f"'{selected_size}' Parcels Lost by Aisle in Cluster {selected_cluster}") # chart title
                    plt.xticks(rotation=0, ha="center") # horizontal labels
                    plt.tight_layout() # stops labels getting cut off
                    st.pyplot(fig6) # displays chart
                else:
                    st.info(f"No '{selected_size}' parcels lost in Cluster {selected_cluster}") # no data message
            else:
                # Location breakdown table
                st.subheader(f"Lost Parcels by {view_detail}")
                st.dataframe(chart_data.reset_index().rename(columns={view_detail: "Location", "count": "Lost Parcels"})) # shows as table

                # Size by zone table
                st.subheader(f"Package Size Breakdown in Cluster {selected_cluster}")
                cluster_size_tbl = filtered_df.groupby(["Aisle", "Size Category"]).size().unstack(fill_value=0) # aisles vs sizes table
                cluster_size_tbl["Total"] = cluster_size_tbl.sum(axis=1) # adds total column
                st.dataframe(cluster_size_tbl) # displays table

        # TAB 3: RANKINGS -----------------------------------------------
        with tab3:
            st.caption("See the worst performing locations ranked by number of lost parcels.") # tells user what this tab does

            rank_view = st.selectbox("Rank by:", ["Sort Zone", "Aisle"], key="rank_select") # dropdown to choose ranking type

            rank_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="rank_display") # one toggle controls whole tab

            rank_data = df[rank_view].value_counts().head(10) # top 10 of whichever they picked

            if rank_display == "Chart":
                fig8, ax8 = plt.subplots(figsize=(12, 5)) # creates chart canvas
                ax8.barh(rank_data.index, rank_data.values, color="darkred") # horizontal bar chart
                ax8.set_xlabel("Lost Parcels") # labels x-axis
                ax8.set_ylabel(rank_view) # labels y-axis
                ax8.set_title(f"Top 10 {rank_view}s with Most Lost Parcels ({start_date} - {end_date})") # chart title
                ax8.invert_yaxis() # puts worst at the top
                plt.tight_layout() # stops labels getting cut off
                st.pyplot(fig8) # displays chart
            else:
                st.dataframe(rank_data.reset_index().rename(columns={rank_view: "Location", "count": "Lost Parcels"})) # shows as table

        # TAB 4: DSP & CYCLE --------------------------------------------
        with tab4:
            st.caption("See which DSPs lose the most parcels and compare performance across dispatch cycles.") # tells user what this tab does

            dsp_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="dsp_display") # one toggle controls whole tab

            if dsp_display == "Chart":
                # DSP chart
                st.subheader(f"Lost Parcels by DSP ({start_date} - {end_date})")
                dsp_data = df["DSP Name"].dropna().value_counts() # counts losts per DSP
                fig2, ax2 = plt.subplots(figsize=(12, 5)) # creates chart canvas
                ax2.bar(dsp_data.index, dsp_data.values, color="orange") # draws bars in orange
                ax2.set_xlabel("DSP") # labels x-axis
                ax2.set_ylabel("Lost Parcels") # labels y-axis
                ax2.set_title(f"Lost Parcels by DSP ({start_date} - {end_date})") # chart title
                plt.xticks(rotation=45, ha="right") # rotates DSP names
                plt.tight_layout() # stops labels getting cut off
                st.pyplot(fig2) # displays chart

                # Cycle chart
                st.subheader(f"Lost Parcels by Cycle ({start_date} - {end_date})")
                cycle_data = df["Assigned Cycle"].dropna().value_counts() # counts losts per cycle
                fig4, ax4 = plt.subplots(figsize=(10, 5)) # creates chart canvas
                ax4.bar(cycle_data.index, cycle_data.values, color="purple") # draws bars in purple
                ax4.set_xlabel("Cycle") # labels x-axis
                ax4.set_ylabel("Lost Parcels") # labels y-axis
                ax4.set_title(f"Lost Parcels by Cycle ({start_date} - {end_date})") # chart title
                plt.xticks(rotation=0, ha="center") # horizontal labels
                plt.tight_layout() # stops labels getting cut off
                st.pyplot(fig4) # displays chart
            else:
                # DSP table
                st.subheader(f"Lost Parcels by DSP ({start_date} - {end_date})")
                dsp_data = df["DSP Name"].dropna().value_counts() # counts losts per DSP
                st.dataframe(dsp_data.reset_index().rename(columns={"DSP Name": "DSP", "count": "Lost Parcels"})) # shows as table

                # Cycle table
                st.subheader(f"Lost Parcels by Cycle ({start_date} - {end_date})")
                cycle_data = df["Assigned Cycle"].dropna().value_counts() # counts losts per cycle
                st.dataframe(cycle_data.reset_index().rename(columns={"Assigned Cycle": "Cycle", "count": "Lost Parcels"})) # shows as table

        # TAB 5: TIME ---------------------------------------------------
        with tab5:
            st.caption("See which days of the week have the most losts. Select a day to view individual tracking IDs.") # tells user what this tab does

            df["Day of Week"] = df["Last Updated Time"].dt.day_name() # extracts day name
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_data = df["Day of Week"].value_counts().reindex(day_order, fill_value=0) # counts losts per day in correct order

            time_display = st.radio("Display as:", ["Chart", "Table"], horizontal=True, key="time_display") # one toggle controls whole tab

            if time_display == "Chart":
                st.subheader("Lost Parcels by Day of Week")
                fig3, ax3 = plt.subplots(figsize=(10, 5)) # creates chart canvas
                ax3.bar(day_data.index, day_data.values, color="green") # draws bars in green
                ax3.set_xlabel("Day of Week") # labels x-axis
                ax3.set_ylabel("Lost Parcels") # labels y-axis
                ax3.set_title(f"Lost Parcels by Day of Week ({start_date} - {end_date})") # chart title
                plt.xticks(rotation=0, ha="center") # horizontal labels
                plt.tight_layout() # stops labels getting cut off
                st.pyplot(fig3) # displays chart
            else:
                st.subheader("Lost Parcels by Day of Week")
                st.dataframe(day_data.reset_index().rename(columns={"Day of Week": "Day", "count": "Lost Parcels"})) # shows as table

            # Day breakdown with tracking IDs (always shown as table — this is the detail view)
            st.subheader("Lost Parcel Details by Day")
            selected_day = st.selectbox("Select Day:", day_order, key="day_select") # dropdown to pick a day
            day_df = df[df["Day of Week"] == selected_day] # filters to only that day
            st.write(f"{len(day_df)} parcels lost on {selected_day}")
            st.dataframe(day_df[["Tracking ID", "Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"]]) # shows tracking ID table

        # TAB 6: EXPORT -------------------------------------------------
        with tab6:
            st.caption("Download the cleaned data with sensitive information removed and size categories added.") # tells user what this tab does

            st.subheader("Export Data")

            csv = df.to_csv(index=False) # converts the cleaned dataframe to CSV text

            st.download_button(
                label="Download cleaned data as CSV",
                data=csv,
                file_name="Lost_Parcels_Cleaned.csv",
                mime="text/csv"
            ) # creates a download button
