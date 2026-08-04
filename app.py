import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("SCC Lost Parcel Heatmap")

uploaded_file = st.file_uploader("Upload SCC export as a .csv file please to generate the heatmap", type="csv") # says what filetype is uploaded

if uploaded_file is not None: # prevents app from crashing without an uploaded file
    df = pd.read_csv(uploaded_file) # scans and reads csv file
    st.write("Data Size:", df.shape) # outputs size of table user inputs

    # Sensitive Data Check --------------------------------------------
    
    sensitive_columns = ["Last Scan By", "Driver Id", "Holder Name", "City", "Postal", "Province", "Ordering Order ID", "Order Amount", "Receivable Amount", "Payment Method", "District", "Scheduled Delivery End Time"]

    found_sensitive = [col for col in sensitive_columns if col in df.columns] # go through sensitive column titles and check which ones actually exist in uploaded file

    if found_sensitive:
        st.warning(f"Sensitive Information Column Titles Uploaded: {', '.join(found_sensitive)}") # shows yellow warning box with variables possible in text
        st.info("These columns have been automatically removed") # shows box telling user columns been removed
        df = df.drop(columns=found_sensitive) # removes sensitive information  
    
    required_columns = ["Tracking ID", "Sort Zone", "Aisle", "Cluster", "Package Length", "Package Width", "Package Height", "DSP Name", "Assigned Cycle", "Last Updated Time"] #required columns
    missing = [col for col in required_columns if col not in df.columns] # finds columns not_in_file 
    
    if missing:
        st.error("Missing required columns:")
        for col in missing:
            st.write(f"   - {col}")
        st.info("Please check your SCC export filters include these fields, then re-upload.") #user_message failure_and must upload more
    else:
        st.success(f"Data loaded - {df.shape[0]} packages ready for analysis.") #user_message success
        #st.info("Check the data table output below")
        #st.dataframe(df.head()) #shows first 5 rows of data for user verification 

        # Summary Stats -------------------------------------------------

        st.subheader("Quick Summary")

        col1, col2, col3, col4 = st.columns(4) # creates 4 columns side by side

        col1.metric("Total Lost", len(df)) # total number of lost parcels
        col2.metric("Worst Cluster", df["Cluster"].value_counts().index[0]) # cluster with most losts
        col3.metric("Worst Aisle", df["Aisle"].value_counts().index[0]) # aisle with most losts
        top_dsp_name = df["DSP Name"].dropna().value_counts().index[0] # gets DSP with most losts
        col4.metric("Top DSP", top_dsp_name[:15]) # shows first 15 characters to stop it getting cut off

        # Parcel Size Grouping ------------------------------------------
        
        df["Package Length"] = df["Package Length"].str.replace(" cm", "").astype(float) #Takes value in csv from  Length measurement into a number by using float
        df["Package Width"] = df["Package Width"].str.replace(" cm", "").astype(float) #Takes value in csv from Width measurement into number using float
        df["Package Height"] = df["Package Height"].str.replace(" cm","").astype(float) #Takes value in csv from Height measurement into number using float

        df["Longest Side"] = df[["Package Length", "Package Width", "Package Height"]].max(axis=1) # .max(axis=1) looks at all 3 values and picks the largest value

        def get_size(longest):
            if longest <= 35:
                return "Small"
            elif longest <= 45:
                return "Medium"
            elif longest <= 61:
                return "Small Oversize"
            else:
                return "Large Oversize"
             
        
        df["Size Category"] = df["Longest Side"].apply(get_size) #records longest side of package and saves it in size category with get_size function applied to it
        st.subheader(" Lost Parcel Size Breakdown")
        st.write(df["Size Category"].value_counts()) # counts how many packages are in size category and displays it on screen

        
        
      # Interactive Heatmap --------------------------------------------

        st.subheader("Lost Parcels by Location") # displays section heading

        selected_cluster = st.selectbox("Select Cluster:", sorted(df["Cluster"].dropna().unique()), key="cluster_select") # dropdown of every unique cluster, key gives it a unique ID so Streamlit doesn't mix it up with other dropdowns

        filtered_df = df[df["Cluster"] == selected_cluster] # filters table to only rows matching selected cluster

        st.write(f"Showing {len(filtered_df)} lost parcels in Cluster {selected_cluster}") # tells user how many packages in that cluster

        view_detail = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="view_select") # dropdown to choose aisle or sort zone view, key gives it unique ID

        chart_data = filtered_df[view_detail].value_counts() # counts lost parcels per aisle/zone in that cluster

        # location chart
        fig, ax = plt.subplots(figsize=(14, 5)) # creates chart canvas, 14 wide 5 tall
        ax.bar(chart_data.index, chart_data.values) # draws bars, index = labels, values = heights
        ax.set_xlabel(view_detail) # labels x-axis with whatever user picked
        ax.set_ylabel("Lost Parcels") # labels y-axis
        ax.set_title(f"Lost Parcels in Cluster {selected_cluster} by {view_detail}") # chart title changes based on user selection
        if view_detail == "Sort Zone": # if user picked sort zone
            plt.xticks(rotation=45, ha="right") # rotate labels 45 degrees because sort zone names are long
        else: # if user picked aisle
            plt.xticks(rotation=0, ha="center") # keep labels horizontal because aisle names are short
        plt.tight_layout() # stops labels from getting cut off at edges
        st.pyplot(fig) # displays the chart in the app

        # DSP Breakdown -------------------------------------------------

        st.subheader(f"Lost Parcels by DSP in Cluster {selected_cluster}") # section heading that updates with selected cluster

        dsp_data = filtered_df["DSP Name"].dropna().value_counts() # counts losts per DSP in selected cluster, dropna removes empty values

        fig2, ax2 = plt.subplots(figsize=(12, 5)) # creates second chart canvas
        ax2.bar(dsp_data.index, dsp_data.values, color="orange") # draws bars in orange to visually separate from first chart
        ax2.set_xlabel("DSP") # labels x-axis
        ax2.set_ylabel("Lost Parcels") # labels y-axis
        ax2.set_title(f"Lost Parcels by DSP in Cluster {selected_cluster}") # chart title with selected cluster
        plt.xticks(rotation=45, ha="right") # rotates DSP names 45 degrees because they are long
        plt.tight_layout() # stops labels getting cut off
        st.pyplot(fig2) # displays second chart in the ap 


        # Day of Week Analysis ------------------------------------------

        st.subheader("Lost Parcels by Day of Week")

        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"]) # converts text to date/time
        df["Day of Week"] = df["Last Updated Time"].dt.day_name() # extracts day name

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_data = df["Day of Week"].value_counts().reindex(day_order, fill_value=0) # counts losts per day in correct order

        fig3, ax3 = plt.subplots(figsize=(10, 5)) # creates chart canvas
        ax3.bar(day_data.index, day_data.values, color="green") # draws bars in green
        ax3.set_xlabel("Day of Week") # labels x-axis
        ax3.set_ylabel("Lost Parcels") # labels y-axis
        ax3.set_title("Lost Parcels by Day of Week") # chart title
        plt.xticks(rotation=0, ha="center") # keeps day names horizontal
        plt.tight_layout() # stops labels getting cut off
        st.pyplot(fig3) # displays chart in app 


        # Day breakdown table with tracking IDs -------------------------

        st.subheader("Lost Parcel Details by Day")

        selected_day = st.selectbox("Select Day:", day_order, key="day_select") # dropdown to pick a day

        day_df = df[df["Day of Week"] == selected_day] # filters to only that day

        st.write(f"{len(day_df)} parcels lost on {selected_day}")
        st.dataframe(day_df[["Tracking ID", "Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"]]) # shows table with key info 

        # Cycle Comparison ----------------------------------------------

        st.subheader("Lost Parcels by Cycle")

        # get date range from the data
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"])
        start_date = df["Last Updated Time"].min().strftime("%d %b %Y") # earliest date formatted nicely
        end_date = df["Last Updated Time"].max().strftime("%d %b %Y") # latest date formatted nicely

        cycle_data = df["Assigned Cycle"].dropna().value_counts()

        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.bar(cycle_data.index, cycle_data.values, color="purple")
        ax4.set_xlabel("Cycle")
        ax4.set_ylabel("Lost Parcels")
        ax4.set_title(f"Lost Parcels by Cycle ({start_date} - {end_date})") # title now shows date range
        plt.xticks(rotation=0, ha="center")
        plt.tight_layout()
        st.pyplot(fig4) 

        # Summary Table 

        st.subheader("Lost Parcel Summary") 

        summary_tbl = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0) # counts parcels by cluster AND size, makes a table
        
        summary_tbl["Total"] = summary_tbl.sum(axis=1) # adds a Total column at the end #
        
        st.dataframe(summary_tbl) # displays the table in the app


        # Size by Zone --------------------------------------------------

        st.subheader(f"Package Size by Aisle ({start_date} - {end_date})")

        selected_size = st.selectbox("Select Size Category:", sorted(df["Size Category"].dropna().unique()), key="size_select") # dropdown to pick a size

        size_df = df[df["Size Category"] == selected_size] # filters to only that size

        st.write(f"{len(size_df)} '{selected_size}' parcels lost")

        size_zone_data = size_df["Aisle"].value_counts() # counts which aisles lose that size most

        fig6, ax6 = plt.subplots(figsize=(12, 5)) # creates chart canvas
        ax6.bar(size_zone_data.index, size_zone_data.values, color="red") # draws bars in red
        ax6.set_xlabel("Aisle") # labels x-axis
        ax6.set_ylabel("Lost Parcels") # labels y-axis
        ax6.set_title(f"'{selected_size}' Parcels Lost by Aisle") # chart title
        plt.xticks(rotation=0, ha="right") # rotates aisle names
        plt.tight_layout() # stops labels getting cut off
        st.pyplot(fig6) # displays chart in app 



        # Location Ranking ----------------------------------------------

        st.subheader(f"Top 10 Location Ranking ({start_date} - {end_date})")

        rank_view = st.selectbox("Rank by:", ["Sort Zone", "Aisle"], key="rank_select") # dropdown to choose which ranking to show

        rank_data = df[rank_view].value_counts().head(10) # top 10 of whichever they picked

        fig8, ax8 = plt.subplots(figsize=(12, 5)) # creates chart canvas
        ax8.barh(rank_data.index, rank_data.values, color="darkred") # horizontal bar chart
        ax8.set_xlabel("Lost Parcels") # labels x-axis
        ax8.set_ylabel(rank_view) # labels y-axis with whatever user picked
        ax8.set_title(f"Top 10 {rank_view}s with Most Lost Parcels ({start_date} - {end_date})") # chart title changes based on selection
        ax8.invert_yaxis() # puts worst at the top
        plt.tight_layout() # stops labels getting cut off
        st.pyplot(fig8) # displays chart in app




