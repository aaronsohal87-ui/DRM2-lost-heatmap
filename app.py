import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("DRM2 Lost Parcel Heatmap")

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
        st.info("Check the data table output below")
        st.dataframe(df.head()) #shows first 5 rows of data for user verification

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
        st.write(" Lost Parcel Size Breakdown")
        st.write(df["Size Category"].value_counts()) # counts how many packages are in size category and displays it on screen

        # Summary Table 

        st.subheader("Lost Parcel Summary") 

        summary_tbl = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0) # counts parcels by cluster AND size, makes a table
        
        summary_tbl["Total"] = summary_tbl.sum(axis=1) # adds a Total column at the end #
        
        st.dataframe(summary_tbl) # displays the table in the app
        
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


        # Day of Week by Shift Analysis ---------------------------------

        st.subheader("Lost Parcels by Day and Shift")

        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"]) # converts dispatch time text to date/time format

        # function to assign shift based on hour
        def get_shift(hour):
            if hour < 7:
                return "NS"
            elif hour < 15:
                return "AM"
            else:
                return "PM"

        df["Shift"] = df["Dispatch Time"].dt.hour.apply(get_shift) # extracts hour and assigns shift label
        df["Day of Week"] = df["Dispatch Time"].dt.day_name() # extracts day name

        # put days in correct order
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # count losts per day per shift
        shift_data = df.groupby(["Day of Week", "Shift"]).size().unstack(fill_value=0) # creates table: rows = days, columns = shifts
        shift_data = shift_data.reindex(day_order, fill_value=0) # puts days in correct order

        # stacked bar chart
        fig3, ax3 = plt.subplots(figsize=(10, 5)) # creates chart canvas
        shift_data.plot(kind="bar", stacked=True, ax=ax3, color={"NS": "navy", "AM": "orange", "PM": "green"}) # stacked bars coloured by shift
        ax3.set_xlabel("Day of Week") # labels x-axis
        ax3.set_ylabel("Lost Parcels") # labels y-axis
        ax3.set_title("Lost Parcels by Day and Shift") # chart title
        ax3.legend(title="Shift") # shows colour legend
        plt.xticks(rotation=0, ha="center") # keeps day names horizontal
        plt.tight_layout() # stops labels getting cut off
        st.pyplot(fig3) # displays chart in app

        





