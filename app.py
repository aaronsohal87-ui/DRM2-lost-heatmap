import streamlit as st
import pandas as pd

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
 
                return "Small"
            elif longest <= 45:
                return "Medium"
            elif longest <= 61: 
                return "Small Oversize"
            elif longest <= 75: 
                return "Large Oversize" 
        
        df["Size Category"] = df["Longest Side"].apply(get_size) #records longest side of package and saves it in size category with get_size function applied to it
        st.write(" Lost Parcel Size Breakdown")
        st.write(df["Size Category"].value_counts()) # counts how many packages are in size category and displays it on screen 
        
        # Interactive Heatmap --------------------------------------------

        st.subheader("Lost Parcels by Location") # displays small heading

        selected_cluster = st.selectbox("Select Cluster:" , sorted(df["Cluster"].unique())) # creates a dropdown menu of every unique cluster 

        filtered_df = df[df["Cluster"] == selected_cluster] # Filters table to only rows where cluster matches what user picks 

        st.write(f"Showing {len(filtered_df)} lost parcels in Cluster {selected_cluster}") # tells user how many packages are in the cluster
        
        view_detail = st.selectbox("View by:" ,["Aisle", "Sort Zone"] # creates selection boxes which allows user to choose what they want to view data by 

        chart_data = filtered_df[view_detail].value_counts() # counts how many parcels left in the cluster

        st.bar_chart(chart_data) # outputs bar chart of the data
        





