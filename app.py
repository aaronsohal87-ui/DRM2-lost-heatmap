import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from io import BytesIO

st.set_page_config(page_title="DRM2 Lost Heatmap", page_icon="📦", layout="wide")
st.title("📦 DRM2 Lost Parcel Heatmap")
st.markdown("---")

STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]
SHIFT_ORDER = ["NS", "AM", "PM", "OTR"]
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen", "OTR": "firebrick"}
SHIFT_DEFINITIONS = {"NS": "23:45 – 09:45 (Night Sort — stow)", "AM": "09:45 – 14:00 (Pick, stage, dispatch)", "PM": "14:00 – 23:45 (Dispatch, RELO)", "OTR": "On The Road (DSP responsibility)"}
SHIFT_HOUR_MAP = {0:"NS",1:"NS",2:"NS",3:"NS",4:"NS",5:"NS",6:"NS",7:"NS",8:"NS",9:"NS",10:"AM",11:"AM",12:"AM",13:"AM",14:"PM",15:"PM",16:"PM",17:"PM",18:"PM",19:"PM",20:"PM",21:"PM",22:"PM",23:"NS"}
SUB_BUCKET_SHIFT_MAP = {"Lost At Station - Inducted Not Stowed":"NS","Lost At Station - Stowed Not Picked Up":"AM","Lost At Station - Debrief Receive(RTS)":"PM","Lost On Road - Attempted":"OTR","Lost On Road - Damage":"OTR","Lost On Road - No Further Status":"OTR"}
SENSITIVE_COLS = ["Holder Name","Ordering Order ID","Order Amount","Receivable Amount","Payment Method","District","Scheduled Delivery End Time"]
REQUIRED_SCC_COLS = ["Tracking ID","Sort Zone","Aisle","Cluster","Package Length","Package Width","Package Height","DSP Name","Assigned Cycle","Last Updated Time"]
REQUIRED_PM_COLS = ["tracking_id","sub_bucket"]
CHART = (7, 2.5)
DSP_MAX = 20
LABEL_MAX = 25

SIZE_TIER_INFO = """
| Size Tier | Longest Side | Amazon Classification |
|-----------|-------------|----------------------|
| Small Envelope | ≤ 20 cm | Thin, flat items |
| Standard Envelope | ≤ 33 cm | Books, small items |
| Standard Parcel | ≤ 45 cm | Most consumer goods |
| Small Oversize | ≤ 61 cm | Larger boxed items |
| Standard Oversize | ≤ 120 cm | Furniture, large appliances |
| Large Oversize | > 120 cm | Very large items |
"""

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────
def get_size(val):
    if pd.isna(val): return "Unknown"
    if val <= 20: return "Small Envelope"
    if val <= 33: return "Standard Envelope"
    if val <= 45: return "Standard Parcel"
    if val <= 61: return "Small Oversize"
    if val <= 120: return "Standard Oversize"
    return "Large Oversize"

def hour_to_shift(hour):
    if pd.isna(hour): return "Unknown"
    return SHIFT_HOUR_MAP.get(int(hour), "Unknown")

def assign_shift(row):
    sb = row.get("Sub Bucket", "")
    if sb in SUB_BUCKET_SHIFT_MAP: return SUB_BUCKET_SHIFT_MAP[sb]
    prev_dt = row.get("Prev Event DT")
    if pd.notna(prev_dt): return hour_to_shift(prev_dt.hour)
    disp_dt = row.get("Dispatch Time")
    if pd.notna(disp_dt): return hour_to_shift(disp_dt.hour)
    cyc = row.get("Assigned Cycle", "")
    if pd.notna(cyc):
        u = str(cyc).upper()
        if "NS" in u or "NIGHT" in u: return "NS"
        if "PM" in u or "RELO" in u or "C2" in u: return "PM"
        if "AM" in u or "C1" in u: return "AM"
    return "Unknown"

def classify_otr_utr(sub_bucket):
    if pd.isna(sub_bucket): return "Unknown"
    if "Lost On Road" in str(sub_bucket): return "OTR"
    if "Lost At Station" in str(sub_bucket): return "UTR"
    return "Unknown"

def is_flex_driver(dsp_name):
    if pd.isna(dsp_name): return False
    return "CSP_COMPANY_NAME" in str(dsp_name).upper()

def clean_dsp_name(dsp_name):
    if pd.isna(dsp_name): return dsp_name
    if is_flex_driver(dsp_name): return "FLEX DRIVER"
    return str(dsp_name).strip()

def clean_scc(df):
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])
    df = df.drop(columns=[c for c in ["Last Scan By","Driver Id"] if c in df.columns])
    for col in ["Package Length","Package Width","Package Height"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"\s*cm", "", regex=True), errors="coerce")
    dims = ["Package Length","Package Width","Package Height"]
    df["Longest Side"] = df[dims].max(axis=1) if all(c in df.columns for c in dims) else float("nan")
    df["Size Category"] = df["Longest Side"].apply(get_size)
    if "Last Updated Time" in df.columns: df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], dayfirst=True, errors="coerce")
    if "Dispatch Time" in df.columns: df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], dayfirst=True, errors="coerce")
    if "Province" in df.columns: df["Province"] = df["Province"].astype(str).str.strip().str.title().replace("Nan", pd.NA)
    if "City" in df.columns: df["City"] = df["City"].astype(str).str.strip().str.title().replace("Nan", pd.NA)
    if "DSP Name" in df.columns: df["DSP Name"] = df["DSP Name"].apply(clean_dsp_name)
    # Keep "Stowed By" if present (for PNOV associate tracking)
    if "Stowed By" not in df.columns:
        # Try to find it under alternate names
        for alt in ["stowed_by", "Stowed_By", "StowedBy", "Last Scan By"]:
            if alt in df.columns:
                df["Stowed By"] = df[alt]
                break
    return df

def merge_data(pm_df, scc_df):
    scc_clean = clean_scc(scc_df.copy())
    pm_keep = ["tracking_id","bucket","sub_bucket","previous_event_datetime","previous_reason","previous_reason_3","event_datetime","shipment_value"]
    pm_cols = pm_df[[c for c in pm_keep if c in pm_df.columns]].copy()
    pm_cols = pm_cols.rename(columns={"tracking_id":"Tracking ID"})
    pm_cols["Prev Event DT"] = pd.to_datetime(pm_cols.get("previous_event_datetime", pd.Series(dtype="object")), format="%d/%m/%Y %H:%M", errors="coerce")
    if "event_datetime" in pm_cols.columns: pm_cols["Marked Lost DT"] = pd.to_datetime(pm_cols["event_datetime"], dayfirst=True, errors="coerce")
    if "shipment_value" in pm_cols.columns: pm_cols["Cost (£)"] = pd.to_numeric(pm_cols["shipment_value"].astype(str), errors="coerce")
    merged = pm_cols.merge(scc_clean, on="Tracking ID", how="left")
    merged["Sub Bucket"] = merged["sub_bucket"]; merged["Bucket"] = merged.get("bucket")
    merged["Type"] = merged["Sub Bucket"].apply(classify_otr_utr)
    if "previous_reason" in merged.columns: merged["Loss Reason"] = merged["previous_reason"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else: merged["Loss Reason"] = "Unknown"
    if "previous_reason_3" in merged.columns: merged["UTR Reason"] = merged["previous_reason_3"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else: merged["UTR Reason"] = "Unknown"
    merged["Shift"] = merged.apply(assign_shift, axis=1)
    merged["Driver Type"] = merged["DSP Name"].apply(lambda x: "Flex" if x == "FLEX DRIVER" else ("DSP" if pd.notna(x) else "Unknown"))
    for col in ["Cluster","Aisle","Sort Zone","DSP Name","Size Category","City","Province","Postal","Cost (£)"]:
        if col not in merged.columns: merged[col] = None
    if "Stowed By" not in merged.columns: merged["Stowed By"] = None
    return merged

def get_date_range(df):
    for col in ["Marked Lost DT","Dispatch Time","Last Updated Time"]:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid)>0:
                s,e = valid.min().strftime("%d %b %Y"), valid.max().strftime("%d %b %Y")
                return s if s==e else f"{s} – {e}"
    return ""
def safe_top(s): c = s.dropna().value_counts(); return c.index[0] if len(c)>0 else "N/A"
def trunc(labels, mx=LABEL_MAX): return [str(l)[:mx]+"..." if len(str(l))>mx else str(l) for l in labels]
def fmt_cost(val):
    if pd.isna(val): return "£0.00"
    return f"£{val:,.2f}"
def make_table(series, c1, c2): t = series.reset_index(); t.columns = [c1, c2]; t.index = range(1,len(t)+1); return t
def make_cost_table(df, group_col):
    grouped = df.groupby(group_col).agg(Lost=("Tracking ID","count"), Total_Cost=("Cost (£)","sum")).sort_values("Lost", ascending=False)
    grouped["Total_Cost"] = grouped["Total_Cost"].apply(fmt_cost); grouped = grouped.rename(columns={"Total_Cost":"Cost Lost"}).reset_index()
    grouped.index = range(1,len(grouped)+1); return grouped
def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    h = max(2,len(data)*0.3); fig,ax = plt.subplots(figsize=(figsize_width,h))
    labs = trunc(data.index,max_label); ax.barh(labs,data.values,color=color); ax.invert_yaxis()
    for i,v in enumerate(data.values): ax.text(v+0.2,i,str(int(v)),va="center",fontsize=7)
    ax.set_xlabel("Lost Parcels",fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); return fig
def make_bar_shift(data, title):
    data = data.reindex(SHIFT_ORDER,fill_value=0); fig,ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER,[data[s] for s in SHIFT_ORDER],color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.2,str(int(b.get_height())),ha="center",fontsize=7)
    ax.set_xlabel("Shift",fontsize=8); ax.set_ylabel("Lost",fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); return fig
def make_pie_otr_utr(df, total, title):
    otr_n = len(df[df["Type"]=="OTR"]); utr_n = len(df[df["Type"]=="UTR"])
    labels,sizes,colors,explode = [],[],[],[]
    if utr_n>0: labels.append(f"UTR ({utr_n})"); sizes.append(utr_n); colors.append("darkorange"); explode.append(0)
    if otr_n>0: labels.append(f"OTR ({otr_n})"); sizes.append(otr_n); colors.append("firebrick"); explode.append(0.05)
    if not sizes: return plt.subplots(figsize=(2,1.5))[0]
    fig,ax = plt.subplots(figsize=(2,1.5))
    ax.pie(sizes,labels=labels,colors=colors,explode=explode,autopct="%1.0f%%",startangle=90,textprops={"fontsize":5})
    ax.set_title(title,fontsize=6); plt.tight_layout(); return fig

# ─── HEALTH SCORE ─────────────────────────────────────────────────────────────
def render_health_score(df, total):
    score = 10; reasons = []
    cl_c = df["Cluster"].dropna().value_counts()
    if len(cl_c) >= 2:
        vals = cl_c.values.astype(float); n = len(vals); sv = np.sort(vals)
        gini = (2 * np.sum(np.arange(1, n+1) * sv) - (n+1) * np.sum(sv)) / (n * np.sum(sv))
        if gini > 0.6: score -= 3; reasons.append("Losses very concentrated")
        elif gini > 0.4: score -= 2; reasons.append("Losses somewhat concentrated")
        elif gini > 0.3: score -= 1; reasons.append("Slight concentration")
    shift_counts = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
    assigned = shift_counts.sum()
    if assigned >= 20:
        observed = np.array([shift_counts[s] for s in SHIFT_ORDER])
        expected = np.array([assigned/4]*4)
        _, p = sp_stats.chisquare(observed, f_exp=expected)
        if p < 0.01: score -= 2; reasons.append("Shift imbalance severe")
        elif p < 0.05: score -= 1; reasons.append("Shift imbalance present")
    otr = df[df["Type"]=="OTR"]
    if len(otr) >= 5:
        dc = otr["DSP Name"].dropna().value_counts()
        if len(dc) >= 3:
            mu = dc.mean(); sigma = dc.std()
            if sigma > 0:
                outliers_n = ((dc - mu) / sigma > 1.5).sum()
                if outliers_n >= 2: score -= 2; reasons.append(f"{outliers_n} DSP outliers")
                elif outliers_n == 1: score -= 1; reasons.append("1 DSP outlier")
    sc3 = df["Size Category"].value_counts()
    if len(sc3) >= 2:
        ov = sum(sc3.get(k, 0) for k in ["Small Oversize", "Standard Oversize", "Large Oversize"])
        if ov / total > 0.3: score -= 1; reasons.append("Oversize elevated")
    score = max(1, min(10, score))
    if score >= 8: color = "🟢"; label = "Good"
    elif score >= 5: color = "🟡"; label = "Needs attention"
    else: color = "🔴"; label = "Action required"
    return score, color, label, reasons

# ─── SUMMARY TAB (contains Clusters, Shifts, Size, Sub Buckets) ───────────────
def render_summary_tab(df, total, dr, kp=""):
    # OTR vs UTR pie
    with st.expander("🥧 OTR vs UTR"):
        st.pyplot(make_pie_otr_utr(df, total, f"OTR vs UTR ({dr})"))

    # ─── CLUSTERS (combined single bar chart + drill-down) ────────────────────
    with st.expander("📍 Clusters — All Losses", expanded=True):
        cc = df["Cluster"].dropna().value_counts()
        if len(cc) > 0:
            # Combined bar chart
            st.pyplot(make_bar_horiz(cc, f"All Clusters ({dr})"))
            # Shift breakdown below bar chart
            st.markdown("**By Shift per Cluster (top 10):**")
            top_clusters = cc.head(10).index.tolist()
            shift_by_cluster = df[df["Cluster"].isin(top_clusters)].groupby(["Cluster","Shift"]).size().unstack(fill_value=0)
            shift_by_cluster = shift_by_cluster.reindex(columns=SHIFT_ORDER, fill_value=0)
            shift_by_cluster = shift_by_cluster.loc[top_clusters]
            st.dataframe(shift_by_cluster, width="stretch")
        else:
            st.info("No cluster data available.")

    # ─── CLUSTER DRILL-DOWN (selectbox) ───────────────────────────────────────
    with st.expander("🔍 Cluster Drill-Down"):
        clusters = sorted(df["Cluster"].dropna().unique())
        if clusters:
            sel_cluster = st.selectbox("Select Cluster:", clusters, key=f"{kp}sum_cl_drill")
            filt = df[df["Cluster"] == sel_cluster]
            st.write(f"**{len(filt)} parcels** in {sel_cluster} — {fmt_cost(filt['Cost (£)'].sum())}")
            # Bar chart of aisles
            ad = filt["Aisle"].dropna().value_counts()
            if len(ad) > 0:
                st.pyplot(make_bar_horiz(ad, f"{sel_cluster} — By Aisle", color="steelblue"))
            # Shift breakdown below
            st.markdown("**By Shift:**")
            sd = filt[filt["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
            st.pyplot(make_bar_shift(sd, f"{sel_cluster} — By Shift"))
            # Table
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(make_table(ad, "Aisle", "Count") if len(ad) > 0 else pd.DataFrame(), width="stretch")
            with col2:
                st.dataframe(make_table(sd, "Shift", "Count"), width="stretch")
        else:
            st.info("No cluster data available.")

    # ─── SHIFT SECTION (moved here from separate tab) ─────────────────────────
    with st.expander("⏰ Shifts — Leaderboard & Hours", expanded=True):
        st.markdown("**Shift Windows (DRM2):**")
        st.caption("NS: 23:45–09:45 | AM: 09:45–14:00 | PM: 14:00–23:45 | OTR: On The Road")
        sc = df[df["Shift"]!="Unknown"]["Shift"].value_counts()
        # Leaderboard
        rows = []
        for s in SHIFT_ORDER:
            sdf = df[df["Shift"]==s]; n = len(sdf)
            rows.append({"Shift":s,"Lost":n,"%":f"{round(n/total*100,1)}%","Cost":fmt_cost(sdf["Cost (£)"].sum()),"Window":SHIFT_DEFINITIONS[s]})
        rows.sort(key=lambda r:r["Lost"],reverse=True)
        st.dataframe(pd.DataFrame(rows,index=range(1,len(rows)+1)),width="stretch",height=200)
        # Bar chart
        if len(sc)>0: st.pyplot(make_bar_shift(sc,f"By Shift ({dr})"))
        # Hour of loss chart (from Prev Event DT)
        st.markdown("---")
        st.markdown("**⏰ Hour of Last Scan (loss timing proxy):**")
        st.caption("Uses previous_event_datetime — the time of the last scan before the parcel was lost. "
                   "Coloured by shift window. This is the best proxy for WHEN losses occur.")
        if "Prev Event DT" in df.columns:
            hours = df["Prev Event DT"].dropna().dt.hour
            if len(hours) > 0:
                hour_counts = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                fig, ax = plt.subplots(figsize=(8, 2.5))
                colors = [SHIFT_COLORS.get(SHIFT_HOUR_MAP.get(h, "Unknown"), "gray") for h in range(24)]
                ax.bar(range(24), hour_counts.values, color=colors)
                ax.set_xlabel("Hour of Day", fontsize=8); ax.set_ylabel("Parcels", fontsize=8)
                ax.set_title("Last Scan Hour Before Loss (coloured by shift)", fontsize=9)
                ax.set_xticks(range(24)); ax.tick_params(labelsize=7); plt.tight_layout()
                st.pyplot(fig)
                st.caption("🟦 NS (23:45–09:45) | 🟧 AM (09:45–14:00) | 🟩 PM (14:00–23:45)")
            else:
                st.info("No previous_event_datetime data available.")
        else:
            st.info("No previous event datetime column found.")

    # ─── SUB BUCKETS ──────────────────────────────────────────────────────────
    with st.expander("🏷️ Sub Buckets"):
        sb2 = df["Sub Bucket"].value_counts()
        if len(sb2)>0:
            vm = st.radio("View:", ["Chart","Table + Cost"], horizontal=True, key=f"{kp}sv")
            if vm=="Chart": st.pyplot(make_bar_horiz(sb2, f"Sub Buckets ({dr})", color="teal"))
            else: st.dataframe(make_cost_table(df, "Sub Bucket"), width="stretch")

    # ─── SIZE CATEGORY ────────────────────────────────────────────────────────
    with st.expander("📏 Lost Parcels by Size (Amazon UK FBA Tiers)"):
        sc_data = df["Size Category"].value_counts()
        if len(sc_data) > 0:
            size_order = ["Small Envelope", "Standard Envelope", "Standard Parcel", "Small Oversize", "Standard Oversize", "Large Oversize", "Unknown"]
            sc_ordered = sc_data.reindex([s for s in size_order if s in sc_data.index])
            vm = st.radio("View:", ["Chart", "Table + Cost"], horizontal=True, key=f"{kp}sz_v")
            if vm == "Chart":
                fig, ax = plt.subplots(figsize=(7, max(2, len(sc_ordered)*0.3)))
                colors_size = {"Small Envelope": "#2ecc71", "Standard Envelope": "#27ae60", "Standard Parcel": "#3498db",
                              "Small Oversize": "#f39c12", "Standard Oversize": "#e74c3c", "Large Oversize": "#8e44ad", "Unknown": "#95a5a6"}
                bar_colors = [colors_size.get(s, "gray") for s in sc_ordered.index]
                ax.barh(list(sc_ordered.index), sc_ordered.values, color=bar_colors)
                ax.invert_yaxis()
                for i, v in enumerate(sc_ordered.values): ax.text(v+0.2, i, str(int(v)), va="center", fontsize=7)
                ax.set_xlabel("Lost Parcels", fontsize=8); ax.set_title(f"Lost by Size Tier ({dr})", fontsize=9)
                ax.tick_params(labelsize=7); plt.tight_layout()
                st.pyplot(fig)
            else:
                size_tbl = df.groupby("Size Category").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).reindex([s for s in size_order if s in df["Size Category"].values]).reset_index()
                size_tbl["Cost"] = size_tbl["Cost"].apply(fmt_cost)
                size_tbl["% of Total"] = (size_tbl["Lost"] / total * 100).round(1)
                size_tbl.index = range(1, len(size_tbl)+1)
                st.dataframe(size_tbl, width="stretch")
            st.caption("Size tiers based on longest side: Small Envelope ≤20cm, Std Envelope ≤33cm, Std Parcel ≤45cm, Small Oversize ≤61cm, Std Oversize ≤120cm, Large Oversize >120cm")
        else:
            st.info("No size data available.")


# ─── LOCATIONS TAB (simplified: Top 10s + OTR/UTR filter) ─────────────────────
def render_locations_tab(df, total, dr, kp=""):
    lf = st.radio("Show:", ["All Parcels","OTR Only","UTR Only"], horizontal=True, key=f"{kp}lf")
    if lf == "OTR Only":
        vdf = df[df["Type"]=="OTR"].copy(); st.write(f"**{len(vdf)} OTR** — {fmt_cost(vdf['Cost (£)'].sum())}")
        if len(vdf)==0: return
        with st.expander("🚚 Top 10 DSPs", expanded=True):
            d = vdf["DSP Name"].dropna().value_counts().head(10)
            if len(d)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}od")
                if vm=="Chart": st.pyplot(make_bar_horiz(d,f"Top 10 DSPs — OTR ({dr})",color="firebrick",max_label=DSP_MAX))
                else: st.dataframe(make_cost_table(vdf.dropna(subset=["DSP Name"]),"DSP Name").head(10),width="stretch")
        with st.expander("📍 Top 10 Delivery Areas"):
            ab = st.radio("By:",["City","Province","Postal"],horizontal=True,key=f"{kp}oa")
            ad = vdf[ab].dropna().value_counts().head(10)
            if len(ad)>0:
                st.pyplot(make_bar_horiz(ad,f"Top 10 {ab} — OTR ({dr})",color="darkred"))
        with st.expander("❓ Top Loss Reasons"):
            r = vdf["Loss Reason"].dropna().value_counts().head(10)
            if len(r)>0: st.pyplot(make_bar_horiz(r,"Top OTR Reasons",color="crimson"))
    elif lf == "UTR Only":
        vdf = df[df["Type"]=="UTR"].copy(); st.write(f"**{len(vdf)} UTR** — {fmt_cost(vdf['Cost (£)'].sum())}")
        if len(vdf)==0: return
        with st.expander("📍 Top 10 Clusters", expanded=True):
            cl = vdf["Cluster"].dropna().value_counts().head(10)
            if len(cl)>0: st.pyplot(make_bar_horiz(cl,f"Top 10 Clusters — UTR ({dr})",color="darkorange"))
        with st.expander("🏷️ Top 10 Aisles"):
            al = vdf["Aisle"].dropna().value_counts().head(10)
            if len(al)>0: st.pyplot(make_bar_horiz(al,f"Top 10 Aisles — UTR ({dr})",color="orange"))
        with st.expander("🏷️ Sub Buckets"):
            sb = vdf["Sub Bucket"].value_counts()
            if len(sb)>0: st.pyplot(make_bar_horiz(sb,"UTR Sub Buckets",color="darkorange"))
    else:
        vdf = df.copy(); st.write(f"**{len(vdf)} all** — {fmt_cost(vdf['Cost (£)'].sum())}")
        with st.expander("📍 Top 10 Clusters", expanded=True):
            cl = vdf["Cluster"].dropna().value_counts().head(10)
            if len(cl)>0: st.pyplot(make_bar_horiz(cl,f"Top 10 Clusters ({dr})"))
        with st.expander("🏷️ Top 10 Aisles"):
            al = vdf["Aisle"].dropna().value_counts().head(10)
            if len(al)>0: st.pyplot(make_bar_horiz(al,f"Top 10 Aisles ({dr})",color="teal"))
        with st.expander("🗂️ Top 10 Sort Zones"):
            sz = vdf["Sort Zone"].dropna().value_counts().head(10)
            if len(sz)>0: st.pyplot(make_bar_horiz(sz,f"Top 10 Sort Zones ({dr})",color="purple"))
        with st.expander("🚚 Top 10 DSPs"):
            d = vdf["DSP Name"].dropna().value_counts().head(10)
            if len(d)>0: st.pyplot(make_bar_horiz(d,f"Top 10 DSPs ({dr})",color="firebrick",max_label=DSP_MAX))


# ─── PNOV TAB (with Stowed By associate for upskilling) ───────────────────────
def render_pnov_tab(df, total, dr, kp=""):
    st.markdown("#### 📦 PNOV — Package Not On Vehicle")
    st.caption("PNOV parcels were dispatched but never confirmed delivered or returned. "
               "This tab identifies responsible DSPs/Flex drivers and — for stowed parcels — "
               "the associate who stowed them (for upskilling).")

    pnov_keywords = ["PNOV", "No Further Status", "Package Not On Vehicle"]
    pnov_mask = df["Sub Bucket"].fillna("").apply(lambda x: any(k.lower() in x.lower() for k in pnov_keywords))
    pnov_df = df[pnov_mask].copy()

    if len(pnov_df) == 0:
        st.info("No PNOV parcels found in this dataset. PNOV parcels have sub-bucket containing 'No Further Status' or 'PNOV'.")
        return

    st.markdown(f"**{len(pnov_df)} PNOV parcels** — {fmt_cost(pnov_df['Cost (£)'].sum())} ({round(len(pnov_df)/total*100,1)}% of all losses)")

    c1, c2, c3 = st.columns(3)
    flex_count = len(pnov_df[pnov_df["Driver Type"] == "Flex"])
    dsp_count = len(pnov_df[pnov_df["Driver Type"] == "DSP"])
    c1.metric("Flex Drivers", flex_count)
    c2.metric("DSP Drivers", dsp_count)
    c3.metric("Avg Cost/Parcel", fmt_cost(pnov_df["Cost (£)"].mean()))

    # Responsible DSPs/Flex
    with st.expander("🚚 Responsible Drivers (DSP + Flex)", expanded=True):
        dsp_data = pnov_df.dropna(subset=["DSP Name"])
        if len(dsp_data) > 0:
            dc = dsp_data["DSP Name"].value_counts()
            vm = st.radio("View:", ["Chart", "Table + Cost"], horizontal=True, key=f"{kp}pnov_dsp_v")
            if vm == "Chart":
                st.pyplot(make_bar_horiz(dc, f"PNOV by Driver ({dr})", color="firebrick", max_label=DSP_MAX))
            else:
                tbl = dsp_data.groupby("DSP Name").agg(
                    PNOV_Count=("Tracking ID", "count"),
                    Cost=("Cost (£)", "sum"),
                    Driver_Type=("Driver Type", "first")
                ).sort_values("PNOV_Count", ascending=False).reset_index()
                tbl["Cost"] = tbl["Cost"].apply(fmt_cost)
                tbl.index = range(1, len(tbl)+1)
                st.dataframe(tbl, width="stretch")
            # Flex vs DSP
            st.markdown("---")
            type_counts = dsp_data["Driver Type"].value_counts()
            if len(type_counts) > 0:
                fig, ax = plt.subplots(figsize=(3, 1.5))
                colors_map = {"Flex": "#e74c3c", "DSP": "#3498db", "Unknown": "#95a5a6"}
                ax.bar(type_counts.index, type_counts.values, color=[colors_map.get(x, "gray") for x in type_counts.index])
                for i, v in enumerate(type_counts.values): ax.text(i, v+0.2, str(int(v)), ha="center", fontsize=8)
                ax.set_ylabel("PNOV Count", fontsize=8); ax.set_title("PNOV: Flex vs DSP", fontsize=9)
                ax.tick_params(labelsize=7); plt.tight_layout()
                st.pyplot(fig)

    # ─── STOWED BY ASSOCIATES (upskilling list) ──────────────────────────────
    with st.expander("👤 Associates — Stowed By (Upskilling List)", expanded=True):
        st.caption("If a parcel was 'Stowed Not Picked Up' before being PNOV, "
                   "the associate who last stowed it is shown here. "
                   "Use this list for upskilling conversations.")
        has_stowed = pnov_df["Stowed By"].dropna()
        if len(has_stowed) > 0:
            assoc_counts = has_stowed.value_counts()
            st.markdown(f"**{len(assoc_counts)} associate(s)** involved in stowing PNOV parcels:")
            # Table with associate, count, parcels
            assoc_tbl = pnov_df[pnov_df["Stowed By"].notna()].groupby("Stowed By").agg(
                Parcels=("Tracking ID", "count"),
                Cost=("Cost (£)", "sum"),
                Clusters=("Cluster", lambda x: ", ".join(x.dropna().unique()[:3]))
            ).sort_values("Parcels", ascending=False).reset_index()
            assoc_tbl["Cost"] = assoc_tbl["Cost"].apply(fmt_cost)
            assoc_tbl.index = range(1, len(assoc_tbl)+1)
            st.dataframe(assoc_tbl, width="stretch")
            # Download button
            st.download_button("⬇️ Download upskilling list", assoc_tbl.to_csv(index=False), "PNOV_upskilling_associates.csv", "text/csv", key=f"{kp}dl_upskill")
        else:
            st.warning("No 'Stowed By' data available. This column needs to be in the SCC export. "
                       "Check if your SCC export includes 'Stowed By' or 'Last Scan By' column.")
            st.info("**How to get this data:** In SCC, ensure you export with the 'Stowed By' or 'Last Scan By' column included. "
                    "This shows which associate was responsible for the last stow action on the parcel.")

    # By Cluster
    with st.expander("📍 PNOV by Cluster"):
        cl = pnov_df["Cluster"].dropna().value_counts()
        if len(cl) > 0:
            st.pyplot(make_bar_horiz(cl.head(15), f"PNOV by Cluster ({dr})", color="purple"))

    # All PNOV tracking IDs
    with st.expander("📋 All PNOV Tracking IDs"):
        tid_cols = [c for c in ["Tracking ID","Marked Lost DT","Cluster","Aisle","DSP Name","Driver Type","Stowed By","Cost (£)","Loss Reason"] if c in pnov_df.columns]
        tid_df = pnov_df[tid_cols].copy()
        if "Marked Lost DT" in tid_df.columns:
            tid_df = tid_df.sort_values("Marked Lost DT", ascending=True, na_position="last")
        tid_df = tid_df.reset_index(drop=True)
        tid_df.index = range(1, len(tid_df)+1)
        st.dataframe(tid_df, width="stretch", height=400)
        st.download_button("⬇️ Download PNOV parcels", tid_df.to_csv(index=False), "PNOV_parcels.csv", "text/csv", key=f"{kp}dl_pnov")


# ─── COST TAB ─────────────────────────────────────────────────────────────────
def render_cost_tab(df, total, dr, kp=""):
    tc = df["Cost (£)"].sum(); avg = tc/total if total>0 else 0
    otr_df = df[df["Type"]=="OTR"]; utr_df = df[df["Type"]=="UTR"]
    st.markdown(f"### 💰 {fmt_cost(tc)} total ({total} parcels, avg {fmt_cost(avg)}/parcel)")
    st.markdown(f"**OTR:** {len(otr_df)} — {fmt_cost(otr_df['Cost (£)'].sum())} | **UTR:** {len(utr_df)} — {fmt_cost(utr_df['Cost (£)'].sum())}")
    with st.expander("💰 By Sub Bucket + Type", expanded=True):
        csb = df.groupby(["Sub Bucket","Type"]).agg(Count=("Tracking ID","count"),Cost=("Cost (£)","sum")).sort_values("Cost",ascending=False).reset_index()
        csb["Avg"] = (csb["Cost"]/csb["Count"]).apply(fmt_cost); csb["Cost"] = csb["Cost"].apply(fmt_cost)
        csb.index = range(1,len(csb)+1)
        vm = st.radio("View:",["Table","Chart"],horizontal=True,key=f"{kp}csb")
        if vm=="Chart":
            cd = df.groupby("Sub Bucket")["Cost (£)"].sum().sort_values(ascending=False)
            h = max(2,len(cd)*0.3); fig,ax = plt.subplots(figsize=(7,h))
            ax.barh(trunc(cd.index,30),cd.values,color="teal"); ax.invert_yaxis()
            for i,v in enumerate(cd.values): ax.text(v+0.5,i,fmt_cost(v),va="center",fontsize=6)
            ax.set_xlabel("£",fontsize=8); ax.set_title("Cost by Sub Bucket",fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
        else: st.dataframe(csb,width="stretch")
    with st.expander("💰 By DSP (with top reason)"):
        dsp_df = df.dropna(subset=["DSP Name"])
        if len(dsp_df)>0:
            dc = dsp_df.groupby("DSP Name").agg(Count=("Tracking ID","count"),Cost=("Cost (£)","sum")).sort_values("Cost",ascending=False).reset_index()
            reasons = []
            for dsp in dc["DSP Name"]:
                dp = dsp_df[dsp_df["DSP Name"]==dsp]; rc = dp["Loss Reason"].dropna().value_counts()
                reasons.append(rc.index[0] if len(rc)>0 else "N/A")
            dc["Top Reason"] = reasons; dc["Avg"] = (dc["Cost"]/dc["Count"]).apply(fmt_cost); dc["Cost"] = dc["Cost"].apply(fmt_cost)
            dc.index = range(1,len(dc)+1)
            st.dataframe(dc,width="stretch")
    with st.expander("💰 Top 10 Most Expensive Parcels"):
        top = df.nlargest(10,"Cost (£)")
        sc2 = [c for c in ["Tracking ID","Sub Bucket","Type","Shift","Cost (£)","DSP Name","Cluster","Loss Reason"] if c in top.columns]
        out = top[sc2].reset_index(drop=True); out.index = range(1,len(out)+1); st.dataframe(out,width="stretch")


# ─── ANALYSIS TAB ─────────────────────────────────────────────────────────────
def render_analysis_tab(df, total, dr, kp=""):
    st.markdown("### 🔬 Analysis")
    st.warning("⚠️ This section helps you process data and offers suggestions, "
               "but **SHOULD NOT BE USED AS A GUIDE, IT IS ONLY AN AID.** "
               "Always use your own judgement and local knowledge.")

    total_cost = df["Cost (£)"].sum()
    findings = []

    # Loss Reasons
    with st.expander("📊 Loss Reasons", expanded=True):
        lr = df.groupby("Loss Reason").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
        lr["% of Total"] = (lr["Count"] / total * 100).round(1).astype(str) + "%"
        lr["Avg £"] = (lr["Cost"] / lr["Count"]).apply(fmt_cost)
        lr["Cost"] = lr["Cost"].apply(fmt_cost)
        lr.index = range(1, len(lr)+1)
        st.dataframe(lr, width="stretch")

    # Cluster concentration
    with st.expander("📍 Cluster Concentration"):
        cl_c = df["Cluster"].dropna().value_counts()
        if len(cl_c) >= 2:
            top3 = cl_c.head(3)
            top3_pct = round(top3.sum() / cl_c.sum() * 100, 1)
            top3_cost = df[df["Cluster"].isin(top3.index)]["Cost (£)"].sum()
            vals = cl_c.values.astype(float); n = len(vals); sv = np.sort(vals)
            gini = (2 * np.sum(np.arange(1, n+1) * sv) - (n+1) * np.sum(sv)) / (n * np.sum(sv))
            if gini > 0.5:
                st.error(f"🎯 Losses piling up. Top 3 clusters = **{top3_pct}%** ({fmt_cost(top3_cost)})")
            elif gini > 0.3:
                st.warning(f"⚠️ Some concentration. Top 3 = **{top3_pct}%**")
            else:
                st.success(f"✅ Fairly spread. Top 3 = {top3_pct}%")
            findings.append(f"Top 3 clusters = {top3_pct}% of losses ({fmt_cost(top3_cost)})")

    # DSP outliers
    with st.expander("🚚 DSP Outliers"):
        otr = df[df["Type"] == "OTR"]
        if len(otr) >= 5:
            dc = otr["DSP Name"].dropna().value_counts()
            if len(dc) >= 3:
                mu = dc.mean(); sigma = dc.std()
                z = (dc - mu) / sigma if sigma > 0 else pd.Series(0, index=dc.index)
                outliers = z[z > 1.5]
                tbl = pd.DataFrame({"DSP": dc.index, "Losses": dc.values, "Verdict": ["🎯 Above average" if x > 1.5 else "✅ Normal" for x in z.values]})
                tbl.index = range(1, len(tbl)+1); st.dataframe(tbl, width="stretch")
                if len(outliers) > 0:
                    st.error(f"🎯 {len(outliers)} DSP(s) losing significantly more than peers.")
                    for dsp, zv in outliers.items():
                        findings.append(f"DSP '{dsp}': {int(dc[dsp])} losses (avg {mu:.1f})")

    # Size analysis
    with st.expander("📏 Size Impact"):
        sc3 = df["Size Category"].value_counts()
        if len(sc3) >= 2:
            oversize_cats = ["Small Oversize", "Standard Oversize", "Large Oversize"]
            ov = sum(sc3.get(k, 0) for k in oversize_cats)
            ov_pct = round(ov / total * 100, 1)
            if ov_pct > 30:
                st.error(f"🎯 Oversized = **{ov_pct}%** of losses (elevated)")
            elif ov_pct > 20:
                st.warning(f"⚠️ Oversized = {ov_pct}% — slightly elevated")
            else:
                st.success(f"✅ Oversized = {ov_pct}% — normal range")
            stbl = df.groupby("Size Category").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Count", ascending=False).reset_index()
            stbl["Cost"] = stbl["Cost"].apply(fmt_cost); stbl.index = range(1, len(stbl)+1)
            st.dataframe(stbl, width="stretch")

    # Repeat aisles
    with st.expander("🔁 Repeat Offender Aisles"):
        aisle_c = df["Aisle"].dropna().value_counts()
        repeat = aisle_c[aisle_c >= 3]
        if len(repeat) > 0:
            st.error(f"🎯 {len(repeat)} aisle(s) lost 3+ parcels:")
            repeat_df = pd.DataFrame({"Aisle": repeat.index, "Losses": repeat.values})
            repeat_df.index = range(1, len(repeat_df)+1)
            st.dataframe(repeat_df, width="stretch")
        else:
            st.success("✅ No single aisle has 3+ losses.")

    if findings:
        st.markdown("---")
        st.markdown("#### 💡 Key findings")
        for i, f in enumerate(findings, 1):
            st.markdown(f"**{i}.** {f}")


# ─── TREND TAB ────────────────────────────────────────────────────────────────
def render_trend_tab(df, total, dr, kp=""):
    st.markdown("#### 📈 Week-over-Week Trend")
    st.warning("⚠️ Trends need 3-4+ weeks to be meaningful.")
    st.caption("Enter totals from PerfectMile → L&U → Lost Focus (Weekly view)")

    num_weeks = st.slider("How many weeks?", 1, 12, 4, key=f"{kp}nw")
    weeks_data = []
    cols = st.columns(min(num_weeks, 6))
    for i in range(num_weeks):
        with cols[i % 6]:
            wk_label = st.text_input(f"Week {i+1}:", value=f"W{i+1}", key=f"{kp}wl{i}")
            wk_count = st.number_input(f"Total lost:", min_value=0, value=0, step=1, key=f"{kp}wc{i}")
            if wk_count > 0:
                weeks_data.append({"Week": wk_label, "Total": int(wk_count)})

    if len(weeks_data) >= 2:
        weekly = pd.DataFrame(weeks_data)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(weekly["Week"], weekly["Total"], marker="o", color="steelblue", linewidth=2)
        for i, row in weekly.iterrows():
            ax.annotate(str(int(row["Total"])), xy=(row["Week"], row["Total"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        avg = weekly["Total"].mean()
        ax.axhline(y=avg, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Week", fontsize=8); ax.set_ylabel("Losses", fontsize=8)
        ax.set_title("Lost Parcels per Week", fontsize=9); ax.tick_params(labelsize=7)
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)
        first_w = int(weekly.iloc[0]["Total"]); last_w = int(weekly.iloc[-1]["Total"])
        pct_change = round((last_w - first_w) / first_w * 100, 1) if first_w > 0 else 0
        if last_w < first_w * 0.8:
            st.success(f"📉 **Improving!** {first_w} → {last_w} ({pct_change:+.1f}%)")
        elif last_w > first_w * 1.2:
            st.error(f"📈 **Getting worse.** {first_w} → {last_w} ({pct_change:+.1f}%)")
        else:
            st.info(f"➡️ **Stable.** {first_w} → {last_w} ({pct_change:+.1f}%)")
        st.download_button("⬇️ Download trend", weekly.to_csv(index=False), "weekly_trend.csv", "text/csv", key=f"{kp}dl_trend")
    elif len(weeks_data) == 1:
        st.info("Enter 1 more week to compare.")


# ─── EXPORT TAB ───────────────────────────────────────────────────────────────
def render_export_tab(df, total, dr, kp="", station_name=""):
    st.markdown("#### 💾 Export")
    import zipfile
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","shipment_value"]
        clean_cols = [c for c in df.columns if c not in exc]
        zf.writestr("All_Data.csv", df[clean_cols].to_csv(index=False))
        if df["Cluster"].dropna().count() > 0:
            loc_df = df.groupby("Cluster").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Lost", ascending=False).reset_index()
            zf.writestr("By_Cluster.csv", loc_df.to_csv(index=False))
        shift_df = df[df["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).reindex(SHIFT_ORDER).reset_index()
        zf.writestr("By_Shift.csv", shift_df.to_csv(index=False))
        dsp_data = df.dropna(subset=["DSP Name"])
        if len(dsp_data) > 0:
            dsp_df = dsp_data.groupby("DSP Name").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Lost", ascending=False).reset_index()
            zf.writestr("By_DSP.csv", dsp_df.to_csv(index=False))
    output.seek(0)
    fname = f"{station_name}_Analysis.zip" if station_name else "Lost_Parcel_Analysis.zip"
    st.download_button(f"⬇️ Download ZIP ({fname})", output, fname, "application/zip", key=f"{kp}dl_zip")
    st.markdown("---")
    exc2 = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","shipment_value"]
    ec = [c for c in df.columns if c not in exc2]
    fname2 = f"{station_name}_Clean.csv" if station_name else "Lost_Clean.csv"
    st.download_button(f"⬇️ Single CSV", df[ec].to_csv(index=False), fname2, "text/csv", key=f"{kp}dl_clean")


# ─── GUIDE ────────────────────────────────────────────────────────────────────
def render_guide():
    st.markdown("### 📖 How to Use This Tool")
    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
**You need two CSV files:**

| File | Where to get it |
|------|----------------|
| **Perfect Mile** | PerfectMile → L&U → Lost → Export CSV |
| **SCC** | SCC → paste Tracking IDs → Export |

**Steps:** Upload both → Summary tab → Pick one problem → Go observe
""")
    with st.expander("📊 What each tab shows"):
        st.markdown("""
| Tab | What it tells you |
|-----|------------------|
| 📊 **Summary** | Clusters, shifts, sub-buckets, size, hour of loss |
| 📍 **Locations** | Top 10s — clusters, aisles, DSPs, sort zones |
| 📦 **PNOV** | Package Not On Vehicle — drivers + associates for upskilling |
| 💰 **Cost** | Financial impact by type, DSP |
| 🔬 **Analysis** | Pattern detection (an AID, not a GUIDE) |
| 📈 **Trend** | Week-over-week (needs 2+ weeks) |
| 💾 **Export** | Download for further analysis |
""")
    with st.expander("📏 Size Classification (Amazon UK FBA Tiers)"):
        st.markdown(SIZE_TIER_INFO)


# ─── MISSING PARCELS ──────────────────────────────────────────────────────────
def render_missing_parcels(df, total, matched):
    mc = total - matched
    if mc > 0:
        st.info(f"ℹ️ **{mc} parcel(s)** in PM had no SCC match — included but no location data.")
        with st.expander(f"🔍 View {mc} unmatched"):
            mdf = df[df["Cluster"].isna()].copy()
            if len(mdf)>0:
                sel = st.selectbox("Parcel:", mdf["Tracking ID"].tolist(), key="miss")
                r = mdf[mdf["Tracking ID"]==sel].iloc[0]
                st.markdown(f"**TID:** {sel} | **Sub Bucket:** {r.get('Sub Bucket','N/A')} | **Type:** {r.get('Type','N/A')} | **Shift:** {r.get('Shift','N/A')} | **Cost:** {fmt_cost(r.get('Cost (£)'))}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
mode = st.radio("Mode:", ["📖 Guide","Single Station","Multi-Station / Compare"], horizontal=True, key="mode")

if mode == "📖 Guide":
    render_guide()

elif mode == "Single Station":
    c_pm, c_scc = st.columns(2)
    with c_pm: pm_file = st.file_uploader("📊 Perfect Mile", type="csv", key="pm")
    with c_scc: scc_file = st.file_uploader("📋 SCC", type="csv", key="scc")
    if pm_file and scc_file:
        pm_df, scc_df = pd.read_csv(pm_file), pd.read_csv(scc_file)
        pm_miss = [c for c in REQUIRED_PM_COLS if c not in pm_df.columns]
        if pm_miss: st.error(f"❌ PM missing: {pm_miss}"); st.stop()
        scc_miss = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
        if scc_miss: st.error(f"❌ SCC missing: {scc_miss}"); st.stop()
        found = [c for c in SENSITIVE_COLS if c in scc_df.columns]
        if found: st.warning(f"🔒 PII removed: {', '.join(found)}")
        df = merge_data(pm_df, scc_df); total = len(df)
        if total == 0: st.stop()
        matched = df["Cluster"].notna().sum(); tc = df["Cost (£)"].sum()
        st.success(f"✅ **{total} parcels** — {fmt_cost(tc)} (PM:{len(pm_df)}, SCC:{len(scc_df)}, Matched:{matched})")
        render_missing_parcels(df, total, matched)
        dr = get_date_range(df)
        # Health Score
        score, color, label, score_reasons = render_health_score(df, total)
        st.markdown(f"**Health Score: {color} {score}/10 — {label}**" + (f" ({', '.join(score_reasons)})" if score_reasons else ""))
        # Metrics
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Lost", total); c2.metric("Cost", fmt_cost(tc)); c3.metric("Cluster", safe_top(df["Cluster"]))
        c4.metric("Aisle", safe_top(df["Aisle"])); c5.metric("DSP", str(safe_top(df["DSP Name"]))[:15])
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]; c6.metric("Shift", safe_top(sk) if len(sk)>0 else "N/A")
        # TABS (simplified: Summary, Locations, PNOV, Cost, Analysis, Trend, Export)
        t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📊 Summary","📍 Locations","📦 PNOV","💰 Cost","🔬 Analysis","📈 Trend","💾 Export"])
        with t1: render_summary_tab(df, total, dr, kp="s_")
        with t2: render_locations_tab(df, total, dr, kp="s_")
        with t3: render_pnov_tab(df, total, dr, kp="s_")
        with t4: render_cost_tab(df, total, dr, kp="s_")
        with t5: render_analysis_tab(df, total, dr, kp="s_")
        with t6: render_trend_tab(df, total, dr, kp="s_")
        with t7: render_export_tab(df, total, dr, kp="s_")
    else: st.info("👆 Upload both files.")

else:
    st.caption("Upload multiple stations or time periods to compare.")
    num = st.slider("Datasets to compare:", 2, 5, 2, key="ns"); uploaded = {}
    for i in range(num):
        with st.expander(f"Dataset {i+1}", expanded=(i<2)):
            nm_input = st.text_input(f"Label:", key=f"nm_{i}", placeholder=f"e.g. DRM2 Week 1")
            a, b = st.columns(2)
            with a: pf = st.file_uploader(f"PM ({i+1})", type="csv", key=f"mp{i}")
            with b: sf = st.file_uploader(f"SCC ({i+1})", type="csv", key=f"ms{i}")
            if pf and sf: uploaded[i] = (pf, sf, nm_input)
    if len(uploaded) >= 2:
        stations, names = {}, []
        for i, (pf, sf, nm_input) in uploaded.items():
            pt, s2 = pd.read_csv(pf), pd.read_csv(sf); m = merge_data(pt, s2)
            nm = nm_input.strip() if nm_input and nm_input.strip() else f"Dataset {i+1}"
            stations[nm] = m; names.append(nm)
        st.success(f"✅ {', '.join(names)}")
        for n in names:
            sc_val, sc_col, sc_lab, sc_reas = render_health_score(stations[n], len(stations[n]))
            st.caption(f"{n}: {sc_col} {sc_val}/10 — {sc_lab}" + (f" ({', '.join(sc_reas)})" if sc_reas else ""))
        t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📊 Summary","📍 Locations","📦 PNOV","💰 Cost","🔬 Analysis","📈 Trend","💾 Export"])
        with t1:
            for n in names:
                sdf = stations[n]
                st.markdown(f"**{n}:** {len(sdf)} parcels — {fmt_cost(sdf['Cost (£)'].sum())} | Top cluster: {safe_top(sdf['Cluster'])} | Top shift: {safe_top(sdf[sdf['Shift'].isin(SHIFT_ORDER)]['Shift'])}")
        with t2:
            sel = st.selectbox("Dataset:", names, key="mcl"); render_locations_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"m{sel}_")
        with t3:
            sel = st.selectbox("Dataset:", names, key="mcp"); render_pnov_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mp{sel}_")
        with t4:
            sel = st.selectbox("Dataset:", names, key="mcc"); render_cost_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc{sel}_")
        with t5:
            sel = st.selectbox("Dataset:", names, key="mca"); render_analysis_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"ma{sel}_")
        with t6:
            sel = st.selectbox("Dataset:", names, key="mct"); render_trend_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mt{sel}_")
        with t7:
            sel = st.selectbox("Dataset:", names, key="mce"); render_export_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"me{sel}_", station_name=sel)
    elif len(uploaded) == 1: st.warning("Need 2+ datasets to compare.")
    else: st.info("👆 Upload pairs of PM + SCC files above.")
