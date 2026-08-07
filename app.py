import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from io import BytesIO

st.set_page_config(page_title="DRM2 Lost Heatmap", page_icon="📦", layout="wide")
st.title("📦 DRM2 Lost Parcel Heatmap")
st.markdown("---")

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

# Potential SCC columns that might hold the "stowed by" associate name
STOWED_BY_CANDIDATES = [
    "Stowed By", "stowed_by", "Stowed_By", "StowedBy",
    "Last Scan By", "last_scan_by", "Last_Scan_By",
    "Scanned By", "scanned_by", "Container Scan By",
    "Inducted By", "inducted_by", "Pick Up By", "pick_up_by",
    "Picked By", "picked_by"
]

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────
def get_size(val):
    if pd.isna(val): return "Unknown"
    try:
        val = float(val)
    except (ValueError, TypeError):
        return "Unknown"
    if val <= 20: return "Small Envelope"
    if val <= 33: return "Standard Envelope"
    if val <= 45: return "Standard Parcel"
    if val <= 61: return "Small Oversize"
    if val <= 120: return "Standard Oversize"
    return "Large Oversize"

def hour_to_shift(hour):
    if pd.isna(hour): return "Unknown"
    try:
        return SHIFT_HOUR_MAP.get(int(hour), "Unknown")
    except (ValueError, TypeError):
        return "Unknown"

def assign_shift(row):
    sb = row.get("Sub Bucket", "")
    if pd.notna(sb) and sb in SUB_BUCKET_SHIFT_MAP:
        return SUB_BUCKET_SHIFT_MAP[sb]
    prev_dt = row.get("Prev Event DT")
    if pd.notna(prev_dt):
        try:
            return hour_to_shift(prev_dt.hour)
        except (AttributeError, TypeError):
            pass
    disp_dt = row.get("Dispatch Time")
    if pd.notna(disp_dt):
        try:
            return hour_to_shift(disp_dt.hour)
        except (AttributeError, TypeError):
            pass
    cyc = row.get("Assigned Cycle", "")
    if pd.notna(cyc):
        u = str(cyc).upper()
        if "NS" in u or "NIGHT" in u: return "NS"
        if "PM" in u or "RELO" in u or "C2" in u: return "PM"
        if "AM" in u or "C1" in u: return "AM"
    return "Unknown"

def classify_otr_utr(sub_bucket):
    if pd.isna(sub_bucket): return "Unknown"
    s = str(sub_bucket)
    if "Lost On Road" in s: return "OTR"
    if "Lost At Station" in s: return "UTR"
    return "Unknown"

def is_flex_driver(dsp_name):
    if pd.isna(dsp_name): return False
    return "CSP_COMPANY_NAME" in str(dsp_name).upper()

def clean_dsp_name(dsp_name):
    if pd.isna(dsp_name): return dsp_name
    if is_flex_driver(dsp_name): return "FLEX DRIVER"
    return str(dsp_name).strip()

def find_stowed_by_column(df):
    """Search all SCC columns for the one that holds the stowed-by associate."""
    for candidate in STOWED_BY_CANDIDATES:
        if candidate in df.columns:
            non_null = df[candidate].dropna()
            if len(non_null) > 0:
                return candidate
    # Fallback: look for any column with 'stow' or 'scan' in name
    for col in df.columns:
        cl = col.lower()
        if ("stow" in cl or "scan" in cl) and "time" not in cl and "date" not in cl:
            non_null = df[col].dropna()
            if len(non_null) > 0 and non_null.dtype == "object":
                return col
    return None

def clean_scc(df):
    # Remove sensitive columns but DON'T remove all others — keep everything
    # that might be useful (including potential stowed-by columns)
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])
    # Parse dimensions
    for col in ["Package Length","Package Width","Package Height"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"\s*cm", "", regex=True), errors="coerce")
    dims = ["Package Length","Package Width","Package Height"]
    if all(c in df.columns for c in dims):
        df["Longest Side"] = df[dims].max(axis=1)
    else:
        df["Longest Side"] = float("nan")
    df["Size Category"] = df["Longest Side"].apply(get_size)
    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], dayfirst=True, errors="coerce")
    if "Dispatch Time" in df.columns:
        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], dayfirst=True, errors="coerce")
    if "Province" in df.columns:
        df["Province"] = df["Province"].astype(str).str.strip().str.title().replace("Nan", pd.NA)
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip().str.title().replace("Nan", pd.NA)
    if "DSP Name" in df.columns:
        df["DSP Name"] = df["DSP Name"].apply(clean_dsp_name)
    # Find the stowed-by column dynamically
    stow_col = find_stowed_by_column(df)
    if stow_col and stow_col != "Stowed By":
        df["Stowed By"] = df[stow_col]
    elif "Stowed By" not in df.columns:
        df["Stowed By"] = None
    return df

def merge_data(pm_df, scc_df):
    scc_clean = clean_scc(scc_df.copy())
    pm_keep = ["tracking_id","bucket","sub_bucket","previous_event_datetime","previous_reason","previous_reason_3","event_datetime","shipment_value"]
    pm_cols = pm_df[[c for c in pm_keep if c in pm_df.columns]].copy()
    pm_cols = pm_cols.rename(columns={"tracking_id":"Tracking ID"})
    # Parse dates robustly
    if "previous_event_datetime" in pm_cols.columns:
        pm_cols["Prev Event DT"] = pd.to_datetime(pm_cols["previous_event_datetime"], format="%d/%m/%Y %H:%M", errors="coerce")
        # Fallback: try dayfirst
        mask = pm_cols["Prev Event DT"].isna() & pm_cols["previous_event_datetime"].notna()
        if mask.any():
            pm_cols.loc[mask, "Prev Event DT"] = pd.to_datetime(pm_cols.loc[mask, "previous_event_datetime"], dayfirst=True, errors="coerce")
    else:
        pm_cols["Prev Event DT"] = pd.NaT
    if "event_datetime" in pm_cols.columns:
        pm_cols["Marked Lost DT"] = pd.to_datetime(pm_cols["event_datetime"], dayfirst=True, errors="coerce")
    if "shipment_value" in pm_cols.columns:
        pm_cols["Cost (£)"] = pd.to_numeric(pm_cols["shipment_value"].astype(str).str.replace("[£$,]", "", regex=True), errors="coerce")
    else:
        pm_cols["Cost (£)"] = 0.0
    merged = pm_cols.merge(scc_clean, on="Tracking ID", how="left")
    merged["Sub Bucket"] = merged.get("sub_bucket", pd.Series(dtype="object"))
    merged["Bucket"] = merged.get("bucket", pd.Series(dtype="object"))
    merged["Type"] = merged["Sub Bucket"].apply(classify_otr_utr)
    if "previous_reason" in merged.columns:
        merged["Loss Reason"] = merged["previous_reason"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else:
        merged["Loss Reason"] = "Unknown"
    if "previous_reason_3" in merged.columns:
        merged["UTR Reason"] = merged["previous_reason_3"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else:
        merged["UTR Reason"] = "Unknown"
    merged["Shift"] = merged.apply(assign_shift, axis=1)
    merged["Driver Type"] = merged.get("DSP Name", pd.Series(dtype="object")).apply(
        lambda x: "Flex" if x == "FLEX DRIVER" else ("DSP" if pd.notna(x) else "Unknown"))
    # Ensure all needed columns exist
    for col in ["Cluster","Aisle","Sort Zone","DSP Name","Size Category","City","Province","Postal","Cost (£)","Stowed By"]:
        if col not in merged.columns: merged[col] = None
    # Fill NaN costs with 0
    merged["Cost (£)"] = merged["Cost (£)"].fillna(0)
    return merged

def get_date_range(df):
    for col in ["Marked Lost DT","Dispatch Time","Last Updated Time"]:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid)>0:
                s,e = valid.min().strftime("%d %b %Y"), valid.max().strftime("%d %b %Y")
                return s if s==e else f"{s} – {e}"
    return ""
def safe_top(s):
    try:
        c = s.dropna().value_counts()
        return c.index[0] if len(c)>0 else "N/A"
    except Exception:
        return "N/A"
def trunc(labels, mx=LABEL_MAX): return [str(l)[:mx]+"..." if len(str(l))>mx else str(l) for l in labels]
def fmt_cost(val):
    try:
        if pd.isna(val): return "£0.00"
        return f"£{float(val):,.2f}"
    except (ValueError, TypeError):
        return "£0.00"
def make_table(series, c1, c2):
    if len(series) == 0: return pd.DataFrame(columns=[c1, c2])
    t = series.reset_index(); t.columns = [c1, c2]; t.index = range(1,len(t)+1); return t
def make_cost_table(df, group_col):
    valid = df.dropna(subset=[group_col])
    if len(valid) == 0: return pd.DataFrame(columns=[group_col, "Lost", "Cost Lost"])
    grouped = valid.groupby(group_col).agg(Lost=("Tracking ID","count"), Total_Cost=("Cost (£)","sum")).sort_values("Lost", ascending=False)
    grouped["Total_Cost"] = grouped["Total_Cost"].apply(fmt_cost)
    grouped = grouped.rename(columns={"Total_Cost":"Cost Lost"}).reset_index()
    grouped.index = range(1,len(grouped)+1); return grouped
def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    if len(data) == 0: return plt.subplots(figsize=(figsize_width, 2))[0]
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
    if not sizes: fig,ax = plt.subplots(figsize=(2,1.5)); ax.text(0.5,0.5,"No data",ha="center"); return fig
    fig,ax = plt.subplots(figsize=(2,1.5))
    ax.pie(sizes,labels=labels,colors=colors,explode=explode,autopct="%1.0f%%",startangle=90,textprops={"fontsize":5})
    ax.set_title(title,fontsize=6); plt.tight_layout(); return fig

# ─── HEALTH SCORE ─────────────────────────────────────────────────────────────
def render_health_score(df, total):
    if total == 0: return 5, "🟡", "No data", []
    score = 10; reasons = []
    cl_c = df["Cluster"].dropna().value_counts()
    if len(cl_c) >= 2:
        vals = cl_c.values.astype(float); n = len(vals); sv = np.sort(vals)
        denom = n * np.sum(sv)
        if denom > 0:
            gini = (2 * np.sum(np.arange(1, n+1) * sv) - (n+1) * np.sum(sv)) / denom
            if gini > 0.6: score -= 3; reasons.append("Losses very concentrated")
            elif gini > 0.4: score -= 2; reasons.append("Losses somewhat concentrated")
            elif gini > 0.3: score -= 1; reasons.append("Slight concentration")
    shift_counts = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
    assigned = shift_counts.sum()
    if assigned >= 20:
        observed = np.array([shift_counts[s] for s in SHIFT_ORDER])
        expected = np.array([assigned/4]*4)
        try:
            _, p = sp_stats.chisquare(observed, f_exp=expected)
            if p < 0.01: score -= 2; reasons.append("Shift imbalance severe")
            elif p < 0.05: score -= 1; reasons.append("Shift imbalance present")
        except Exception:
            pass
    otr = df[df["Type"]=="OTR"]
    if len(otr) >= 5:
        dc = otr["DSP Name"].dropna().value_counts()
        if len(dc) >= 3:
            mu = dc.mean(); sigma = dc.std()
            if sigma > 0:
                outliers_n = ((dc - mu) / sigma > 1.5).sum()
                if outliers_n >= 2: score -= 2; reasons.append(f"{outliers_n} DSP outliers")
                elif outliers_n == 1: score -= 1; reasons.append("1 DSP outlier")
    score = max(1, min(10, score))
    if score >= 8: color = "🟢"; label = "Good"
    elif score >= 5: color = "🟡"; label = "Needs attention"
    else: color = "🔴"; label = "Action required"
    return score, color, label, reasons

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TAB
# ═══════════════════════════════════════════════════════════════════════════════
def render_summary_tab(df, total, dr, kp=""):
    if total == 0: st.warning("No data to display."); return

    # OTR vs UTR
    with st.expander("🥧 OTR vs UTR"):
        st.pyplot(make_pie_otr_utr(df, total, f"OTR vs UTR ({dr})"))

    # ─── CLUSTERS + LOCATION DRILL-DOWN (combined) ────────────────────────────
    with st.expander("📍 Clusters & Location Drill-Down", expanded=True):
        cc = df["Cluster"].dropna().value_counts()
        if len(cc) > 0:
            # View selector
            cluster_view = st.selectbox("View:", ["All Clusters", "Location Drill-Down"], key=f"{kp}cl_view")
            if cluster_view == "All Clusters":
                # Combined bar chart
                st.pyplot(make_bar_horiz(cc, f"All Clusters ({dr})"))
                # Shift breakdown below
                st.markdown("**By Shift per Cluster (top 10):**")
                top_clusters = cc.head(10).index.tolist()
                shift_by_cluster = df[df["Cluster"].isin(top_clusters)].groupby(["Cluster","Shift"]).size().unstack(fill_value=0)
                shift_by_cluster = shift_by_cluster.reindex(columns=SHIFT_ORDER, fill_value=0)
                # Reorder rows by loss count
                shift_by_cluster = shift_by_cluster.reindex(top_clusters)
                st.dataframe(shift_by_cluster, width="stretch")
            else:
                # Location Drill-Down
                clusters = sorted(cc.index.tolist())
                sel_cluster = st.selectbox("Select Cluster:", clusters, key=f"{kp}loc_drill")
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
                # Tables
                col1, col2 = st.columns(2)
                with col1:
                    if len(ad) > 0: st.dataframe(make_table(ad, "Aisle", "Count"), width="stretch")
                with col2:
                    st.dataframe(make_table(sd, "Shift", "Count"), width="stretch")
        else:
            st.info("No cluster data available.")

    # ─── SHIFT (selectbox: Shift Windows or Hours Lost) ───────────────────────
    with st.expander("⏰ Shifts", expanded=True):
        shift_view = st.selectbox("Show:", ["Shift Windows", "Hours Lost"], key=f"{kp}shift_view")
        if shift_view == "Shift Windows":
            st.markdown("**Shift Windows (DRM2):**")
            st.caption("NS: 23:45–09:45 | AM: 09:45–14:00 | PM: 14:00–23:45 | OTR: On The Road")
            sc = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts()
            # Leaderboard — always shows all shifts even if 0
            rows = []
            for s in SHIFT_ORDER:
                sdf = df[df["Shift"]==s]; n = len(sdf)
                rows.append({"Shift":s,"Lost":n,"%":f"{round(n/total*100,1)}%","Cost":fmt_cost(sdf["Cost (£)"].sum()),"Window":SHIFT_DEFINITIONS[s]})
            rows.sort(key=lambda r:r["Lost"],reverse=True)
            st.dataframe(pd.DataFrame(rows,index=range(1,len(rows)+1)),width="stretch",height=200)
            if len(sc)>0: st.pyplot(make_bar_shift(sc,f"By Shift ({dr})"))
        else:
            # Hours Lost
            st.markdown("**⏰ Hour of Last Scan (loss timing proxy):**")
            st.caption("Uses previous_event_datetime — the last scan before loss. "
                       "Coloured by shift. Best proxy for WHEN losses occur.")
            if "Prev Event DT" in df.columns:
                hours = df["Prev Event DT"].dropna().dt.hour
                if len(hours) > 0:
                    hour_counts = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                    fig, ax = plt.subplots(figsize=(8, 2.5))
                    colors = [SHIFT_COLORS.get(SHIFT_HOUR_MAP.get(h, "Unknown"), "gray") for h in range(24)]
                    ax.bar(range(24), hour_counts.values, color=colors)
                    for h in range(24):
                        v = hour_counts.values[h]
                        if v > 0: ax.text(h, v+0.1, str(int(v)), ha="center", fontsize=6)
                    ax.set_xlabel("Hour of Day", fontsize=8); ax.set_ylabel("Parcels", fontsize=8)
                    ax.set_title("Last Scan Hour Before Loss", fontsize=9)
                    ax.set_xticks(range(24)); ax.tick_params(labelsize=7); plt.tight_layout()
                    st.pyplot(fig)
                    st.caption("🟦 NS (23:45–09:45) | 🟧 AM (09:45–14:00) | 🟩 PM (14:00–23:45)")
                    # Peak hour info
                    peak_hour = hour_counts.idxmax()
                    st.info(f"Peak hour: **{peak_hour}:00** ({int(hour_counts.max())} parcels) — {SHIFT_HOUR_MAP.get(peak_hour, 'Unknown')} shift")
                else:
                    st.warning("No valid previous_event_datetime data. PNOV parcels often have corrupt timestamps — "
                              "this is expected. Other sub-buckets should show data.")
            else:
                st.info("No Prev Event DT column found.")

    # ─── LOST SUB-BUCKET TITLE + COST ─────────────────────────────────────────
    with st.expander("🏷️ Lost Sub-Bucket Title + Cost"):
        sb2 = df["Sub Bucket"].dropna().value_counts()
        if len(sb2)>0:
            # Always show table with cost
            sb_cost = df.groupby("Sub Bucket").agg(
                Lost=("Tracking ID","count"),
                Cost=("Cost (£)","sum")
            ).sort_values("Lost", ascending=False).reset_index()
            sb_cost["Avg Cost"] = (sb_cost["Cost"] / sb_cost["Lost"]).apply(fmt_cost)
            sb_cost["% of Total"] = (sb_cost["Lost"] / total * 100).round(1)
            sb_cost["Cost"] = sb_cost["Cost"].apply(fmt_cost)
            sb_cost.index = range(1, len(sb_cost)+1)
            st.dataframe(sb_cost, width="stretch")
            # Chart
            st.pyplot(make_bar_horiz(sb2, f"Lost Sub-Buckets ({dr})", color="teal"))
        else:
            st.info("No sub-bucket data.")

    # ─── SIZE CATEGORY ────────────────────────────────────────────────────────
    with st.expander("📏 Lost Parcels by Size (Amazon UK FBA Tiers)"):
        sc_data = df["Size Category"].value_counts()
        if len(sc_data) > 0:
            size_order = ["Small Envelope", "Standard Envelope", "Standard Parcel", "Small Oversize", "Standard Oversize", "Large Oversize", "Unknown"]
            sc_ordered = sc_data.reindex([s for s in size_order if s in sc_data.index])
            if len(sc_ordered) > 0:
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
                st.caption("Longest side: Small Envelope ≤20cm | Std Envelope ≤33cm | Std Parcel ≤45cm | Small Oversize ≤61cm | Std Oversize ≤120cm | Large Oversize >120cm")
        else:
            st.info("No size data available.")


# ═══════════════════════════════════════════════════════════════════════════════
# LOCATIONS TAB
# ═══════════════════════════════════════════════════════════════════════════════
def render_locations_tab(df, total, dr, kp=""):
    if total == 0: st.warning("No data."); return
    lf = st.radio("Show:", ["All Parcels","OTR Only","UTR Only"], horizontal=True, key=f"{kp}lf")
    if lf == "OTR Only":
        vdf = df[df["Type"]=="OTR"].copy()
        st.write(f"**{len(vdf)} OTR** — {fmt_cost(vdf['Cost (£)'].sum())}")
        if len(vdf)==0: st.info("No OTR parcels."); return
        with st.expander("🚚 Top 10 DSPs", expanded=True):
            d = vdf["DSP Name"].dropna().value_counts().head(10)
            if len(d)>0: st.pyplot(make_bar_horiz(d,f"Top 10 DSPs — OTR ({dr})",color="firebrick",max_label=DSP_MAX))
        with st.expander("📍 Top 10 Delivery Areas"):
            ab = st.radio("By:",["City","Province","Postal"],horizontal=True,key=f"{kp}oa")
            if ab in vdf.columns:
                ad = vdf[ab].dropna().value_counts().head(10)
                if len(ad)>0: st.pyplot(make_bar_horiz(ad,f"Top 10 {ab} — OTR ({dr})",color="darkred"))
        with st.expander("❓ Top Loss Reasons"):
            r = vdf["Loss Reason"].dropna().value_counts().head(10)
            if len(r)>0: st.pyplot(make_bar_horiz(r,"Top OTR Reasons",color="crimson"))
    elif lf == "UTR Only":
        vdf = df[df["Type"]=="UTR"].copy()
        st.write(f"**{len(vdf)} UTR** — {fmt_cost(vdf['Cost (£)'].sum())}")
        if len(vdf)==0: st.info("No UTR parcels."); return
        with st.expander("📍 Top 10 Clusters", expanded=True):
            cl = vdf["Cluster"].dropna().value_counts().head(10)
            if len(cl)>0: st.pyplot(make_bar_horiz(cl,f"Top 10 Clusters — UTR ({dr})",color="darkorange"))
        with st.expander("🏷️ Top 10 Aisles"):
            al = vdf["Aisle"].dropna().value_counts().head(10)
            if len(al)>0: st.pyplot(make_bar_horiz(al,f"Top 10 Aisles — UTR ({dr})",color="orange"))
        with st.expander("🏷️ Sub Buckets"):
            sb = vdf["Sub Bucket"].dropna().value_counts()
            if len(sb)>0: st.pyplot(make_bar_horiz(sb,"UTR Sub Buckets",color="darkorange"))
    else:
        vdf = df.copy()
        st.write(f"**{len(vdf)} all** — {fmt_cost(vdf['Cost (£)'].sum())}")
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


# ═══════════════════════════════════════════════════════════════════════════════
# PNOV TAB
# ═══════════════════════════════════════════════════════════════════════════════
def render_pnov_tab(df, total, dr, kp=""):
    st.markdown("#### 📦 PNOV — Package Not On Vehicle")
    st.caption("Identifies DSPs/Flex drivers responsible + associates who stowed (for upskilling).")

    pnov_keywords = ["PNOV", "No Further Status", "Package Not On Vehicle"]
    pnov_mask = df["Sub Bucket"].fillna("").apply(lambda x: any(k.lower() in str(x).lower() for k in pnov_keywords))
    pnov_df = df[pnov_mask].copy()

    if len(pnov_df) == 0:
        st.info("No PNOV parcels found. PNOV = sub-bucket containing 'No Further Status' or 'PNOV'.")
        return

    st.markdown(f"**{len(pnov_df)} PNOV parcels** — {fmt_cost(pnov_df['Cost (£)'].sum())} ({round(len(pnov_df)/total*100,1)}% of all losses)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Flex Drivers", len(pnov_df[pnov_df["Driver Type"] == "Flex"]))
    c2.metric("DSP Drivers", len(pnov_df[pnov_df["Driver Type"] == "DSP"]))
    c3.metric("Avg Cost/Parcel", fmt_cost(pnov_df["Cost (£)"].mean()))

    with st.expander("🚚 Responsible Drivers (DSP + Flex)", expanded=True):
        dsp_data = pnov_df.dropna(subset=["DSP Name"])
        if len(dsp_data) > 0:
            dc = dsp_data["DSP Name"].value_counts()
            vm = st.radio("View:", ["Chart", "Table + Cost"], horizontal=True, key=f"{kp}pnov_dsp_v")
            if vm == "Chart":
                st.pyplot(make_bar_horiz(dc, f"PNOV by Driver ({dr})", color="firebrick", max_label=DSP_MAX))
            else:
                tbl = dsp_data.groupby("DSP Name").agg(
                    PNOV_Count=("Tracking ID", "count"), Cost=("Cost (£)", "sum"), Driver_Type=("Driver Type", "first")
                ).sort_values("PNOV_Count", ascending=False).reset_index()
                tbl["Cost"] = tbl["Cost"].apply(fmt_cost); tbl.index = range(1, len(tbl)+1)
                st.dataframe(tbl, width="stretch")
            # Flex vs DSP chart
            st.markdown("---")
            type_counts = dsp_data["Driver Type"].value_counts()
            if len(type_counts) > 0:
                fig, ax = plt.subplots(figsize=(3, 1.5))
                cmap = {"Flex": "#e74c3c", "DSP": "#3498db", "Unknown": "#95a5a6"}
                ax.bar(type_counts.index, type_counts.values, color=[cmap.get(x, "gray") for x in type_counts.index])
                for i, v in enumerate(type_counts.values): ax.text(i, v+0.2, str(int(v)), ha="center", fontsize=8)
                ax.set_title("PNOV: Flex vs DSP", fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("No driver data for PNOV parcels.")

    # ─── STOWED BY ASSOCIATES (upskilling) ────────────────────────────────────
    with st.expander("👤 Associates — Stowed By (Upskilling List)", expanded=True):
        st.caption("The associate who last stowed/scanned the parcel before it was lost. "
                   "Use this list for upskilling conversations.")
        has_stowed = pnov_df["Stowed By"].dropna()
        if len(has_stowed) > 0:
            assoc_counts = has_stowed.value_counts()
            st.markdown(f"**{len(assoc_counts)} associate(s)** involved:")
            assoc_tbl = pnov_df[pnov_df["Stowed By"].notna()].groupby("Stowed By").agg(
                Parcels=("Tracking ID", "count"),
                Cost=("Cost (£)", "sum"),
                Clusters=("Cluster", lambda x: ", ".join(sorted(x.dropna().unique()[:3])))
            ).sort_values("Parcels", ascending=False).reset_index()
            assoc_tbl["Cost"] = assoc_tbl["Cost"].apply(fmt_cost)
            assoc_tbl.index = range(1, len(assoc_tbl)+1)
            st.dataframe(assoc_tbl, width="stretch")
            st.download_button("⬇️ Download upskilling list", assoc_tbl.to_csv(index=False), "PNOV_upskilling.csv", "text/csv", key=f"{kp}dl_upskill")
        else:
            # Show which columns WERE available (debugging help)
            scc_cols = [c for c in df.columns if c not in ["Tracking ID","Sub Bucket","Bucket","Type","Shift","Loss Reason","UTR Reason","Driver Type","Cost (£)","Prev Event DT","Marked Lost DT","Longest Side","Size Category"]]
            st.warning("No 'Stowed By' data found in this SCC export.")
            st.info("**How to get this:** Ensure SCC export includes a column like 'Stowed By', 'Last Scan By', or 'Scanned By'.")
            with st.expander("🔧 Available SCC columns (for debugging)"):
                st.write("These columns were found in your SCC file:")
                st.code(", ".join(sorted(scc_cols)))

    with st.expander("📍 PNOV by Cluster"):
        cl = pnov_df["Cluster"].dropna().value_counts()
        if len(cl) > 0: st.pyplot(make_bar_horiz(cl.head(15), f"PNOV by Cluster ({dr})", color="purple"))

    with st.expander("📋 All PNOV Tracking IDs"):
        tid_cols = [c for c in ["Tracking ID","Cluster","Aisle","DSP Name","Driver Type","Stowed By","Cost (£)","Loss Reason"] if c in pnov_df.columns]
        tid_df = pnov_df[tid_cols].reset_index(drop=True)
        tid_df.index = range(1, len(tid_df)+1)
        st.dataframe(tid_df, width="stretch", height=400)
        st.download_button("⬇️ Download PNOV", tid_df.to_csv(index=False), "PNOV_parcels.csv", "text/csv", key=f"{kp}dl_pnov")


# ═══════════════════════════════════════════════════════════════════════════════
# COST BREAKDOWN TAB
# ═══════════════════════════════════════════════════════════════════════════════
def render_cost_tab(df, total, dr, kp=""):
    if total == 0: st.warning("No data."); return
    tc = df["Cost (£)"].sum(); avg = tc/total if total>0 else 0
    otr_df = df[df["Type"]=="OTR"]; utr_df = df[df["Type"]=="UTR"]
    st.markdown(f"### 💰 Cost Breakdown")
    # Key metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cost", fmt_cost(tc))
    c2.metric("Avg/Parcel", fmt_cost(avg))
    c3.metric("OTR Cost", fmt_cost(otr_df["Cost (£)"].sum()))
    c4.metric("UTR Cost", fmt_cost(utr_df["Cost (£)"].sum()))
    st.caption(f"{total} parcels | OTR: {len(otr_df)} ({fmt_cost(otr_df['Cost (£)'].sum())}) | UTR: {len(utr_df)} ({fmt_cost(utr_df['Cost (£)'].sum())})")

    with st.expander("💰 Cost by Sub Bucket", expanded=True):
        csb = df.groupby(["Sub Bucket","Type"]).agg(Count=("Tracking ID","count"),Cost=("Cost (£)","sum")).sort_values("Cost",ascending=False).reset_index()
        csb["Avg/Parcel"] = (csb["Cost"]/csb["Count"]).apply(fmt_cost)
        csb["Cost"] = csb["Cost"].apply(fmt_cost)
        csb.index = range(1,len(csb)+1)
        st.dataframe(csb,width="stretch")

    with st.expander("💰 Cost by Shift"):
        shift_cost = df.groupby("Shift").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).reindex(SHIFT_ORDER, fill_value=0).reset_index()
        shift_cost["Avg/Parcel"] = shift_cost.apply(lambda r: fmt_cost(r["Cost"]/r["Count"]) if r["Count"]>0 else "£0.00", axis=1)
        shift_cost["Cost"] = shift_cost["Cost"].apply(fmt_cost)
        shift_cost.index = range(1, len(shift_cost)+1)
        st.dataframe(shift_cost, width="stretch")

    with st.expander("💰 Cost by DSP"):
        dsp_df = df.dropna(subset=["DSP Name"])
        if len(dsp_df)>0:
            dc = dsp_df.groupby("DSP Name").agg(Count=("Tracking ID","count"),Cost=("Cost (£)","sum")).sort_values("Cost",ascending=False).reset_index()
            reasons = []
            for dsp in dc["DSP Name"]:
                dp = dsp_df[dsp_df["DSP Name"]==dsp]; rc = dp["Loss Reason"].dropna().value_counts()
                reasons.append(rc.index[0] if len(rc)>0 else "N/A")
            dc["Top Reason"] = reasons
            dc["Avg/Parcel"] = (dc["Cost"]/dc["Count"]).apply(fmt_cost)
            dc["Cost"] = dc["Cost"].apply(fmt_cost)
            dc.index = range(1,len(dc)+1)
            st.dataframe(dc,width="stretch")

    with st.expander("💰 Cost by Size Tier"):
        sz_cost = df.groupby("Size Category").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
        sz_cost["Avg/Parcel"] = (sz_cost["Cost"]/sz_cost["Count"]).apply(fmt_cost)
        sz_cost["% of Cost"] = (sz_cost["Cost"] / tc * 100).round(1) if tc > 0 else 0
        sz_cost["Cost"] = sz_cost["Cost"].apply(fmt_cost)
        sz_cost.index = range(1, len(sz_cost)+1)
        st.dataframe(sz_cost, width="stretch")

    with st.expander("💰 Cost by Cluster (Top 10)"):
        cl_cost = df.dropna(subset=["Cluster"]).groupby("Cluster").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).head(10).reset_index()
        cl_cost["Avg/Parcel"] = (cl_cost["Cost"]/cl_cost["Count"]).apply(fmt_cost)
        cl_cost["Cost"] = cl_cost["Cost"].apply(fmt_cost)
        cl_cost.index = range(1, len(cl_cost)+1)
        st.dataframe(cl_cost, width="stretch")

    with st.expander("💰 Top 10 Most Expensive Parcels"):
        top = df.nlargest(10,"Cost (£)")
        sc2 = [c for c in ["Tracking ID","Sub Bucket","Type","Shift","Cost (£)","DSP Name","Cluster","Loss Reason","Size Category"] if c in top.columns]
        out = top[sc2].reset_index(drop=True); out.index = range(1,len(out)+1)
        st.dataframe(out,width="stretch")

    with st.expander("💰 Flex vs DSP Cost"):
        flex_df = df[df["Driver Type"]=="Flex"]
        dsp_only = df[df["Driver Type"]=="DSP"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Flex", f"{len(flex_df)} parcels — {fmt_cost(flex_df['Cost (£)'].sum())}")
        with col2:
            st.metric("DSP", f"{len(dsp_only)} parcels — {fmt_cost(dsp_only['Cost (£)'].sum())}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS & TREND TAB (combined)
# ═══════════════════════════════════════════════════════════════════════════════
def render_analysis_trend_tab(df, total, dr, kp=""):
    if total == 0: st.warning("No data."); return
    view = st.selectbox("View:", ["🔬 Analysis", "📈 Trend"], key=f"{kp}at_view")

    if view == "🔬 Analysis":
        st.markdown("### 🔬 Analysis")
        st.warning("⚠️ An AID only — use your own judgement and local knowledge.")
        findings = []

        with st.expander("📊 Loss Reasons", expanded=True):
            lr = df.groupby("Loss Reason").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
            if len(lr) > 0:
                lr["% of Total"] = (lr["Count"] / total * 100).round(1).astype(str) + "%"
                lr["Avg £"] = (lr["Cost"] / lr["Count"]).apply(fmt_cost)
                lr["Cost"] = lr["Cost"].apply(fmt_cost)
                lr.index = range(1, len(lr)+1)
                st.dataframe(lr, width="stretch")

        with st.expander("📍 Cluster Concentration"):
            cl_c = df["Cluster"].dropna().value_counts()
            if len(cl_c) >= 2:
                top3 = cl_c.head(3)
                top3_pct = round(top3.sum() / cl_c.sum() * 100, 1)
                top3_cost = df[df["Cluster"].isin(top3.index)]["Cost (£)"].sum()
                vals = cl_c.values.astype(float); n = len(vals); sv = np.sort(vals)
                denom = n * np.sum(sv)
                if denom > 0:
                    gini = (2 * np.sum(np.arange(1, n+1) * sv) - (n+1) * np.sum(sv)) / denom
                    if gini > 0.5: st.error(f"🎯 Concentrated. Top 3 = **{top3_pct}%** ({fmt_cost(top3_cost)})")
                    elif gini > 0.3: st.warning(f"⚠️ Some concentration. Top 3 = **{top3_pct}%**")
                    else: st.success(f"✅ Spread out. Top 3 = {top3_pct}%")
                    findings.append(f"Top 3 clusters = {top3_pct}% ({fmt_cost(top3_cost)})")

        with st.expander("🚚 DSP Outliers"):
            otr = df[df["Type"] == "OTR"]
            if len(otr) >= 5:
                dc = otr["DSP Name"].dropna().value_counts()
                if len(dc) >= 3:
                    mu = dc.mean(); sigma = dc.std()
                    if sigma > 0:
                        z = (dc - mu) / sigma
                        outliers = z[z > 1.5]
                        tbl = pd.DataFrame({"DSP": dc.index, "Losses": dc.values, "Status": ["🎯 Above avg" if zv > 1.5 else "✅ Normal" for zv in z.values]})
                        tbl.index = range(1, len(tbl)+1); st.dataframe(tbl, width="stretch")
                        if len(outliers) > 0:
                            st.error(f"🎯 {len(outliers)} DSP(s) above average.")
                    else:
                        st.info("Not enough variation between DSPs.")
                else:
                    st.info("Need 3+ DSPs to compare.")
            else:
                st.info(f"Need 5+ OTR parcels (have {len(otr)}).")

        with st.expander("🔁 Repeat Offender Aisles"):
            aisle_c = df["Aisle"].dropna().value_counts()
            repeat = aisle_c[aisle_c >= 3]
            if len(repeat) > 0:
                st.error(f"🎯 {len(repeat)} aisle(s) lost 3+ parcels:")
                st.dataframe(pd.DataFrame({"Aisle": repeat.index, "Losses": repeat.values}, index=range(1, len(repeat)+1)), width="stretch")
            else:
                st.success("✅ No single aisle has 3+ losses.")

        if findings:
            st.markdown("---")
            st.markdown("#### 💡 Key findings")
            for i, f in enumerate(findings, 1): st.markdown(f"**{i}.** {f}")

    else:
        st.markdown("### 📈 Week-over-Week Trend")
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
            if last_w < first_w * 0.8: st.success(f"📉 **Improving!** {first_w} → {last_w} ({pct_change:+.1f}%)")
            elif last_w > first_w * 1.2: st.error(f"📈 **Getting worse.** {first_w} → {last_w} ({pct_change:+.1f}%)")
            else: st.info(f"➡️ **Stable.** {first_w} → {last_w} ({pct_change:+.1f}%)")
            st.download_button("⬇️ Download trend", weekly.to_csv(index=False), "weekly_trend.csv", "text/csv", key=f"{kp}dl_trend")
        elif len(weeks_data) == 1:
            st.info("Enter 1 more week to compare.")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT TAB
# ═══════════════════════════════════════════════════════════════════════════════
def render_export_tab(df, total, dr, kp="", station_name=""):
    st.markdown("#### 💾 Export")
    import zipfile
    output = BytesIO()
    exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","shipment_value"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
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
    ec = [c for c in df.columns if c not in exc]
    fname2 = f"{station_name}_Clean.csv" if station_name else "Lost_Clean.csv"
    st.download_button(f"⬇️ Single CSV", df[ec].to_csv(index=False), fname2, "text/csv", key=f"{kp}dl_clean")


# ═══════════════════════════════════════════════════════════════════════════════
# GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
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
| 📊 **Summary** | Clusters, shifts (or hours), sub-buckets + cost, size |
| 📍 **Locations** | Top 10s — clusters, aisles, DSPs, sort zones |
| 📦 **PNOV** | Drivers + associates for upskilling |
| 💰 **Cost Breakdown** | Cost by sub-bucket, shift, DSP, size, cluster |
| 🔬 **Analysis & Trend** | Pattern detection + week-over-week |
| 💾 **Export** | Download processed data |
""")
    with st.expander("📏 Size Classification (Amazon UK FBA Tiers)"):
        st.markdown(SIZE_TIER_INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# MISSING PARCELS
# ═══════════════════════════════════════════════════════════════════════════════
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
        try:
            pm_df = pd.read_csv(pm_file)
            scc_df = pd.read_csv(scc_file)
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}"); st.stop()
        pm_miss = [c for c in REQUIRED_PM_COLS if c not in pm_df.columns]
        if pm_miss: st.error(f"❌ PM missing columns: {pm_miss}"); st.stop()
        scc_miss = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
        if scc_miss: st.error(f"❌ SCC missing columns: {scc_miss}"); st.stop()
        found = [c for c in SENSITIVE_COLS if c in scc_df.columns]
        if found: st.warning(f"🔒 PII removed: {', '.join(found)}")
        df = merge_data(pm_df, scc_df); total = len(df)
        if total == 0: st.warning("No parcels after merging."); st.stop()
        matched = df["Cluster"].notna().sum(); tc = df["Cost (£)"].sum()
        st.success(f"✅ **{total} parcels** — {fmt_cost(tc)} (PM:{len(pm_df)}, SCC:{len(scc_df)}, Matched:{matched})")
        render_missing_parcels(df, total, matched)
        dr = get_date_range(df)
        score, color, label, score_reasons = render_health_score(df, total)
        st.markdown(f"**Health Score: {color} {score}/10 — {label}**" + (f" ({', '.join(score_reasons)})" if score_reasons else ""))
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Lost", total); c2.metric("Cost", fmt_cost(tc)); c3.metric("Cluster", safe_top(df["Cluster"]))
        c4.metric("Aisle", safe_top(df["Aisle"])); c5.metric("DSP", str(safe_top(df["DSP Name"]))[:15])
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]; c6.metric("Shift", safe_top(sk) if len(sk)>0 else "N/A")
        # 6 TABS: Summary, Locations, PNOV, Cost Breakdown, Analysis & Trend, Export
        t1,t2,t3,t4,t5,t6 = st.tabs(["📊 Summary","📍 Locations","📦 PNOV","💰 Cost Breakdown","🔬 Analysis & Trend","💾 Export"])
        with t1: render_summary_tab(df, total, dr, kp="s_")
        with t2: render_locations_tab(df, total, dr, kp="s_")
        with t3: render_pnov_tab(df, total, dr, kp="s_")
        with t4: render_cost_tab(df, total, dr, kp="s_")
        with t5: render_analysis_trend_tab(df, total, dr, kp="s_")
        with t6: render_export_tab(df, total, dr, kp="s_")
    else:
        st.info("👆 Upload both files.")

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
            try:
                pt, s2 = pd.read_csv(pf), pd.read_csv(sf)
                m = merge_data(pt, s2)
                nm = nm_input.strip() if nm_input and nm_input.strip() else f"Dataset {i+1}"
                stations[nm] = m; names.append(nm)
            except Exception as e:
                st.error(f"Error with dataset {i+1}: {e}")
        if len(stations) >= 2:
            st.success(f"✅ {', '.join(names)}")
            for n in names:
                sc_val, sc_col, sc_lab, sc_reas = render_health_score(stations[n], len(stations[n]))
                st.caption(f"{n}: {sc_col} {sc_val}/10 — {sc_lab}" + (f" ({', '.join(sc_reas)})" if sc_reas else ""))
            t1,t2,t3,t4,t5,t6 = st.tabs(["📊 Summary","📍 Locations","📦 PNOV","💰 Cost Breakdown","🔬 Analysis & Trend","💾 Export"])
            with t1:
                for n in names:
                    sdf = stations[n]
                    st.markdown(f"**{n}:** {len(sdf)} parcels — {fmt_cost(sdf['Cost (£)'].sum())} | Top cluster: {safe_top(sdf['Cluster'])}")
            with t2:
                sel = st.selectbox("Dataset:", names, key="mcl"); render_locations_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"m{sel}_")
            with t3:
                sel = st.selectbox("Dataset:", names, key="mcp"); render_pnov_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mp{sel}_")
            with t4:
                sel = st.selectbox("Dataset:", names, key="mcc"); render_cost_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc{sel}_")
            with t5:
                sel = st.selectbox("Dataset:", names, key="mca"); render_analysis_trend_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"ma{sel}_")
            with t6:
                sel = st.selectbox("Dataset:", names, key="mce"); render_export_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"me{sel}_", station_name=sel)
    elif len(uploaded) == 1: st.warning("Need 2+ datasets to compare.")
    else: st.info("👆 Upload pairs of PM + SCC files above.")
