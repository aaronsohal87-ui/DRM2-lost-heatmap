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
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SHIFT_ORDER = ["NS", "AM", "PM", "OTR"]
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen", "OTR": "firebrick"}
SHIFT_DEFINITIONS = {"NS": "23:45 – 09:45 (Night Sort — stow)", "AM": "09:45 – 14:00 (Pick, stage, dispatch)", "PM": "14:00 – 23:45 (Dispatch, RELO)", "OTR": "On The Road (DSP responsibility)"}

# Updated shift hour map for new timings: NS=23:45-9:45, AM=9:45-14:00, PM=14:00-23:45
# Hour 0-9 = NS (continuing from previous night), Hour 10-13 = AM, Hour 14-23 = PM
# Note: The 9:45 boundary means hour 9 is mostly NS, hour 10 starts AM territory
# Hour 23 is the start of NS (23:45 onward)
SHIFT_HOUR_MAP = {0:"NS",1:"NS",2:"NS",3:"NS",4:"NS",5:"NS",6:"NS",7:"NS",8:"NS",9:"NS",10:"AM",11:"AM",12:"AM",13:"AM",14:"PM",15:"PM",16:"PM",17:"PM",18:"PM",19:"PM",20:"PM",21:"PM",22:"PM",23:"NS"}

SUB_BUCKET_SHIFT_MAP = {"Lost At Station - Inducted Not Stowed":"NS","Lost At Station - Stowed Not Picked Up":"AM","Lost At Station - Debrief Receive(RTS)":"PM","Lost On Road - Attempted":"OTR","Lost On Road - Damage":"OTR","Lost On Road - No Further Status":"OTR"}
SENSITIVE_COLS = ["Holder Name","Ordering Order ID","Order Amount","Receivable Amount","Payment Method","District","Scheduled Delivery End Time"]
REQUIRED_SCC_COLS = ["Tracking ID","Sort Zone","Aisle","Cluster","Package Length","Package Width","Package Height","DSP Name","Assigned Cycle","Last Updated Time"]
REQUIRED_PM_COLS = ["tracking_id","sub_bucket"]
CHART = (7, 2.5)
DSP_MAX = 20
LABEL_MAX = 25

# Amazon UK FBA Size Tiers (based on longest side of packaged item)
# Small Envelope: ≤ 20cm longest side
# Standard Envelope / Large Envelope: ≤ 33cm longest side
# Standard Parcel: ≤ 45cm longest side
# Small Oversize: ≤ 61cm longest side
# Standard Oversize: ≤ 120cm longest side
# Large Oversize: > 120cm longest side
SIZE_TIER_INFO = """
| Size Tier | Longest Side | Amazon Classification |
|-----------|-------------|----------------------|
| Small Envelope | ≤ 20 cm | Thin, flat items (phone cases, etc.) |
| Standard Envelope | ≤ 33 cm | Books, small items |
| Standard Parcel | ≤ 45 cm | Most consumer goods |
| Small Oversize | ≤ 61 cm | Larger boxed items |
| Standard Oversize | ≤ 120 cm | Furniture, large appliances |
| Large Oversize | > 120 cm | Very large items |
"""

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────
def get_size(val):
    """Amazon UK FBA size tier based on longest side (cm)."""
    if pd.isna(val): return "Unknown"
    if val <= 20: return "Small Envelope"
    if val <= 33: return "Standard Envelope"
    if val <= 45: return "Standard Parcel"
    if val <= 61: return "Small Oversize"
    if val <= 120: return "Standard Oversize"
    return "Large Oversize"

def hour_to_shift(hour):
    """Convert hour to shift using updated DRM2 shift windows."""
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
    """Identify Flex drivers — CSP_COMPANY_NAME indicates a Flex driver."""
    if pd.isna(dsp_name): return False
    return "CSP_COMPANY_NAME" in str(dsp_name).upper()

def clean_dsp_name(dsp_name):
    """Clean DSP name — mark Flex drivers clearly."""
    if pd.isna(dsp_name): return dsp_name
    if is_flex_driver(dsp_name):
        return "FLEX DRIVER"
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
    # Clean DSP names — identify Flex drivers
    if "DSP Name" in df.columns:
        df["DSP Name"] = df["DSP Name"].apply(clean_dsp_name)
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
    if "Marked Lost DT" in merged.columns: merged["Day of Week"] = merged["Marked Lost DT"].dt.day_name()
    elif "Dispatch Time" in merged.columns: merged["Day of Week"] = merged["Dispatch Time"].dt.day_name()
    else: merged["Day of Week"] = None
    if "previous_reason" in merged.columns: merged["Loss Reason"] = merged["previous_reason"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else: merged["Loss Reason"] = "Unknown"
    if "previous_reason_3" in merged.columns: merged["UTR Reason"] = merged["previous_reason_3"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else: merged["UTR Reason"] = "Unknown"
    merged["Shift"] = merged.apply(assign_shift, axis=1)
    # Identify Flex vs DSP
    merged["Driver Type"] = merged["DSP Name"].apply(lambda x: "Flex" if x == "FLEX DRIVER" else ("DSP" if pd.notna(x) else "Unknown"))
    for col in ["Cluster","Aisle","Sort Zone","DSP Name","Size Category","City","Province","Postal","Cost (£)"]:
        if col not in merged.columns: merged[col] = None
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
def verify_totals(df, total, label=""):
    if len(df)!=total: st.error(f"⚠️ MISMATCH {label}: Expected {total}, got {len(df)}."); return False
    return True
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
def make_bar_vert(data, xl, yl, title, color="steelblue", figsize=CHART):
    fig,ax = plt.subplots(figsize=figsize); labs = trunc(data.index,LABEL_MAX); ax.bar(labs,data.values,color=color)
    for i,v in enumerate(data.values): ax.text(i,v+0.2,str(int(v)),ha="center",fontsize=7)
    ax.set_xlabel(xl,fontsize=8); ax.set_ylabel(yl,fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); return fig
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

# ─── TAB RENDERERS ────────────────────────────────────────────────────────────
def render_missing_parcels(df, total, matched):
    mc = total - matched
    if mc > 0:
        st.info(f"ℹ️ **{mc} parcel(s)** in PM had no SCC match — included in analysis but no location data.")
        with st.expander(f"🔍 Why? + View {mc} unmatched"):
            st.markdown("""
**Why some parcels don't match:**
- **Never inducted** — lost before arriving at station (linehaul, sort centre)
- **No scan** — was in station but never got a stow/container scan in SCC
- **ID mismatch** — tracking ID format differs slightly between PM and SCC
- **Virtual loss** — marked lost due to system timeout, never physically at this station
""")
            mdf = df[df["Cluster"].isna()].copy()
            if len(mdf)>0:
                sel = st.selectbox("Parcel:", mdf["Tracking ID"].tolist(), key="miss")
                r = mdf[mdf["Tracking ID"]==sel].iloc[0]
                st.markdown(f"**TID:** {sel} | **Sub Bucket:** {r.get('Sub Bucket','N/A')} | **Type:** {r.get('Type','N/A')} | **Shift:** {r.get('Shift','N/A')} | **Cost:** {fmt_cost(r.get('Cost (£)'))}")


def render_locations_tab(df, total, dr, kp=""):
    verify_totals(df, total, "Locations")
    lf = st.radio("Show:", ["All Parcels","OTR Only","UTR Only"], horizontal=True, key=f"{kp}lf")
    if lf == "OTR Only":
        vdf = df[df["Type"]=="OTR"].copy(); st.write(f"**{len(vdf)} OTR** — {fmt_cost(vdf['Cost (£)'].sum())}")
        if len(vdf)==0: return
        with st.expander("🚚 By DSP"):
            d = vdf["DSP Name"].dropna().value_counts()
            if len(d)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}od")
                if vm=="Chart": st.pyplot(make_bar_horiz(d,f"OTR by DSP ({dr})",color="firebrick",max_label=DSP_MAX))
                else: st.dataframe(make_cost_table(vdf.dropna(subset=["DSP Name"]),"DSP Name"),width="stretch")
        with st.expander("📍 Delivery Areas"):
            ab = st.radio("By:",["City","Province","Postal"],horizontal=True,key=f"{kp}oa")
            ad = vdf[ab].dropna().value_counts()
            if len(ad)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}oav")
                if vm=="Chart": st.pyplot(make_bar_horiz(ad.head(15),f"OTR by {ab}",color="darkred"))
                else: st.dataframe(make_cost_table(vdf.dropna(subset=[ab]),ab),width="stretch")
        with st.expander("❓ Reasons"):
            r = vdf["Loss Reason"].dropna().value_counts()
            if len(r)>0:
                vm = st.radio("View:",["Chart","Table"],horizontal=True,key=f"{kp}or")
                if vm=="Chart": st.pyplot(make_bar_horiz(r,"OTR Reasons",color="crimson"))
                else: st.dataframe(make_table(r,"Reason","Count"),width="stretch")
    elif lf == "UTR Only":
        vdf = df[df["Type"]=="UTR"].copy(); st.write(f"**{len(vdf)} UTR** — {fmt_cost(vdf['Cost (£)'].sum())}")
        if len(vdf)==0: return
        with st.expander("🏷️ Sub Buckets"):
            sb = vdf["Sub Bucket"].value_counts()
            if len(sb)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}usb")
                if vm=="Chart": st.pyplot(make_bar_horiz(sb,"UTR Sub Buckets",color="darkorange"))
                else: st.dataframe(make_cost_table(vdf,"Sub Bucket"),width="stretch")
        with st.expander("📍 By Cluster"):
            cl = vdf["Cluster"].dropna().value_counts()
            if len(cl)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}ul")
                if vm=="Chart": st.pyplot(make_bar_horiz(cl,"UTR Clusters",color="darkorange"))
                else: st.dataframe(make_cost_table(vdf.dropna(subset=["Cluster"]),"Cluster"),width="stretch")
    else:
        vdf = df.copy(); st.write(f"**{len(vdf)} all** — {fmt_cost(vdf['Cost (£)'].sum())}")
        with st.expander("🏆 Top 10 Locations"):
            rb = st.selectbox("By:",["Cluster","Aisle","Sort Zone"],key=f"{kp}rb")
            rd = vdf[rb].dropna().value_counts().head(10)
            if len(rd)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}lv")
                if vm=="Chart": st.pyplot(make_bar_horiz(rd,f"Top 10 {rb}s ({dr})",color="darkred"))
                else: st.dataframe(make_cost_table(vdf.dropna(subset=[rb]),rb).head(10),width="stretch")
        with st.expander("🔍 Cluster Drill-Down"):
            clusters = sorted(vdf["Cluster"].dropna().unique())
            if clusters:
                sel = st.selectbox("Cluster:",clusters,key=f"{kp}cl"); filt = vdf[vdf["Cluster"]==sel]
                st.write(f"**{len(filt)}** in {sel} — {fmt_cost(filt['Cost (£)'].sum())}")
                ad = filt["Aisle"].dropna().value_counts()
                if len(ad)>0:
                    vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}cv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(ad,f"{sel} Aisles",color="steelblue"))
                    else: st.dataframe(make_cost_table(filt.dropna(subset=["Aisle"]),"Aisle"),width="stretch")


def render_opportunities_tab(df, total, dr, kp=""):
    with st.expander("📖 How shift & lost date are assigned"):
        st.markdown("""
**Shift Windows (DRM2):**

| Shift | Window | Responsibility |
|-------|--------|----------------|
| NS | 23:45 – 09:45 | Night Sort — stow |
| AM | 09:45 – 14:00 | Pick, stage, dispatch |
| PM | 14:00 – 23:45 | Dispatch, RELO |
| OTR | N/A | On The Road (DSP/Flex) |

**How is the shift assigned?**

| If the parcel has... | Shift = |
|---------------------|---------|
| Sub bucket = "Inducted Not Stowed" | NS (Night Sort) |
| Sub bucket = "Stowed Not Picked Up" | AM |
| Sub bucket = "Debrief Receive(RTS)" | PM |
| Sub bucket = "Lost On Road - *" | OTR |
| None of the above | Time of last scan (hour-based) |

**How is the "lost date" determined?**

The date shown is when **Perfect Mile marked the parcel as lost** (`event_datetime`). This is the timestamp when the system flagged it — NOT necessarily when it physically went missing.
""")
    sc = df[df["Shift"]!="Unknown"]["Shift"].value_counts(); unk = len(df[df["Shift"]=="Unknown"])
    with st.expander("🏆 Shift Opportunity Leaderboard"):
        rows = []
        for s in SHIFT_ORDER:
            sdf = df[df["Shift"]==s]; n = len(sdf)
            rows.append({"Shift":s,"Lost":n,"%":f"{round(n/total*100,1)}%","Cost":fmt_cost(sdf["Cost (£)"].sum()),"Window":SHIFT_DEFINITIONS[s]})
        rows.sort(key=lambda r:r["Lost"],reverse=True)
        st.dataframe(pd.DataFrame(rows,index=range(1,len(rows)+1)),width="stretch",height=200)
        if len(sc)>0: st.pyplot(make_bar_shift(sc,f"By Shift ({dr})"))
    with st.expander("🔍 Shift Drill-Down"):
        opts = [f"{s} — {int(sc.get(s,0))}" for s in SHIFT_ORDER]
        sel = st.selectbox("Shift:",opts,key=f"{kp}os"); ss = sel.split(" —")[0]
        sdf = df[df["Shift"]==ss]
        if len(sdf)>0:
            st.write(f"**{ss}: {len(sdf)} parcels** — {fmt_cost(sdf['Cost (£)'].sum())}")
            reasons_available = ["All"] + sorted(sdf["Loss Reason"].dropna().unique().tolist())
            reason_filter = st.selectbox("Filter by reason:", reasons_available, key=f"{kp}rf")
            if reason_filter != "All":
                sdf = sdf[sdf["Loss Reason"] == reason_filter]
                st.caption(f"Showing {len(sdf)} parcels with reason: {reason_filter}")
            sbc = sdf["Sub Bucket"].value_counts()
            vm = st.radio("View:",["Sub Buckets","All Tracking IDs"],horizontal=True,key=f"{kp}sd")
            if vm=="Sub Buckets":
                st.dataframe(make_table(sbc,"Sub Bucket","Count"),width="stretch")
            else:
                tid_cols = [c for c in ["Tracking ID","Marked Lost DT","Sub Bucket","Type","Cluster","Aisle","DSP Name","Cost (£)","Loss Reason"] if c in sdf.columns]
                tid_df = sdf[tid_cols].copy()
                if "Marked Lost DT" in tid_df.columns:
                    tid_df = tid_df.sort_values("Marked Lost DT", ascending=True, na_position="last")
                tid_df = tid_df.reset_index(drop=True)
                tid_df.index = range(1, len(tid_df)+1)
                st.dataframe(tid_df, width="stretch", height=400)
                st.download_button(f"⬇️ Download {ss} parcels", tid_df.to_csv(index=False), f"{ss}_parcels.csv", "text/csv", key=f"{kp}dl_{ss}")


def render_pnov_tab(df, total, dr, kp=""):
    """PNOV (Package Not On Vehicle) tab — shows PNOV parcels and responsible associates/DSPs."""
    st.markdown("#### 📦 PNOV — Package Not On Vehicle")
    st.caption("PNOV parcels were dispatched but never confirmed delivered or returned. "
               "This tab identifies which DSPs/Flex drivers and clusters are associated with PNOV losses.")

    # Filter to PNOV sub-bucket
    pnov_keywords = ["PNOV", "No Further Status", "Package Not On Vehicle"]
    pnov_mask = df["Sub Bucket"].fillna("").apply(lambda x: any(k.lower() in x.lower() for k in pnov_keywords))
    pnov_df = df[pnov_mask].copy()

    if len(pnov_df) == 0:
        st.info("No PNOV parcels found in this dataset. PNOV parcels have sub-bucket containing 'No Further Status' or 'PNOV'.")
        return

    st.markdown(f"**{len(pnov_df)} PNOV parcels** — {fmt_cost(pnov_df['Cost (£)'].sum())} ({round(len(pnov_df)/total*100,1)}% of all losses)")

    # Summary metrics
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

            # Flex vs DSP split
            st.markdown("---")
            st.markdown("**Flex vs DSP PNOV Split:**")
            type_counts = dsp_data["Driver Type"].value_counts()
            if len(type_counts) > 0:
                fig, ax = plt.subplots(figsize=(3, 1.5))
                colors_map = {"Flex": "#e74c3c", "DSP": "#3498db", "Unknown": "#95a5a6"}
                ax.bar(type_counts.index, type_counts.values, color=[colors_map.get(x, "gray") for x in type_counts.index])
                for i, v in enumerate(type_counts.values): ax.text(i, v+0.2, str(int(v)), ha="center", fontsize=8)
                ax.set_ylabel("PNOV Count", fontsize=8); ax.set_title("PNOV: Flex vs DSP", fontsize=9)
                ax.tick_params(labelsize=7); plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("No DSP/driver data available for PNOV parcels.")

    # By Cluster (where they were dispatched from)
    with st.expander("📍 PNOV by Dispatch Cluster"):
        cl = pnov_df["Cluster"].dropna().value_counts()
        if len(cl) > 0:
            vm = st.radio("View:", ["Chart", "Table"], horizontal=True, key=f"{kp}pnov_cl_v")
            if vm == "Chart":
                st.pyplot(make_bar_horiz(cl.head(15), f"PNOV by Cluster ({dr})", color="purple"))
            else:
                st.dataframe(make_table(cl, "Cluster", "PNOV Count"), width="stretch")
        else:
            st.info("No cluster data for PNOV parcels.")

    # Loss reasons for PNOV
    with st.expander("❓ PNOV Loss Reasons"):
        lr = pnov_df["Loss Reason"].dropna().value_counts()
        if len(lr) > 0:
            st.dataframe(make_table(lr, "Reason", "Count"), width="stretch")

    # All PNOV tracking IDs
    with st.expander("📋 All PNOV Tracking IDs"):
        tid_cols = [c for c in ["Tracking ID","Marked Lost DT","Cluster","Aisle","DSP Name","Driver Type","Cost (£)","Loss Reason"] if c in pnov_df.columns]
        tid_df = pnov_df[tid_cols].copy()
        if "Marked Lost DT" in tid_df.columns:
            tid_df = tid_df.sort_values("Marked Lost DT", ascending=True, na_position="last")
        tid_df = tid_df.reset_index(drop=True)
        tid_df.index = range(1, len(tid_df)+1)
        st.dataframe(tid_df, width="stretch", height=400)
        st.download_button("⬇️ Download PNOV parcels", tid_df.to_csv(index=False), "PNOV_parcels.csv", "text/csv", key=f"{kp}dl_pnov")


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
            vm = st.radio("View:",["Table","Chart"],horizontal=True,key=f"{kp}cdsp")
            if vm=="Chart":
                cd2 = dsp_df.groupby("DSP Name")["Cost (£)"].sum().sort_values(ascending=False)
                h = max(2,len(cd2)*0.3); fig,ax = plt.subplots(figsize=(7,h))
                ax.barh(trunc(cd2.index,DSP_MAX),cd2.values,color="purple"); ax.invert_yaxis()
                for i,v in enumerate(cd2.values): ax.text(v+0.5,i,fmt_cost(v),va="center",fontsize=6)
                ax.set_xlabel("£",fontsize=8); ax.set_title("Cost by DSP",fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
            else: st.dataframe(dc,width="stretch")
    with st.expander("💰 Top 10 Most Expensive Parcels"):
        top = df.nlargest(10,"Cost (£)")
        sc2 = [c for c in ["Tracking ID","Sub Bucket","Type","Shift","Cost (£)","DSP Name","Cluster","Loss Reason"] if c in top.columns]
        out = top[sc2].reset_index(drop=True); out.index = range(1,len(out)+1); st.dataframe(out,width="stretch")


def render_analysis_tab(df, total, dr, kp=""):
    """Simple, user-friendly root cause analysis."""
    st.markdown("### 🔬 Analysis")
    st.warning(
        "⚠️ This section helps you process the vast amount of data and offers some suggestions, "
        "but **SHOULD NOT BE USED AS A GUIDE, IT IS ONLY AN AID.** "
        "Always use your own judgement and local knowledge when deciding on actions."
    )

    with st.expander("📐 Statistical tests used (for managers)"):
        st.markdown("""
**What tests does this tool use and why?**

| Test | Used for | What it tells you | Why this test |
|------|---------|-------------------|---------------|
| **Chi-squared (χ²)** | Shift balance | "Is the imbalance real or just random chance?" | Standard test for comparing observed vs expected counts. If p < 0.05, there's less than 5% chance the pattern is random. |
| **Gini coefficient** | Cluster concentration | "Are losses piling up in a few spots or spread evenly?" | Ranges 0→1. Higher = more concentrated. |
| **Z-score** | DSP outlier detection | "Is this DSP genuinely worse or just slightly above average?" | Measures how many standard deviations from the mean. Z > 1.5 = statistically unusual. |
""")

    total_cost = df["Cost (£)"].sum()
    findings = []

    # ─── LOSS REASONS OVERVIEW ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Loss Reasons — What to focus on")
    st.caption("Ordered by cost impact. The top reason is where most money is being lost.")
    lr = df.groupby("Loss Reason").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
    lr["% of Total"] = (lr["Count"] / total * 100).round(1).astype(str) + "%"
    lr["Avg £/parcel"] = (lr["Cost"] / lr["Count"]).round(2)
    lr_display = lr.copy()
    lr_display["Cost"] = lr_display["Cost"].apply(fmt_cost)
    lr_display["Avg £/parcel"] = lr_display["Avg £/parcel"].apply(fmt_cost)
    lr_display.index = range(1, len(lr_display)+1)
    with st.expander("📊 Loss Reasons (table + chart)", expanded=True):
        vm = st.radio("View:", ["Table","Chart (by count)","Chart (by cost)"], horizontal=True, key=f"{kp}lr_view")
        if vm == "Table":
            st.dataframe(lr_display, width="stretch")
        elif vm == "Chart (by count)":
            lrc = lr.set_index("Loss Reason")["Count"].sort_values(ascending=False)
            st.pyplot(make_bar_horiz(lrc, f"Loss Reasons by Count ({dr})", color="coral"))
        else:
            lrc2 = lr.set_index("Loss Reason")["Cost"].sort_values(ascending=False)
            h = max(2, len(lrc2)*0.3); fig, ax = plt.subplots(figsize=(7, h))
            ax.barh(trunc(lrc2.index, 30), lrc2.values, color="teal"); ax.invert_yaxis()
            for i, v in enumerate(lrc2.values): ax.text(v+0.5, i, fmt_cost(v), va="center", fontsize=6)
            ax.set_xlabel("£", fontsize=8); ax.set_title(f"Loss Reasons by Cost ({dr})", fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout()
            st.pyplot(fig)
        if len(lr) > 0:
            top_reason = lr.iloc[0]
            findings.append(f"Top loss reason: '{top_reason['Loss Reason']}' — {int(top_reason['Count'])} parcels, {fmt_cost(top_reason['Cost'])}")

    # ─── RESULTS ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 What the data suggests")
    st.caption("Look for 🎯 = key finding. These are patterns in the data, not instructions.")

    # 1. WHERE
    with st.expander("📍 1. Where are most losses happening?", expanded=True):
        cl_c = df["Cluster"].dropna().value_counts()
        if len(cl_c) >= 2:
            top3 = cl_c.head(3)
            top3_pct = round(top3.sum() / cl_c.sum() * 100, 1)
            top3_cost = df[df["Cluster"].isin(top3.index)]["Cost (£)"].sum()
            top3_names = ", ".join(top3.index.tolist())
            vals = cl_c.values.astype(float); n = len(vals)
            sv = np.sort(vals)
            gini = (2 * np.sum(np.arange(1, n+1) * sv) - (n+1) * np.sum(sv)) / (n * np.sum(sv))
            if gini > 0.5:
                st.error(f"🎯 Losses piling up in a few spots. Top 3 clusters = **{top3_pct}%** of all losses.")
            elif gini > 0.3:
                st.warning(f"⚠️ Some areas worse than others. Top 3 = **{top3_pct}%** of losses.")
            else:
                st.info(f"ℹ️ Losses fairly spread out. Top 3 = {top3_pct}%.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Top 3 clusters", f"{top3_pct}% of losses")
                st.caption(top3_names)
            with col2:
                st.metric("Cost in those 3", fmt_cost(top3_cost))
            if gini > 0.3:
                findings.append(f"Top 3 clusters ({top3_names}) = {top3_pct}% of losses, costing {fmt_cost(top3_cost)}")
        else:
            st.warning("Not enough location data.")

    # 2. SHIFT
    with st.expander("⏰ 2. Is one shift losing more than it should?"):
        shift_counts = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
        assigned = shift_counts.sum()
        if assigned >= 20:
            expected_per_shift = assigned / 4
            observed = np.array([shift_counts[s] for s in SHIFT_ORDER])
            expected = np.array([expected_per_shift] * 4)
            chi2, p = sp_stats.chisquare(observed, f_exp=expected)
            worst_s = shift_counts.idxmax(); worst_n = int(shift_counts.max())
            tbl = pd.DataFrame({"Shift": SHIFT_ORDER, "Actual": observed.astype(int), "Expected": expected.astype(int), "Over/Under": (observed - expected).astype(int)})
            tbl.index = range(1, 5); st.dataframe(tbl, width="stretch")
            if p < 0.05:
                st.error(f"🎯 **{worst_s}** shift has {worst_n} losses vs ~{int(expected_per_shift)} expected. Unlikely to be random.")
                st.caption(f"Window: {SHIFT_DEFINITIONS[worst_s]}")
                findings.append(f"{worst_s} shift has {worst_n} losses vs expected {int(expected_per_shift)} — not random")
            else:
                st.success(f"✅ Shift differences look like normal variation. No single shift stands out.")
        else:
            st.warning(f"Need 20+ parcels with shift data (have {assigned}).")

    # 3. COST
    with st.expander("💰 3. Which loss types hit the most expensive parcels?"):
        sb_s = df.groupby("Sub Bucket").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).reset_index()
        if len(sb_s) >= 2 and total_cost > 0:
            sb_s["Avg £/parcel"] = (sb_s["Cost"] / sb_s["Count"]).round(2)
            overall_avg = total_cost / total
            sb_s = sb_s.sort_values("Avg £/parcel", ascending=False)
            high = sb_s[sb_s["Avg £/parcel"] > overall_avg * 1.5]
            if len(high) > 0:
                st.error(f"🎯 {len(high)} loss type(s) are hitting expensive parcels:")
                for _, row in high.iterrows():
                    st.markdown(f"- **{row['Sub Bucket']}** — avg {fmt_cost(row['Avg £/parcel'])}/parcel vs station avg {fmt_cost(overall_avg)}")
                    findings.append(f"{row['Sub Bucket']} avg {fmt_cost(row['Avg £/parcel'])}/parcel (station avg {fmt_cost(overall_avg)})")
            else:
                st.success("✅ All loss types hit similarly-priced parcels.")
            display = sb_s[["Sub Bucket", "Count", "Avg £/parcel"]].copy()
            display["Avg £/parcel"] = display["Avg £/parcel"].apply(fmt_cost)
            display.index = range(1, len(display)+1)
            st.dataframe(display, width="stretch")
            st.caption(f"Station average: {fmt_cost(overall_avg)} per lost parcel")
        else:
            st.warning("Need cost data and 2+ loss types.")

    # 4. DSP
    with st.expander("🚚 4. Is any DSP losing way more than others?"):
        otr = df[df["Type"] == "OTR"]
        if len(otr) >= 5:
            dc = otr["DSP Name"].dropna().value_counts()
            if len(dc) >= 3:
                mu = dc.mean(); sigma = dc.std()
                z = (dc - mu) / sigma if sigma > 0 else pd.Series(0, index=dc.index)
                outliers = z[z > 1.5]
                col1, col2 = st.columns(2)
                col1.metric("Avg losses/DSP", f"{mu:.1f}")
                col2.metric("DSPs tracked", len(dc))
                tbl = pd.DataFrame({"DSP": dc.index, "Losses": dc.values, "Verdict": ["🎯 Well above average" if x > 1.5 else "✅ Normal" for x in z.values]})
                tbl.index = range(1, len(tbl)+1); st.dataframe(tbl, width="stretch")
                if len(outliers) > 0:
                    st.error(f"🎯 {len(outliers)} DSP(s) losing much more than peers:")
                    for dsp, zv in outliers.items():
                        cost = otr[otr["DSP Name"] == dsp]["Cost (£)"].sum()
                        reason = otr[otr["DSP Name"] == dsp]["Loss Reason"].dropna().value_counts()
                        r_str = reason.index[0] if len(reason) > 0 else "Unknown"
                        st.markdown(f"- **{dsp}** — {int(dc[dsp])} losses ({fmt_cost(cost)}), top reason: _{r_str}_")
                        findings.append(f"DSP '{dsp}': {int(dc[dsp])} losses ({fmt_cost(cost)}), reason: {r_str}")
                else:
                    st.success("✅ No outliers — losses spread fairly across DSPs.")
            else:
                st.warning("Need 3+ DSPs to compare.")
        else:
            st.warning(f"Need 5+ OTR parcels (have {len(otr)}).")

    # 5. SIZE
    with st.expander("📏 5. Are big parcels getting lost more?"):
        sc3 = df["Size Category"].value_counts()
        if len(sc3) >= 2:
            oversize_cats = ["Small Oversize", "Standard Oversize", "Large Oversize"]
            ov = sum(sc3.get(k, 0) for k in oversize_cats)
            ov_pct = round(ov / total * 100, 1)
            ov_cost = df[df["Size Category"].isin(oversize_cats)]["Cost (£)"].sum()
            stbl = df.groupby("Size Category").agg(Count=("Tracking ID", "count"), Cost=("Cost (£)", "sum")).sort_values("Count", ascending=False).reset_index()
            stbl["% of losses"] = (stbl["Count"] / total * 100).round(1)
            stbl["Cost"] = stbl["Cost"].apply(fmt_cost); stbl.index = range(1, len(stbl)+1)
            st.dataframe(stbl, width="stretch")
            if ov_pct > 30:
                st.error(f"🎯 Oversized = **{ov_pct}%** of losses (normal ~15-20%). Costing {fmt_cost(ov_cost)}.")
                findings.append(f"Oversized parcels = {ov_pct}% of losses ({fmt_cost(ov_cost)})")
            elif ov_pct > 20:
                st.warning(f"⚠️ Oversized = {ov_pct}% — slightly elevated.")
            else:
                st.success(f"✅ Oversized = {ov_pct}% — normal range.")
        else:
            st.warning("Need package dimension data.")

    # 6. REPEAT AISLES
    with st.expander("🔁 6. Any repeat offender aisles?"):
        aisle_c = df["Aisle"].dropna().value_counts()
        if len(aisle_c) >= 2:
            repeat = aisle_c[aisle_c >= 3]
            if len(repeat) > 0:
                st.error(f"🎯 **{len(repeat)} aisle(s)** lost 3+ parcels each:")
                repeat_df = pd.DataFrame({"Aisle": repeat.index, "Losses": repeat.values})
                repeat_df["Cost"] = [df[df["Aisle"]==a]["Cost (£)"].sum() for a in repeat.index]
                repeat_df["Cost"] = repeat_df["Cost"].apply(fmt_cost)
                repeat_df["Top Sub Bucket"] = [df[df["Aisle"]==a]["Sub Bucket"].value_counts().index[0] if len(df[df["Aisle"]==a]["Sub Bucket"].value_counts())>0 else "N/A" for a in repeat.index]
                repeat_df.index = range(1, len(repeat_df)+1)
                st.dataframe(repeat_df, width="stretch")
                findings.append(f"Aisle {repeat.index[0]} lost {int(repeat.iloc[0])} parcels — repeat offender")
            else:
                st.success("✅ No single aisle has 3+ losses.")
        else:
            st.warning("Not enough aisle data.")

    # 7. UTR vs OTR
    with st.expander("🏠🚚 7. Station vs Road — where's the bigger problem?"):
        otr_n = len(df[df["Type"]=="OTR"]); utr_n = len(df[df["Type"]=="UTR"])
        otr_cost = df[df["Type"]=="OTR"]["Cost (£)"].sum()
        utr_cost = df[df["Type"]=="UTR"]["Cost (£)"].sum()
        if otr_n + utr_n > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏠 UTR (Station)", f"{utr_n} parcels", fmt_cost(utr_cost))
                st.caption(f"{round(utr_n/(otr_n+utr_n)*100,1)}% of losses")
            with col2:
                st.metric("🚚 OTR (Road)", f"{otr_n} parcels", fmt_cost(otr_cost))
                st.caption(f"{round(otr_n/(otr_n+utr_n)*100,1)}% of losses")
            if utr_n > otr_n * 1.5:
                st.error("🎯 Station losses (UTR) dominate. Focus on in-station processes.")
                findings.append(f"UTR dominates: {utr_n} vs {otr_n} OTR")
            elif otr_n > utr_n * 1.5:
                st.error("🎯 Road losses (OTR) dominate. Focus on DSPs and dispatch.")
                findings.append(f"OTR dominates: {otr_n} vs {utr_n} UTR")
            else:
                st.info("ℹ️ Fairly balanced between station and road losses.")
        else:
            st.warning("No type data available.")

    # ─── SUGGESTED ACTIONS ────────────────────────────────────────────────────
    if findings:
        st.markdown("---")
        st.markdown("#### 💡 Suggested areas to look at")
        st.caption("These are suggestions only — not instructions. You know your station best.")
        for i, f in enumerate(findings, 1):
            st.markdown(f"**{i}.** {f}")


def render_trend_tab(df, total, dr, kp=""):
    """Week-over-week trend — manual entry with sub-bucket breakdown."""
    st.markdown("#### 📈 Week-over-Week Trend")
    st.warning("⚠️ **Disclaimer:** Trends are only meaningful with enough data. "
               "Include at least 3-4 weeks for a reliable picture.")
    st.caption("PerfectMile shows weekly data separately (L&U → Lost Focus). "
               "Enter totals manually, or upload your weekly PM CSVs.")

    trend_mode = st.radio("Input method:", ["📝 Enter weekly totals", "📂 Upload weekly CSVs"], horizontal=True, key=f"{kp}tm")

    if trend_mode == "📝 Enter weekly totals":
        st.markdown("**Enter numbers from PerfectMile → L&U → Lost Focus (Weekly view):**")
        num_weeks = st.slider("How many weeks?", 1, 12, 4, key=f"{kp}nw")
        include_breakdown = st.checkbox("Include breakdown by loss type (PNOV, Inducted Not Stowed, etc.)", key=f"{kp}bd")
        loss_types = ["PNOV", "Inducted Not Stowed", "Stowed Not Picked Up", "UTR Reprocess", "Lost On Road"]
        weeks_data = []

        if not include_breakdown:
            cols = st.columns(min(num_weeks, 6))
            for i in range(num_weeks):
                with cols[i % 6]:
                    wk_label = st.text_input(f"Week {i+1}:", value=f"W{i+1}", key=f"{kp}wl{i}")
                    wk_count = st.number_input(f"Total lost:", min_value=0, value=0, step=1, key=f"{kp}wc{i}")
                    if wk_count > 0:
                        weeks_data.append({"Week": wk_label, "Total": int(wk_count)})
        else:
            st.markdown("---")
            for i in range(num_weeks):
                with st.expander(f"Week {i+1}", expanded=(i < 2)):
                    wk_label = st.text_input(f"Label:", value=f"W{i+1}", key=f"{kp}wl{i}")
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        wk_total = st.number_input(f"Total lost:", min_value=0, value=0, step=1, key=f"{kp}wc{i}")
                    with c2:
                        st.caption("Breakdown (optional — leave 0 if unknown):")
                    type_cols = st.columns(len(loss_types))
                    type_values = {}
                    for j, lt in enumerate(loss_types):
                        with type_cols[j]:
                            v = st.number_input(lt, min_value=0, value=0, step=1, key=f"{kp}lt{i}_{j}")
                            type_values[lt] = v
                    if wk_total > 0:
                        row = {"Week": wk_label, "Total": int(wk_total)}
                        for lt in loss_types: row[lt] = type_values[lt]
                        weeks_data.append(row)

        if len(weeks_data) >= 2:
            weekly = pd.DataFrame(weeks_data)
            _render_trend_charts(weekly, loss_types if include_breakdown else None, kp)
        elif len(weeks_data) == 1:
            st.info("Enter 1 more week to compare.")
        else:
            st.info("👆 Enter weekly totals above from PerfectMile → L&U → Lost Focus.")

    else:
        st.markdown("**Upload PM CSVs from different weeks:**")
        num_files = st.slider("How many weeks?", 1, 8, 4, key=f"{kp}nf")
        week_files = []
        loss_types = ["PNOV", "Inducted Not Stowed", "Stowed Not Picked Up", "UTR Reprocess", "Lost On Road"]

        for i in range(num_files):
            col1, col2 = st.columns([1, 3])
            with col1:
                wk_label = st.text_input(f"Label:", value=f"W{i+1}", key=f"{kp}fl{i}")
            with col2:
                f_up = st.file_uploader(f"PM CSV ({wk_label}):", type="csv", key=f"{kp}fu{i}")
            if f_up:
                wdf = pd.read_csv(f_up)
                row = {"Week": wk_label, "Total": len(wdf)}
                if "sub_bucket" in wdf.columns:
                    sb = wdf["sub_bucket"].fillna("")
                    row["PNOV"] = int(sb.str.contains("PNOV|No Further Status", case=False).sum())
                    row["Inducted Not Stowed"] = int(sb.str.contains("Inducted Not Stowed", case=False).sum())
                    row["Stowed Not Picked Up"] = int(sb.str.contains("Stowed Not Picked Up", case=False).sum())
                    row["UTR Reprocess"] = int(sb.str.contains("UTR Reprocess", case=False).sum())
                    row["Lost On Road"] = int(sb.str.contains("Lost On Road", case=False).sum())
                week_files.append(row)

        if len(week_files) >= 2:
            weekly = pd.DataFrame(week_files)
            has_breakdown = any(weekly.get(lt, pd.Series(0)).sum() > 0 for lt in loss_types)
            _render_trend_charts(weekly, loss_types if has_breakdown else None, kp)
        elif len(week_files) == 1:
            st.info("Upload 1 more week to see a trend comparison.")
        else:
            st.info("👆 Upload PM CSVs from different weeks above.")


def _render_trend_charts(weekly, loss_types=None, kp=""):
    """Render trend charts from weekly DataFrame."""
    st.markdown("---")
    st.markdown("##### 📈 Overall Trend")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(weekly["Week"], weekly["Total"], marker="o", color="steelblue", linewidth=2, label="Total Lost")
    for i, row in weekly.iterrows():
        ax.annotate(str(int(row["Total"])), xy=(row["Week"], row["Total"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    avg = weekly["Total"].mean()
    ax.axhline(y=avg, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(weekly)-1, avg+2, f"avg: {avg:.0f}", fontsize=7, color="gray")
    ax.set_xlabel("Week", fontsize=8); ax.set_ylabel("Losses", fontsize=8)
    ax.set_title("Lost Parcels per Week", fontsize=9); ax.tick_params(labelsize=7)
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig)

    first_w = int(weekly.iloc[0]["Total"]); last_w = int(weekly.iloc[-1]["Total"])
    if first_w > 0:
        pct_change = round((last_w - first_w) / first_w * 100, 1)
    else:
        pct_change = 0

    if last_w < first_w * 0.8:
        st.success(f"📉 **Improving!** {first_w} → {last_w} ({pct_change:+.1f}%)")
    elif last_w > first_w * 1.2:
        st.error(f"📈 **Getting worse.** {first_w} → {last_w} ({pct_change:+.1f}%)")
    else:
        st.info(f"➡️ **Stable.** {first_w} → {last_w} ({pct_change:+.1f}%, within ±20%)")

    if loss_types:
        has_data = [lt for lt in loss_types if lt in weekly.columns and weekly[lt].sum() > 0]
        if has_data:
            st.markdown("---")
            st.markdown("##### 📊 Trend by Loss Type")
            selected_types = st.multiselect("Show on chart:", has_data, default=has_data[:3], key=f"{kp}lt_sel")
            if selected_types:
                colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
                fig2, ax2 = plt.subplots(figsize=(7, 3.5))
                for idx, lt in enumerate(selected_types):
                    c = colors[idx % len(colors)]
                    ax2.plot(weekly["Week"], weekly[lt], marker="o", linewidth=1.5, label=lt, color=c)
                ax2.set_xlabel("Week", fontsize=8); ax2.set_ylabel("Losses", fontsize=8)
                ax2.set_title("Loss Types Over Time", fontsize=9); ax2.tick_params(labelsize=7)
                ax2.legend(fontsize=7, loc="upper right")
                plt.xticks(rotation=45); plt.tight_layout()
                st.pyplot(fig2)

    st.download_button("⬇️ Download trend data", weekly.to_csv(index=False), "weekly_trend.csv", "text/csv", key=f"{kp}dl_trend")


def render_distribution_tab(df, total, dr, kp=""):
    """Shows actual loss event distribution across the week using Marked Lost DT date (event_datetime).

    NOTE: The Day of Week from event_datetime shows when the SYSTEM marked the parcel as lost,
    NOT when it actually went missing. This tab shows the distribution as-is and explains the caveat.
    """
    st.markdown("#### 📅 Loss Event Distribution")
    st.warning(
        "⚠️ **Important:** The dates below show when **Perfect Mile flagged the parcel as lost** "
        "(event_datetime), NOT when the loss physically occurred. The system often marks parcels lost "
        "in batches (e.g. end-of-day scrubs). This means the day distribution reflects system timing, "
        "not real loss timing. Use this to understand system patterns, not to identify 'problem days'."
    )

    if "Marked Lost DT" not in df.columns or df["Marked Lost DT"].dropna().count() == 0:
        st.warning("No date data available."); return

    valid_dates = df["Marked Lost DT"].dropna()

    # Show actual date distribution (which specific dates had most losses)
    with st.expander("📊 Losses by Actual Date (chronological)", expanded=True):
        date_counts = valid_dates.dt.date.value_counts().sort_index()
        if len(date_counts) > 0:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(range(len(date_counts)), date_counts.values, color="steelblue")
            if len(date_counts) <= 14:
                ax.set_xticks(range(len(date_counts)))
                ax.set_xticklabels([d.strftime("%d %b") for d in date_counts.index], fontsize=7, rotation=45)
            else:
                step = max(1, len(date_counts) // 10)
                ax.set_xticks(range(0, len(date_counts), step))
                ax.set_xticklabels([date_counts.index[i].strftime("%d %b") for i in range(0, len(date_counts), step)], fontsize=7, rotation=45)
            for i, v in enumerate(date_counts.values):
                if v >= date_counts.values.max() * 0.7:
                    ax.text(i, v+0.2, str(int(v)), ha="center", fontsize=6)
            ax.set_ylabel("Parcels Marked Lost", fontsize=8)
            ax.set_title(f"Loss Events by Date ({dr})", fontsize=9)
            ax.tick_params(labelsize=7); plt.tight_layout()
            st.pyplot(fig)

            st.caption(f"Peak date: **{date_counts.idxmax().strftime('%A %d %b %Y')}** ({int(date_counts.max())} parcels)")

    # Day of week aggregation with caveat
    with st.expander("📅 Day of Week Aggregation (system flag dates)"):
        st.caption("This aggregates by the day the system flagged the loss. Due to batch processing "
                   "(e.g. EOD scrubs at ~23:00), certain days may appear inflated. This is a system artefact, "
                   "not a reflection of when losses occurred.")
        dd = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
        tbl = pd.DataFrame({"Day": DAY_ORDER, "Flagged Lost": [int(dd[d]) for d in DAY_ORDER]})
        tbl["% of Total"] = (tbl["Flagged Lost"] / tbl["Flagged Lost"].sum() * 100).round(1)
        tbl.index = range(1, 8)
        st.dataframe(tbl, width="stretch")

        if dd.sum() >= 14:
            expected = dd.sum() / 7
            max_day = dd.idxmax(); max_val = int(dd.max())
            if max_val > expected * 1.5:
                st.info(f"ℹ️ **{max_day}** shows {max_val} losses (vs ~{int(expected)} expected if even). "
                        f"This is likely due to system batch processing timing, not actual loss patterns.")

    # Hour of day distribution (from Prev Event DT — actual last scan time)
    with st.expander("⏰ Hour of Last Scan (actual loss timing proxy)"):
        st.caption("This uses the `previous_event_datetime` — the time of the last scan before the parcel was lost. "
                   "This is a better proxy for WHEN losses actually occur than the system flag date.")
        if "Prev Event DT" in df.columns:
            hours = df["Prev Event DT"].dropna().dt.hour
            if len(hours) > 0:
                hour_counts = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                fig, ax = plt.subplots(figsize=(8, 2.5))
                colors = []
                for h in range(24):
                    shift = SHIFT_HOUR_MAP.get(h, "Unknown")
                    colors.append(SHIFT_COLORS.get(shift, "gray"))
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


def render_export_tab(df, total, dr, kp="", station_name=""):
    """Export tab."""
    st.markdown("#### 💾 Export All Processed Data")

    exp_mode = st.radio("Format:", ["📊 Multi-file ZIP", "📄 Single CSV"], horizontal=True, key=f"{kp}exp_fmt")

    if exp_mode == "📊 Multi-file ZIP":
        import zipfile
        output = BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","shipment_value"]
            clean_cols = [c for c in df.columns if c not in exc]
            zf.writestr("All_Data.csv", df[clean_cols].to_csv(index=False))
            if df["Cluster"].dropna().count() > 0:
                loc_df = df.groupby("Cluster").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Lost", ascending=False).reset_index()
                zf.writestr("By_Location.csv", loc_df.to_csv(index=False))
            shift_df = df[df["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).reindex(SHIFT_ORDER).reset_index()
            zf.writestr("By_Shift.csv", shift_df.to_csv(index=False))
            dsp_data = df.dropna(subset=["DSP Name"])
            if len(dsp_data) > 0:
                dsp_df = dsp_data.groupby("DSP Name").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Lost", ascending=False).reset_index()
                zf.writestr("By_DSP.csv", dsp_df.to_csv(index=False))
            lr_df = df.groupby("Loss Reason").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
            zf.writestr("Loss_Reasons.csv", lr_df.to_csv(index=False))
        output.seek(0)
        fname = f"{station_name}_Analysis.zip" if station_name else "Lost_Parcel_Analysis.zip"
        st.download_button(f"⬇️ Download ZIP ({fname})", output, fname, "application/zip", key=f"{kp}dl_zip")
    else:
        exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","shipment_value"]
        ec = [c for c in df.columns if c not in exc]
        fname = f"{station_name}_Clean.csv" if station_name else "Lost_Clean.csv"
        st.download_button(f"⬇️ Clean CSV", df[ec].to_csv(index=False), fname, "text/csv", key=f"{kp}dl_clean")


def render_guide():
    """How to use — top-level guide."""
    st.markdown("### 📖 How to Use This Tool")
    st.markdown("---")
    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
**You need two CSV files:**

| File | Where to get it |
|------|----------------|
| **Perfect Mile** | PerfectMile → L&U → Lost → Export CSV |
| **SCC** | SCC → paste Tracking IDs → Export |

**Steps:** Upload both → Read the tabs → Pick one problem to investigate → Go observe
""")
    with st.expander("📊 What each tab shows"):
        st.markdown("""
| Tab | What it tells you |
|-----|------------------|
| 📊 **Summary** | Overview — OTR vs UTR, clusters, sub-buckets, sizes |
| 📍 **Locations** | Where in the station losses happen |
| 💡 **Shifts** | Which time window loses most + tracking IDs |
| 📦 **PNOV** | Package Not On Vehicle — DSPs/Flex responsible |
| 💰 **Cost** | Financial impact by type, DSP |
| 🔬 **Analysis** | Pattern detection (an AID, not a GUIDE) |
| 📈 **Trend** | Week-over-week (needs 2+ weeks) |
| 📅 **Distribution** | When losses are flagged (system timing) |
| 💾 **Export** | Download for further analysis |
""")
    with st.expander("📏 Size Classification (Amazon UK FBA Tiers)"):
        st.markdown(f"""
This tool classifies parcels using **Amazon UK FBA size tiers** based on the longest side dimension:

{SIZE_TIER_INFO}

*Source: Amazon Seller Central Europe — Product Size Tier reference*
""")

# ─── MAIN ────────────────────────────────────────────────────────────────────
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
        # Tabs (removed Heatmap and Day, added PNOV and Distribution)
        t1,t2,t3,t4,t5,t6,t7,t8,t9 = st.tabs(["📊 Summary","📍 Locations","💡 Shifts","📦 PNOV","💰 Cost","🔬 Analysis","📈 Trend","📅 Distribution","💾 Export"])
        with t1:
            # OTR vs UTR pie
            with st.expander("🥧 OTR vs UTR"): st.pyplot(make_pie_otr_utr(df, total, f"OTR vs UTR ({dr})"))

            # Combined Cluster view (single expander with all cluster data)
            with st.expander("📍 Clusters — All Losses", expanded=True):
                cc = df["Cluster"].dropna().value_counts()
                if len(cc)>0:
                    vm = st.radio("View:", ["Chart","Table + Cost"], horizontal=True, key="cv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(cc, f"All Clusters ({dr})"))
                    else: st.dataframe(make_cost_table(df.dropna(subset=["Cluster"]), "Cluster"), width="stretch")

            # Cluster drill-down selectbox in Summary
            with st.expander("🔍 Cluster Drill-Down"):
                clusters = sorted(df["Cluster"].dropna().unique())
                if clusters:
                    sel_cluster = st.selectbox("Select Cluster:", clusters, key="sum_cl_drill")
                    filt = df[df["Cluster"] == sel_cluster]
                    st.write(f"**{len(filt)} parcels** in {sel_cluster} — {fmt_cost(filt['Cost (£)'].sum())}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**By Aisle:**")
                        ad = filt["Aisle"].dropna().value_counts()
                        if len(ad) > 0: st.dataframe(make_table(ad, "Aisle", "Count"), width="stretch")
                    with col2:
                        st.markdown("**By Shift:**")
                        sd = filt[filt["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
                        st.dataframe(make_table(sd, "Shift", "Count"), width="stretch")
                else:
                    st.info("No cluster data available.")

            # Sub Buckets
            with st.expander("🏷️ Sub Buckets"):
                sb2 = df["Sub Bucket"].value_counts()
                if len(sb2)>0:
                    vm = st.radio("View:", ["Chart","Table + Cost"], horizontal=True, key="sv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(sb2, f"Sub Buckets ({dr})", color="teal"))
                    else: st.dataframe(make_cost_table(df, "Sub Bucket"), width="stretch")

            # Size Category (Amazon UK FBA tiers)
            with st.expander("📏 Lost Parcels by Size (Amazon UK FBA Tiers)"):
                sc = df["Size Category"].value_counts()
                if len(sc) > 0:
                    size_order = ["Small Envelope", "Standard Envelope", "Standard Parcel", "Small Oversize", "Standard Oversize", "Large Oversize", "Unknown"]
                    sc_ordered = sc.reindex([s for s in size_order if s in sc.index])
                    vm = st.radio("View:", ["Chart", "Table + Cost"], horizontal=True, key="sz_v")
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
                    st.info("No size data available. Ensure SCC has Package Length/Width/Height columns.")

        with t2: render_locations_tab(df, total, dr, kp="s_")
        with t3: render_opportunities_tab(df, total, dr, kp="s_")
        with t4: render_pnov_tab(df, total, dr, kp="s_")
        with t5: render_cost_tab(df, total, dr, kp="s_")
        with t6: render_analysis_tab(df, total, dr, kp="s_")
        with t7: render_trend_tab(df, total, dr, kp="s_")
        with t8: render_distribution_tab(df, total, dr, kp="s_")
        with t9: render_export_tab(df, total, dr, kp="s_")
    else: st.info("👆 Upload both files.")

else:
    st.caption("Upload multiple stations to compare, OR upload the same station from different time periods.")
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
            if nm_input and nm_input.strip():
                nm = nm_input.strip()
            elif "location" in pt.columns and len(pt["location"].dropna()) > 0:
                nm = pt["location"].dropna().iloc[0]
            else:
                nm = f"Dataset {i+1}"
            stations[nm] = m; names.append(nm)
        st.success(f"✅ {', '.join(names)}")
        for n in names:
            sc_val, sc_col, sc_lab, sc_reas = render_health_score(stations[n], len(stations[n]))
            st.caption(f"{n}: {sc_col} {sc_val}/10 — {sc_lab}" + (f" ({', '.join(sc_reas)})" if sc_reas else ""))
        t1,t2,t3,t4,t5,t6,t7,t8,t9 = st.tabs(["📊 Summary","📍 Locations","💡 Shifts","📦 PNOV","💰 Cost","🔬 Analysis","📈 Trend","📅 Distribution","💾 Export"])
        with t1:
            for n in names:
                sdf = stations[n]
                st.markdown(f"**{n}:** {len(sdf)} parcels — {fmt_cost(sdf['Cost (£)'].sum())} | Top cluster: {safe_top(sdf['Cluster'])} | Top shift: {safe_top(sdf[sdf['Shift'].isin(SHIFT_ORDER)]['Shift'])}")
        with t2:
            sel = st.selectbox("Dataset:", names, key="mcl"); render_locations_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"m{sel}_")
        with t3:
            sel = st.selectbox("Dataset:", names, key="mcs"); render_opportunities_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"ms{sel}_")
        with t4:
            sel = st.selectbox("Dataset:", names, key="mcp"); render_pnov_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mp{sel}_")
        with t5:
            sel = st.selectbox("Dataset:", names, key="mcc"); render_cost_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc{sel}_")
        with t6:
            sel = st.selectbox("Dataset:", names, key="mca"); render_analysis_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"ma{sel}_")
        with t7:
            sel = st.selectbox("Dataset:", names, key="mct"); render_trend_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mt{sel}_")
        with t8:
            sel = st.selectbox("Dataset:", names, key="mcd"); render_distribution_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"md{sel}_")
        with t9:
            sel = st.selectbox("Dataset:", names, key="mce"); render_export_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"me{sel}_", station_name=sel)
    elif len(uploaded) == 1: st.warning("Need 2+ datasets to compare.")
    else: st.info("👆 Upload pairs of PM + SCC files above.")
