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
SHIFT_DEFINITIONS = {"NS": "00:00 – 09:59 (Night Sort — stow)", "AM": "10:00 – 13:59 (Pick, stage, dispatch)", "PM": "14:00 – 23:59 (Dispatch, RELO)", "OTR": "On The Road (DSP responsibility)"}
SHIFT_HOUR_MAP = {0:"NS",1:"NS",2:"NS",3:"NS",4:"NS",5:"NS",6:"NS",7:"NS",8:"NS",9:"NS",10:"AM",11:"AM",12:"AM",13:"AM",14:"PM",15:"PM",16:"PM",17:"PM",18:"PM",19:"PM",20:"PM",21:"PM",22:"PM",23:"PM"}
SUB_BUCKET_SHIFT_MAP = {"Lost At Station - Inducted Not Stowed":"NS","Lost At Station - Stowed Not Picked Up":"AM","Lost At Station - Debrief Receive(RTS)":"PM","Lost On Road - Attempted":"OTR","Lost On Road - Damage":"OTR","Lost On Road - No Further Status":"OTR"}
SENSITIVE_COLS = ["Holder Name","Ordering Order ID","Order Amount","Receivable Amount","Payment Method","District","Scheduled Delivery End Time"]
REQUIRED_SCC_COLS = ["Tracking ID","Sort Zone","Aisle","Cluster","Package Length","Package Width","Package Height","DSP Name","Assigned Cycle","Last Updated Time"]
REQUIRED_PM_COLS = ["tracking_id","sub_bucket"]
CHART = (7, 2.5)
DSP_MAX = 20
LABEL_MAX = 25

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────
def get_size(val):
    if pd.isna(val): return "Unknown"
    if val <= 35: return "Small"
    if val <= 45: return "Medium"
    if val <= 61: return "Small Oversize"
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
        ov = sc3.get("Small Oversize", 0) + sc3.get("Large Oversize", 0)
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
    with st.expander("📖 Shift assignment logic"):
        st.markdown("| Sub Bucket | Shift |\n|---|---|\n| Inducted Not Stowed | NS |\n| Stowed Not Picked Up | AM |\n| Debrief Receive(RTS) | PM |\n| Lost On Road - * | OTR |\n| PNOV / UTR Reprocess / Other | Time-based (hour of last scan) |")
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
            sbc = sdf["Sub Bucket"].value_counts()
            vm = st.radio("View:",["Sub Buckets","All Tracking IDs"],horizontal=True,key=f"{kp}sd")
            if vm=="Sub Buckets":
                st.dataframe(make_table(sbc,"Sub Bucket","Count"),width="stretch")
            else:
                tid_cols = [c for c in ["Tracking ID","Sub Bucket","Type","Cluster","Aisle","DSP Name","Cost (£)","Loss Reason","Day of Week"] if c in sdf.columns]
                tid_df = sdf[tid_cols].reset_index(drop=True)
                tid_df.index = range(1, len(tid_df)+1)
                st.dataframe(tid_df, width="stretch", height=400)
                st.download_button(f"⬇️ Download {ss} parcels", tid_df.to_csv(index=False), f"{ss}_parcels.csv", "text/csv", key=f"{kp}dl_{ss}")


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

    total_cost = df["Cost (£)"].sum()
    findings = []

    # ─── LOSS REASONS OVERVIEW ────────────────────────────────────────────
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

    # ─── DATA SUFFICIENCY ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Do we have enough data?")
    st.caption("Green = good to go. Red = need more uploads for that test to work.")

    cluster_count = len(df["Cluster"].dropna().value_counts())
    shift_count = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].count()
    otr_count = len(df[df["Type"]=="OTR"])
    day_count = df["Day of Week"].dropna().count() if "Day of Week" in df.columns else 0
    size_count = len(df["Size Category"].value_counts())

    checks = [
        ("Where are losses happening?", "✅ Ready" if cluster_count >= 2 else "❌ Need more cluster data", ""),
        ("Which shift is worst?", "✅ Ready" if shift_count >= 20 else f"❌ Need 20+ parcels (have {shift_count})", "Upload more days"),
        ("Which costs the most?", "✅ Ready" if total_cost > 0 else "❌ No cost data", "Check PM has shipment_value"),
        ("Any problem DSPs?", "✅ Ready" if otr_count >= 5 else f"❌ Need 5+ OTR (have {otr_count})", "Upload more data"),
        ("Any problem days?", "✅ Ready" if day_count >= 14 else f"❌ Need 14+ parcels (have {day_count})", "Upload 2+ weeks"),
        ("Size a factor?", "✅ Ready" if size_count >= 2 else "❌ Need size data", "Check SCC has dimensions"),
        ("Repeat offender aisles?", "✅ Ready" if cluster_count >= 2 else "❌ Need cluster data", ""),
        ("Happy path departure?", "✅ Ready" if len(df["Sub Bucket"].value_counts()) >= 2 else "❌ Need sub-bucket data", ""),
    ]

    ready = sum(1 for _, s, _ in checks if "✅" in s)
    if ready == len(checks):
        st.success(f"✅ All {len(checks)} tests ready!")
    else:
        st.info(f"{ready}/{len(checks)} tests ready. Upload more data to unlock the rest.")
    st.dataframe(pd.DataFrame(checks, columns=["Question", "Status", "How to fix"]), width="stretch", hide_index=True)

    # ─── RESULTS ──────────────────────────────────────────────────────────
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
                        findings.append(f"DSP \'{dsp}\': {int(dc[dsp])} losses ({fmt_cost(cost)}), reason: {r_str}")
                else:
                    st.success("✅ No outliers — losses spread fairly across DSPs.")
            else:
                st.warning("Need 3+ DSPs to compare.")
        else:
            st.warning(f"Need 5+ OTR parcels (have {len(otr)}).")

    # 5. DAY
    with st.expander("📅 5. Do losses spike on certain days?"):
        if "Day of Week" in df.columns:
            dc2 = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0); dt = dc2.sum()
            if dt >= 14:
                exp = np.array([dt/7]*7); obs = np.array([dc2[d] for d in DAY_ORDER])
                chi2d, pd2 = sp_stats.chisquare(obs, f_exp=exp)
                tbl = pd.DataFrame({"Day": DAY_ORDER, "Losses": [int(dc2[d]) for d in DAY_ORDER], "Expected": [int(dt/7)]*7})
                tbl.index = range(1, 8); st.dataframe(tbl, width="stretch")
                if pd2 < 0.05:
                    wd = dc2.idxmax(); wn = int(dc2.max())
                    st.error(f"🎯 **{wd}** is the worst day: {wn} losses vs ~{int(dt/7)} expected.")
                    findings.append(f"{wd} has {wn} losses vs expected {int(dt/7)}")
                else:
                    st.success("✅ No day stands out — losses spread evenly.")
            else:
                st.warning(f"Need 14+ dated parcels (have {dt}).")
        else:
            st.warning("No date data available.")

    # 6. SIZE
    with st.expander("📏 6. Are big parcels getting lost more?"):
        sc3 = df["Size Category"].value_counts()
        if len(sc3) >= 2:
            ov = sc3.get("Small Oversize", 0) + sc3.get("Large Oversize", 0)
            ov_pct = round(ov / total * 100, 1)
            ov_cost = df[df["Size Category"].isin(["Small Oversize", "Large Oversize"])]["Cost (£)"].sum()
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

    # 7. REPEAT AISLES
    with st.expander("🔁 7. Any repeat offender aisles?"):
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

    # 8. UTR vs OTR
    with st.expander("🏠🚚 8. Station vs Road — where\'s the bigger problem?"):
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

    # 9. HAPPY PATH DEPARTURE
    with st.expander("🛤️ 9. Where do parcels leave the happy path?"):
        st.caption("Happy path: Inducted → Stowed → Picked → Dispatched → Delivered. Each sub-bucket shows where the parcel fell off.")
        sb_counts = df["Sub Bucket"].value_counts()
        if len(sb_counts) >= 2:
            sb_total = sb_counts.sum()
            hp_order = [
                ("Inducted Not Stowed", "Between induction and stow (NS)"),
                ("Stowed Not Picked Up", "Between stow and pick (AM)"),
                ("Debrief Receive(RTS)", "Returned after dispatch (PM)"),
                ("Attempted", "Driver attempted, failed delivery (OTR)"),
                ("No Further Status", "Dispatched, no scan after (OTR)"),
                ("Damage", "Damaged on road (OTR)"),
            ]
            rows = []
            for keyword, stage in hp_order:
                matching = sb_counts[sb_counts.index.str.contains(keyword, case=False, na=False, regex=False)]
                if len(matching) > 0:
                    count = int(matching.sum())
                    pct = round(count / sb_total * 100, 1)
                    cost = df[df["Sub Bucket"].str.contains(keyword, case=False, na=False)]["Cost (£)"].sum()
                    rows.append({"Stage": keyword, "What happened": stage, "Count": count, "%": f"{pct}%", "Cost": fmt_cost(cost)})
            if rows:
                hp_df = pd.DataFrame(rows); hp_df.index = range(1, len(hp_df)+1)
                st.dataframe(hp_df, width="stretch")
                worst = max(rows, key=lambda r: r["Count"])
                st.error(f"🎯 Most parcels leave the happy path at: **{worst['Stage']}** ({worst['Count']} parcels, {worst['%']})")
                st.caption(f"What this means: {worst['What happened']}")
                findings.append(f"Most parcels leave happy path at \'{worst['Stage']}\' ({worst['Count']} parcels, {worst['%']})")
            else:
                st.info("Could not map sub-buckets to happy path stages.")
        else:
            st.warning("Need 2+ sub-bucket types.")

    # ─── SUGGESTED ACTIONS ────────────────────────────────────────────────
    if findings:
        st.markdown("---")
        st.markdown("#### 💡 Suggested areas to look at")
        st.caption("These are suggestions only — not instructions. You know your station best.")
        for i, f in enumerate(findings, 1):
            st.markdown(f"**{i}.** {f}")

def render_heatmap_tab(df, total, dr, kp=""):
    """Cluster x Shift heatmap."""
    st.markdown("#### 🗺️ Heatmap — Cluster × Shift")
    st.caption("Darker = more losses. Shows WHERE and WHEN losses happen together.")
    clusters_with_data = df["Cluster"].dropna().value_counts().head(15).index.tolist()
    if len(clusters_with_data) < 2:
        st.warning("Need 2+ clusters with data."); return
    hm_df = df[df["Cluster"].isin(clusters_with_data) & df["Shift"].isin(SHIFT_ORDER)]
    if len(hm_df) == 0:
        st.warning("No data for heatmap."); return
    pivot = hm_df.groupby(["Cluster","Shift"]).size().unstack(fill_value=0).reindex(columns=SHIFT_ORDER, fill_value=0)
    pivot = pivot.loc[clusters_with_data]
    fig, ax = plt.subplots(figsize=(6, max(3, len(clusters_with_data)*0.4)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(SHIFT_ORDER))); ax.set_xticklabels(SHIFT_ORDER, fontsize=8)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(trunc(pivot.index, 15), fontsize=7)
    for i in range(len(pivot.index)):
        for j in range(len(SHIFT_ORDER)):
            val = pivot.values[i, j]
            if val > 0:
                ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=7, color="black" if val < pivot.values.max()*0.7 else "white")
    ax.set_xlabel("Shift", fontsize=8); ax.set_ylabel("Cluster", fontsize=8)
    ax.set_title(f"Losses by Cluster × Shift ({dr})", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8); plt.tight_layout()
    st.pyplot(fig)
    worst_cell = pivot.stack().idxmax()
    worst_val = int(pivot.stack().max())
    st.markdown(f"🎯 **Hotspot:** {worst_cell[0]} during **{worst_cell[1]}** shift — {worst_val} losses")

def render_trend_tab(df, total, dr, kp=""):
    """Week-over-week trend."""
    st.markdown("#### 📈 Week-over-Week Trend")
    date_col = None
    for col in ["Marked Lost DT", "Dispatch Time"]:
        if col in df.columns and df[col].dropna().count() >= 14:
            date_col = col; break
    if date_col is None:
        st.warning("⚠️ Need 2+ weeks of dated data to show trends. Upload a larger date range.")
        return
    wdf = df.dropna(subset=[date_col]).copy()
    wdf["Week"] = wdf[date_col].dt.isocalendar().week.astype(int)
    wdf["Year"] = wdf[date_col].dt.year
    wdf["YearWeek"] = wdf["Year"].astype(str) + "-W" + wdf["Week"].astype(str).str.zfill(2)
    weekly = wdf.groupby("YearWeek").agg(Losses=("Tracking ID","count"), Cost=("Cost (£)","sum")).reset_index()
    weekly = weekly.sort_values("YearWeek")
    if len(weekly) < 2:
        st.warning("Only 1 week of data. Upload 2+ weeks to see trends."); return
    st.caption(f"Based on \'{date_col}\' column. Each point = one calendar week.")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(weekly["YearWeek"], weekly["Losses"], marker="o", color="steelblue", linewidth=2)
    for i, row in weekly.iterrows():
        ax.annotate(str(int(row["Losses"])), xy=(row["YearWeek"], row["Losses"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Week", fontsize=8); ax.set_ylabel("Losses", fontsize=8)
    ax.set_title("Lost Parcels per Week", fontsize=9); ax.tick_params(labelsize=7)
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig)
    first_w = int(weekly.iloc[0]["Losses"]); last_w = int(weekly.iloc[-1]["Losses"])
    if last_w < first_w * 0.8:
        st.success(f"📉 **Improving!** Down from {first_w} to {last_w} per week.")
    elif last_w > first_w * 1.2:
        st.error(f"📈 **Getting worse.** Up from {first_w} to {last_w} per week.")
    else:
        st.info(f"➡️ **Stable.** {first_w} → {last_w} per week (within ±20%).")
    fig2, ax2 = plt.subplots(figsize=(7, 2.5))
    ax2.bar(weekly["YearWeek"], weekly["Cost"], color="teal")
    for i, row in weekly.iterrows():
        ax2.text(row["YearWeek"], row["Cost"]+0.5, fmt_cost(row["Cost"]), ha="center", fontsize=6)
    ax2.set_xlabel("Week", fontsize=8); ax2.set_ylabel("£", fontsize=8)
    ax2.set_title("Cost per Week", fontsize=9); ax2.tick_params(labelsize=7)
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig2)
    st.dataframe(weekly.rename(columns={"YearWeek":"Week"}), width="stretch", hide_index=True)

def render_day_tab(df, total, dr, kp=""):
    """Day of week tab with tracking ID drill-down."""
    st.markdown("#### 📅 Day of Week")
    st.caption("The day shown is when Perfect Mile marked the parcel as LOST (event_datetime). "
               "This is NOT necessarily when the parcel actually went missing — it\'s when the system "
               "flagged it. A parcel that went missing on Friday night might only be marked lost on Monday.")
    if "Day of Week" not in df.columns or df["Day of Week"].dropna().count() == 0:
        st.warning("No date data available."); return
    dd = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
    with st.expander("📅 Overview", expanded=True):
        vm = st.radio("View:", ["Chart","Table"], horizontal=True, key=f"{kp}dv")
        if vm == "Chart":
            fig, ax = plt.subplots(figsize=CHART)
            ax.plot(dd.index, dd.values, marker="o", color="green", linewidth=2)
            for i, (d, v) in enumerate(dd.items()): ax.annotate(str(int(v)), xy=(i, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
            ax.set_ylabel("Lost", fontsize=8); ax.set_title(f"By Day ({dr})", fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout()
            st.pyplot(fig)
        else:
            st.dataframe(make_table(dd, "Day", "Lost"), width="stretch")
    with st.expander("🔍 Tracking IDs by Day"):
        day_sel = st.selectbox("Select day:", DAY_ORDER, key=f"{kp}day_sel")
        day_df = df[df["Day of Week"] == day_sel]
        if len(day_df) > 0:
            st.write(f"**{day_sel}: {len(day_df)} parcels** — {fmt_cost(day_df['Cost (£)'].sum())}")
            tid_cols = [c for c in ["Tracking ID","Sub Bucket","Type","Cluster","Aisle","DSP Name","Cost (£)","Loss Reason","Shift"] if c in day_df.columns]
            tid_df = day_df[tid_cols].reset_index(drop=True)
            tid_df.index = range(1, len(tid_df)+1)
            st.dataframe(tid_df, width="stretch", height=400)
            st.download_button(f"⬇️ Download {day_sel} parcels", tid_df.to_csv(index=False), f"{day_sel}_parcels.csv", "text/csv", key=f"{kp}dl_{day_sel}")
        else:
            st.info(f"No parcels marked lost on {day_sel}.")

def render_export_tab(df, total, dr, kp="", station_name=""):
    """Export tab — multi-sheet download as ZIP of CSVs or single CSV."""
    st.markdown("#### 💾 Export All Processed Data")
    st.caption("Download everything for further analysis in Excel, R, or Python.")
    
    exp_mode = st.radio("Format:", ["📊 Multi-file ZIP (separate tables)", "📄 Single CSV (all data)"], horizontal=True, key=f"{kp}exp_fmt")
    
    if exp_mode == "📊 Multi-file ZIP (separate tables)":
        st.markdown("""**Files included in ZIP:**
- **All_Data.csv** — every parcel with all processed columns
- **By_Location.csv** — cluster/aisle breakdown with costs
- **By_Shift.csv** — shift breakdown with costs
- **By_Day.csv** — day of week breakdown
- **By_DSP.csv** — DSP breakdown (OTR only)
- **Loss_Reasons.csv** — all reasons ordered by cost""")
        
        import zipfile
        output = BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            # All Data
            exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","shipment_value"]
            clean_cols = [c for c in df.columns if c not in exc]
            zf.writestr("All_Data.csv", df[clean_cols].to_csv(index=False))
            
            # By Location
            if df["Cluster"].dropna().count() > 0:
                loc_df = df.groupby("Cluster").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Lost", ascending=False).reset_index()
                zf.writestr("By_Location.csv", loc_df.to_csv(index=False))
            
            # By Shift
            shift_df = df[df["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).reindex(SHIFT_ORDER).reset_index()
            zf.writestr("By_Shift.csv", shift_df.to_csv(index=False))
            
            # By Day
            if "Day of Week" in df.columns:
                day_df = df.groupby("Day of Week").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).reindex(DAY_ORDER).reset_index()
                zf.writestr("By_Day.csv", day_df.to_csv(index=False))
            
            # By DSP
            dsp_data = df.dropna(subset=["DSP Name"])
            if len(dsp_data) > 0:
                dsp_df = dsp_data.groupby("DSP Name").agg(Lost=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Lost", ascending=False).reset_index()
                zf.writestr("By_DSP.csv", dsp_df.to_csv(index=False))
            
            # Loss Reasons
            lr_df = df.groupby("Loss Reason").agg(Count=("Tracking ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
            lr_df["% of Total"] = (lr_df["Count"] / total * 100).round(1)
            zf.writestr("Loss_Reasons.csv", lr_df.to_csv(index=False))
        
        output.seek(0)
        fname = f"{station_name}_Analysis.zip" if station_name else "Lost_Parcel_Analysis.zip"
        st.download_button(f"⬇️ Download ZIP ({fname})", output, fname, "application/zip", key=f"{kp}dl_zip")
        st.caption(f"{len(df)} parcels across 6 files — open each CSV as a separate tab in Excel")
    
    else:
        clean_mode = st.radio("Columns:", ["Clean (no raw columns)", "Full (all columns for modelling)"], horizontal=True, key=f"{kp}exp_mode")
        if clean_mode == "Clean (no raw columns)":
            exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","shipment_value"]
            ec = [c for c in df.columns if c not in exc]
            fname = f"{station_name}_Clean.csv" if station_name else "Lost_Clean.csv"
            st.download_button(f"⬇️ Clean CSV", df[ec].to_csv(index=False), fname, "text/csv", key=f"{kp}dl_clean")
            st.caption(f"{len(ec)} columns, {len(df)} rows")
        else:
            fname = f"{station_name}_Full.csv" if station_name else "Lost_Full.csv"
            st.download_button(f"⬇️ Full CSV", df.to_csv(index=False), fname, "text/csv", key=f"{kp}dl_full")
            st.caption(f"{len(df.columns)} columns, {len(df)} rows")
    
    st.markdown("---")
    with st.expander("📖 Column guide"):
        st.markdown("""
| Column | Description |
|--------|-------------|
| Tracking ID | Unique parcel identifier |
| Sub Bucket | Where in the process it was lost |
| Type | OTR (road) or UTR (station) |
| Shift | NS / AM / PM / OTR |
| Day of Week | Day parcel was marked lost |
| Cost (£) | Shipment value |
| Cluster / Aisle / Sort Zone | Physical location in station |
| DSP Name | Delivery partner (OTR only) |
| Size Category | Small / Medium / Small Oversize / Large Oversize |
| Loss Reason | Reason code from Perfect Mile |
| UTR Reason | More specific reason for station losses |
| Longest Side | Max dimension in cm |
""")
    st.info("💡 **Tip:** Open the CSVs in Excel (each becomes a tab), or paste into R/Python for deeper modelling and forecasting.")


def render_guide():
    """How to use — top-level guide with expandable sections."""
    st.markdown("### 📖 How to Use This Tool")
    st.markdown("---")
    
    with st.expander("🚀 Quick Start (read this first)", expanded=True):
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
| Tab | What it tells you | When to use it |
|-----|------------------|----------------|
| 📊 **Summary** | Overview — OTR vs UTR, clusters, sub-buckets | First look at the data |
| 📍 **Locations** | Where in the station losses happen | Finding problem areas to walk |
| 💡 **Shifts** | Which time window loses most + all tracking IDs | Identifying shift patterns |
| 💰 **Cost** | Financial impact by type, DSP | Prioritising by £ value |
| 🔬 **Analysis** | Pattern detection (an AID, not a GUIDE) | Deeper investigation |
| 🗺️ **Heatmap** | Cluster × Shift grid | Spotting hotspots at a glance |
| 📈 **Trend** | Week-over-week (needs 2+ weeks) | Tracking improvement |
| 📅 **Day** | Worst day + drill-down to parcels | Finding day patterns |
| 💾 **Export** | Multi-tab Excel or CSV for R/Python | Further modelling |
""")
    
    with st.expander("❓ Common Questions"):
        st.markdown("""
**"What does 'No Reason' mean as a loss reason?"**

This means Perfect Mile has no recorded reason for why the parcel was lost. It happens when:
- The system auto-concessed the parcel (timeout, no scan for X days)
- The parcel was marked lost in bulk without individual investigation
- The reason field was never filled in by the team

**It does NOT mean there IS no reason** — it means the reason wasn't captured. These are often worth investigating as they may indicate process gaps in recording why parcels go missing.

**"Why don't all parcels match?"**

See the ℹ️ info box after upload — parcels may not match if they were never inducted, had no SCC scan, or have a tracking ID format mismatch.

**"Can I compare the same station across different weeks?"**

Yes! Use **Multi-Station mode** — upload Week 1 as "Station 1" and Week 2 as "Station 2". The comparison works the same way. Label them by date range so you can tell them apart.

**"How accurate is the Analysis tab?"**

It uses statistical tests (chi-squared, z-scores, Gini) to find patterns. These are INDICATIONS, not proof. Always verify by walking the floor and talking to the team. The Analysis tab says "here's something unusual" — YOU decide if it matters.
""")
    
    with st.expander("💡 Tips for better results"):
        st.markdown("""
- 📅 **More data = better** — 2+ weeks recommended for trend analysis
- 🔒 **PII auto-removed** — customer names/order IDs are stripped on upload
- 🔄 **Multi-station** — compare stations OR compare time periods
- 💾 **Export** for deeper modelling in R or Python
- 🎯 **Focus on one thing** — pick the top problem, go investigate, come back with fresh data
""")

# ─── MAIN ──────────────────────────────────────────────────────────────────────
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
        # Tabs
        t1,t2,t3,t4,t5,t6,t7,t8,t9 = st.tabs(["📊 Summary","📍 Locations","💡 Shifts","💰 Cost","🔬 Analysis","🗺️ Heatmap","📈 Trend","📅 Day","💾 Export"])
        with t1:
            with st.expander("🥧 OTR vs UTR"): st.pyplot(make_pie_otr_utr(df, total, f"OTR vs UTR ({dr})"))
            with st.expander("📍 Clusters"):
                cc = df["Cluster"].dropna().value_counts()
                if len(cc)>0:
                    vm = st.radio("View:", ["Chart","Table + Cost"], horizontal=True, key="cv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(cc, f"Clusters ({dr})"))
                    else: st.dataframe(make_cost_table(df.dropna(subset=["Cluster"]), "Cluster"), width="stretch")
            with st.expander("🏷️ Sub Buckets"):
                sb2 = df["Sub Bucket"].value_counts()
                if len(sb2)>0:
                    vm = st.radio("View:", ["Chart","Table + Cost"], horizontal=True, key="sv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(sb2, f"Sub Buckets ({dr})", color="teal"))
                    else: st.dataframe(make_cost_table(df, "Sub Bucket"), width="stretch")
        with t2: render_locations_tab(df, total, dr, kp="s_")
        with t3: render_opportunities_tab(df, total, dr, kp="s_")
        with t4: render_cost_tab(df, total, dr, kp="s_")
        with t5: render_analysis_tab(df, total, dr, kp="s_")
        with t6: render_heatmap_tab(df, total, dr, kp="s_")
        with t7: render_trend_tab(df, total, dr, kp="s_")
        with t8: render_day_tab(df, total, dr, kp="s_")
        with t9: render_export_tab(df, total, dr, kp="s_")
    else: st.info("👆 Upload both files.")

else:
    st.caption("Upload multiple stations to compare, OR upload the same station from different time periods to track progress.")
    num = st.slider("Datasets to compare:", 2, 5, 2, key="ns"); uploaded = {}
    for i in range(num):
        with st.expander(f"Dataset {i+1} (e.g. Station or Week)", expanded=(i<2)):
            nm_input = st.text_input(f"Label (optional):", key=f"nm_{i}", placeholder=f"e.g. DRM2 Week 1, or Station Name")
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
        t1,t2,t3,t4,t5,t6,t7,t8,t9 = st.tabs(["📊 Summary","📍 Locations","💡 Shifts","💰 Cost","🔬 Analysis","🗺️ Heatmap","📈 Trend","📅 Day","💾 Export"])
        with t1:
            for n in names:
                sdf = stations[n]
                st.markdown(f"**{n}:** {len(sdf)} parcels — {fmt_cost(sdf['Cost (£)'].sum())} | Top cluster: {safe_top(sdf['Cluster'])} | Top shift: {safe_top(sdf[sdf['Shift'].isin(SHIFT_ORDER)]['Shift'])}")
        with t2:
            sel = st.selectbox("Dataset:", names, key="mcl"); render_locations_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"m{sel}_")
        with t3:
            sel = st.selectbox("Dataset:", names, key="mcs"); render_opportunities_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"ms{sel}_")
        with t4:
            sel = st.selectbox("Dataset:", names, key="mcc"); render_cost_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc{sel}_")
        with t5:
            sel = st.selectbox("Dataset:", names, key="mca"); render_analysis_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"ma{sel}_")
        with t6:
            sel = st.selectbox("Dataset:", names, key="mch"); render_heatmap_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mh{sel}_")
        with t7:
            sel = st.selectbox("Dataset:", names, key="mct"); render_trend_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mt{sel}_")
        with t8:
            sel = st.selectbox("Dataset:", names, key="mcd"); render_day_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"md{sel}_")
        with t9:
            sel = st.selectbox("Dataset:", names, key="mce"); render_export_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"me{sel}_", station_name=sel)
    elif len(uploaded) == 1: st.warning("Need 2+ datasets to compare.")
    else: st.info("👆 Upload pairs of PM + SCC files above.")
