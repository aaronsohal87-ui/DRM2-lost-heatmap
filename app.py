import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

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

def render_missing_parcels(df, total, matched):
    mc = total - matched
    if mc > 0:
        st.info(f"ℹ️ **{mc} parcel(s)** in PM had no SCC match — included but no location detail.")
        mdf = df[df["Cluster"].isna()].copy()
        if len(mdf)>0:
            with st.expander(f"🔍 View {len(mdf)} Missing"):
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
                else: st.dataframe(make_cost_table(vdf.dropna(subset=["DSP Name"]),"DSP Name"),use_container_width=True)
        with st.expander("📍 Delivery Areas"):
            ab = st.radio("By:",["City","Province","Postal"],horizontal=True,key=f"{kp}oa")
            ad = vdf[ab].dropna().value_counts()
            if len(ad)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}oav")
                if vm=="Chart": st.pyplot(make_bar_horiz(ad.head(15),f"OTR by {ab}",color="darkred"))
                else: st.dataframe(make_cost_table(vdf.dropna(subset=[ab]),ab),use_container_width=True)
        with st.expander("❓ Reasons"):
            r = vdf["Loss Reason"].dropna().value_counts()
            if len(r)>0:
                vm = st.radio("View:",["Chart","Table"],horizontal=True,key=f"{kp}or")
                if vm=="Chart": st.pyplot(make_bar_horiz(r,f"OTR Reasons",color="crimson"))
                else: st.dataframe(make_table(r,"Reason","Count"),use_container_width=True)
    elif lf == "UTR Only":
        vdf = df[df["Type"]=="UTR"].copy(); st.write(f"**{len(vdf)} UTR** — {fmt_cost(vdf['Cost (£)'].sum())}")
        if len(vdf)==0: return
        with st.expander("🏷️ Sub Buckets"):
            sb = vdf["Sub Bucket"].value_counts()
            if len(sb)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}usb")
                if vm=="Chart": st.pyplot(make_bar_horiz(sb,f"UTR Sub Buckets",color="darkorange"))
                else: st.dataframe(make_cost_table(vdf,"Sub Bucket"),use_container_width=True)
        with st.expander("📍 By Cluster"):
            cl = vdf["Cluster"].dropna().value_counts()
            if len(cl)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}ul")
                if vm=="Chart": st.pyplot(make_bar_horiz(cl,f"UTR Clusters",color="darkorange"))
                else: st.dataframe(make_cost_table(vdf.dropna(subset=["Cluster"]),"Cluster"),use_container_width=True)
    else:
        vdf = df.copy(); st.write(f"**{len(vdf)} all** — {fmt_cost(vdf['Cost (£)'].sum())}")
        with st.expander("🏆 Top 10 Locations"):
            rb = st.selectbox("By:",["Cluster","Aisle","Sort Zone"],key=f"{kp}rb")
            rd = vdf[rb].dropna().value_counts().head(10)
            if len(rd)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}lv")
                if vm=="Chart": st.pyplot(make_bar_horiz(rd,f"Top 10 {rb}s ({dr})",color="darkred"))
                else: st.dataframe(make_cost_table(vdf.dropna(subset=[rb]),rb).head(10),use_container_width=True)
        with st.expander("🔍 Cluster Drill-Down"):
            clusters = sorted(vdf["Cluster"].dropna().unique())
            if clusters:
                sel = st.selectbox("Cluster:",clusters,key=f"{kp}cl"); filt = vdf[vdf["Cluster"]==sel]
                st.write(f"**{len(filt)}** in {sel} — {fmt_cost(filt['Cost (£)'].sum())}")
                ad = filt["Aisle"].dropna().value_counts()
                if len(ad)>0:
                    vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}cv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(ad,f"{sel} Aisles",color="steelblue"))
                    else: st.dataframe(make_cost_table(filt.dropna(subset=["Aisle"]),"Aisle"),use_container_width=True)

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
        st.dataframe(pd.DataFrame(rows,index=range(1,len(rows)+1)),use_container_width=True,height=200)
        if len(sc)>0: st.pyplot(make_bar_shift(sc,f"By Shift ({dr})"))
    with st.expander("🔍 Shift Drill-Down"):
        opts = [f"{s} — {int(sc.get(s,0))}" for s in SHIFT_ORDER]
        sel = st.selectbox("Shift:",opts,key=f"{kp}os"); ss = sel.split(" —")[0]
        sdf = df[df["Shift"]==ss]
        if len(sdf)>0:
            st.write(f"**{ss}: {len(sdf)} parcels** — {fmt_cost(sdf['Cost (£)'].sum())}")
            sbc = sdf["Sub Bucket"].value_counts()
            vm = st.radio("View:",["Table","Chart"],horizontal=True,key=f"{kp}sd")
            if vm=="Chart": st.pyplot(make_bar_horiz(sbc,f"{ss} Sub Buckets",color=SHIFT_COLORS.get(ss,"steelblue")))
            else: st.dataframe(make_table(sbc,"Sub Bucket","Count"),use_container_width=True)

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
        else: st.dataframe(csb,use_container_width=True)
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
            else: st.dataframe(dc,use_container_width=True)
    with st.expander("💰 Top 10 Most Expensive Parcels"):
        top = df.nlargest(10,"Cost (£)")
        sc2 = [c for c in ["Tracking ID","Sub Bucket","Type","Shift","Cost (£)","DSP Name","Cluster","Loss Reason"] if c in top.columns]
        out = top[sc2].reset_index(drop=True); out.index = range(1,len(out)+1); st.dataframe(out,use_container_width=True)

def render_analysis_tab(df, total, dr, kp=""):
    """Statistical root cause analysis — user-friendly with explanations."""
    st.markdown("### 🔬 Root Cause Analysis")
    st.markdown("""
> **What is this page?** This tab uses statistical methods to go beyond "what happened" and answer 
> **"why is it happening?"** Each test below looks at your data from a different angle to find 
> patterns that aren't obvious from charts alone. The results tell you where to focus your 
> problem-solving efforts for maximum impact.
""")

    total_cost = df["Cost (£)"].sum()
    findings = []

    # ─── DATA SUFFICIENCY (moved to top) ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Data Sufficiency — Can We Trust the Results?")
    st.markdown("""
> **Why this matters:** Statistical tests need a minimum amount of data to produce reliable conclusions. 
> Think of it like a survey — asking 3 people isn't enough to draw conclusions about a whole population. 
> The table below shows which tests have enough data to run. ❌ tests need more data before the results 
> are meaningful — the "How to Fix" column tells you what to upload.
""")

    cluster_ok = len(df["Cluster"].dropna().value_counts()) >= 2
    shift_ok = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].count() >= 20
    dsp_ok = len(df[df["Type"]=="OTR"]) >= 5
    day_ok = df["Day of Week"].dropna().count() >= 14 if "Day of Week" in df.columns else False

    checks = [
        ("📊 Concentration (Pareto)", "✅ Ready" if cluster_ok else "❌ Need 2+ clusters", 
         "Upload SCC data with cluster info" if not cluster_ok else "—"),
        ("⏰ Shift Significance", "✅ Ready" if shift_ok else f"❌ Need 20+ parcels (have {df[df['Shift'].isin(SHIFT_ORDER)]['Shift'].count()})", 
         "Upload more days of data" if not shift_ok else "—"),
        ("💰 Cost Disproportionality", "✅ Ready" if total_cost > 0 else "❌ No cost data", 
         "Ensure PM export includes shipment_value" if total_cost == 0 else "—"),
        ("🚚 DSP Outlier Detection", "✅ Ready" if dsp_ok else f"❌ Need 5+ OTR parcels (have {len(df[df['Type']=='OTR'])})", 
         "Upload more data or check OTR classification" if not dsp_ok else "—"),
        ("📅 Day-of-Week Pattern", "✅ Ready" if day_ok else f"❌ Need 14+ dated parcels (have {df['Day of Week'].dropna().count() if 'Day of Week' in df.columns else 0})", 
         "Upload a full 2-week period minimum" if not day_ok else "—"),
        ("📏 Size Analysis", "✅ Ready" if len(df["Size Category"].value_counts()) >= 2 else "❌ Need size data", 
         "Ensure SCC export includes Package Length/Width/Height" if len(df["Size Category"].value_counts()) < 2 else "—"),
        ("📈 Trend Analysis (over time)", "❌ Need 2+ weeks of data", "Upload data spanning at least 2 weeks"),
        ("🔮 Forecasting", "❌ Need 4+ weeks of data", "Upload data spanning at least 4 weeks"),
    ]

    ready_count = sum(1 for _,s,_ in checks if "✅" in s)
    st.markdown(f"**{ready_count}/{len(checks)}** tests have sufficient data to run.")
    st.dataframe(
        pd.DataFrame(checks, columns=["Test", "Status", "How to Fix"]),
        use_container_width=True, hide_index=True
    )
    if ready_count < len(checks):
        st.caption("💡 **Tip:** Upload a larger date range (2+ weeks) to unlock all tests and get stronger conclusions.")

    # ─── ANALYSIS RESULTS ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Analysis Results")
    st.markdown("""
> **How to read the results:** Each section below runs a specific test. Look for 🎯 markers — 
> these highlight **statistically significant** findings (i.e., patterns that are very unlikely 
> to be random chance). These are your highest-confidence action items.
""")

    # 1. CONCENTRATION (Pareto / Gini)
    with st.expander("📊 Test 1: Concentration — Are losses spread out or focused in a few spots?", expanded=True):
        st.markdown("""
**What this tells you:** Are your lost parcels spread evenly across the station, or concentrated 
in just a few locations? If losses are concentrated, you can get the biggest impact by fixing 
just those few problem areas rather than trying to fix everything at once.

**How to interpret:**
- **Gini coefficient** (0 to 1): 0 = perfectly equal (every cluster loses the same), 1 = all losses in one place. 
  Above 0.4 = "concentrated enough to target specific areas."
- **Top 3 clusters:** If these account for >50% of losses, start your problem-solving here.
- **80/20 rule:** Shows how many clusters account for 80% of all losses — fewer = more concentrated = easier to fix.
""")
        cl_c = df["Cluster"].dropna().value_counts()
        if len(cl_c) >= 2:
            cumsum = cl_c.cumsum(); t80 = cl_c.sum()*0.8
            c80 = len(cumsum[cumsum <= t80]) + 1; pct80 = round(c80/len(cl_c)*100,1)
            top3_pct = round(cl_c.head(3).sum()/cl_c.sum()*100,1)
            top3_cost = df[df["Cluster"].isin(cl_c.head(3).index)]["Cost (£)"].sum()
            vals = cl_c.values.astype(float); n = len(vals)
            sv = np.sort(vals); gini = (2*np.sum(np.arange(1,n+1)*sv)-(n+1)*np.sum(sv))/(n*np.sum(sv))
            conc = "highly concentrated" if gini>0.5 else "moderately concentrated" if gini>0.3 else "fairly spread out"
            
            # Verdict
            if gini > 0.5:
                st.error(f"🎯 **Losses are {conc}** (Gini: {gini:.2f}) — a small number of locations are driving most losses.")
            elif gini > 0.3:
                st.warning(f"⚠️ **Losses are {conc}** (Gini: {gini:.2f}) — some locations are worse than others.")
            else:
                st.info(f"ℹ️ **Losses are {conc}** (Gini: {gini:.2f}) — no single area dominates.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Top 3 Clusters", f"{top3_pct}% of losses", f"{fmt_cost(top3_cost)} cost")
                st.caption(f"Clusters: {', '.join(cl_c.head(3).index.tolist())}")
            with col2:
                st.metric("80% of losses come from", f"{c80} of {len(cl_c)} clusters", f"({pct80}% of locations)")
            
            st.markdown("**🔑 Action:** " + (
                f"Focus problem-solving on **{', '.join(cl_c.head(3).index.tolist())}** — fixing just these 3 clusters addresses {top3_pct}% of your losses ({fmt_cost(top3_cost)}). Walk these areas and look for: stow density issues, cage management gaps, or layout problems."
                if gini > 0.4 else
                "Losses are fairly spread — no single cluster is an obvious fix. Look at shift or DSP patterns instead."
            ))
            if gini > 0.4:
                findings.append(f"Losses concentrated in top 3 clusters ({', '.join(cl_c.head(3).index.tolist())}) = {top3_pct}% of all losses ({fmt_cost(top3_cost)}). Likely root cause: stow density, cage management, or physical layout in these areas.")
        else:
            st.warning("⚠️ Not enough cluster data (need 2+ different clusters). Check your SCC export includes the 'Cluster' column.")

    # 2. SHIFT SIGNIFICANCE (Chi-squared)
    with st.expander("⏰ Test 2: Shift Significance — Is one shift losing more than expected?"):
        st.markdown("""
**What this tells you:** If parcels were being lost purely by random chance, you'd expect roughly 
equal numbers across all shifts. This test checks: **is one shift significantly worse, or is the 
variation just normal randomness?**

**How to interpret:**
- **p-value** (0 to 1): The probability that the difference is due to random chance alone.
  - **p < 0.05** = Less than 5% chance it's random → 🎯 **Significant! Something is actually different about that shift.**
  - **p > 0.05** = Could easily be random variation → No action needed on shift-specific processes.
- **Observed vs Expected:** Shows how many losses each shift had vs. how many you'd expect if losses were random.
""")
        shift_counts = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER,fill_value=0)
        assigned = shift_counts.sum()
        if assigned >= 20:
            expected = np.array([assigned/4]*4); observed = np.array([shift_counts[s] for s in SHIFT_ORDER])
            chi2, p = sp_stats.chisquare(observed, f_exp=expected)
            worst_s = shift_counts.idxmax(); worst_n = int(shift_counts.max())
            
            if p < 0.05:
                st.error(f"🎯 **Statistically significant** (p = {p:.4f}) — the difference between shifts is NOT random.")
                st.markdown(f"**{worst_s} shift is the problem** — {worst_n} losses ({round(worst_n/assigned*100,1)}%) vs expected {int(assigned/4)} if equal. "
                           f"Window: {SHIFT_DEFINITIONS[worst_s]}")
                findings.append(f"Shift imbalance is statistically significant (p={p:.4f}). {worst_s} has {worst_n} losses ({round(worst_n/assigned*100,1)}%) vs expected {int(assigned/4)}. Not random — process gap in {worst_s} shift ({SHIFT_DEFINITIONS[worst_s]}).")
            else:
                st.success(f"✅ **Not significant** (p = {p:.2f}) — shift differences are within normal random variation.")
                st.markdown("No single shift stands out. The variation you see is likely just natural randomness, not a process problem.")
            
            tbl = pd.DataFrame({"Shift":SHIFT_ORDER,"Observed":observed.astype(int),"Expected (if equal)":expected.astype(int),"Difference":(observed-expected).astype(int)})
            tbl.index = range(1,5); st.dataframe(tbl,use_container_width=True)
            
            st.markdown("**🔑 Action:** " + (
                f"Investigate what's different during **{worst_s}** ({SHIFT_DEFINITIONS[worst_s]}). "
                f"Consider: staffing levels, process compliance, handover quality, volume spikes during this window."
                if p < 0.05 else
                "No shift-specific intervention needed — focus efforts elsewhere (e.g., location or DSP)."
            ))
        else:
            st.warning(f"⚠️ Not enough data — need 20+ parcels with shift assignments (currently have {assigned}). Upload more data.")

    # 3. COST DISPROPORTIONALITY
    with st.expander("💰 Test 3: Cost Disproportionality — Are high-value items lost in specific ways?"):
        st.markdown("""
**What this tells you:** Some loss types might affect more expensive parcels than others. This test 
checks whether certain sub-buckets are losing **disproportionately expensive** items — meaning the 
*financial* impact is bigger than the *count* suggests.

**How to interpret:**
- **Ratio** column: Compares the % of total cost vs % of total count for each loss type.
  - **Ratio = 1.0** = Average (costs proportional to count)
  - **Ratio > 1.5** = 🎯 This type loses items that cost 50%+ more than average — high priority!
  - **Ratio < 0.7** = Loses cheaper items (still bad, but less £ impact per parcel)
""")
        sb_s = df.groupby("Sub Bucket").agg(Count=("Tracking ID","count"),Cost=("Cost (£)","sum")).reset_index()
        if len(sb_s)>=2 and total_cost>0:
            sb_s["% of Total Count"] = (sb_s["Count"]/total*100).round(1)
            sb_s["% of Total Cost"] = (sb_s["Cost"]/total_cost*100).round(1)
            sb_s["Ratio (Cost/Count)"] = (sb_s["% of Total Cost"]/sb_s["% of Total Count"]).round(2)
            sb_s["Avg Cost/Parcel"] = (sb_s["Cost"]/sb_s["Count"]).round(2)
            sb_s = sb_s.sort_values("Ratio (Cost/Count)",ascending=False)
            high = sb_s[sb_s["Ratio (Cost/Count)"]>1.5]
            
            if len(high)>0:
                st.error(f"🎯 **{len(high)} loss type(s)** are disproportionately hitting high-value parcels:")
                for _,row in high.iterrows():
                    st.markdown(f"- **{row['Sub Bucket']}**: Only {row['% of Total Count']}% of lost parcels but {row['% of Total Cost']}% of cost "
                               f"(avg {fmt_cost(row['Avg Cost/Parcel'])} vs {fmt_cost(total_cost/total)} overall)")
                    findings.append(f"{row['Sub Bucket']}: {row['% of Total Count']}% of losses but {row['% of Total Cost']}% of cost (avg {fmt_cost(row['Avg Cost/Parcel'])} vs {fmt_cost(total_cost/total)} overall). High-value items disproportionately lost here.")
            else:
                st.success("✅ No disproportionate cost impact — all loss types affect similarly-valued parcels.")
            
            display = sb_s[["Sub Bucket","Count","% of Total Count","% of Total Cost","Ratio (Cost/Count)","Avg Cost/Parcel"]].copy()
            display["Avg Cost/Parcel"] = display["Avg Cost/Parcel"].apply(fmt_cost)
            display.index = range(1,len(display)+1)
            st.dataframe(display,use_container_width=True)
            st.caption("📖 **Reading the table:** Ratio > 1.5 means that loss type costs 50%+ more per parcel than the station average. These are your highest-value targets for cost reduction.")
            
            st.markdown("**🔑 Action:** " + (
                "Prioritise preventing " + ", ".join(high["Sub Bucket"].tolist()) + " — even small reductions here save more money per parcel than other types."
                if len(high) > 0 else
                "Cost impact is proportional — prioritise by volume (most frequent loss type) rather than per-parcel value."
            ))
        else:
            st.warning("⚠️ Need 2+ sub-buckets with cost data to run this test.")

    # 4. DSP OUTLIERS (Z-score)
    with st.expander("🚚 Test 4: DSP Outlier Detection — Is any delivery partner losing far more than others?"):
        st.markdown("""
**What this tells you:** Among all DSPs (Delivery Service Partners) handling your parcels, is any one 
losing **significantly more** than average? This uses a z-score — essentially measuring how many 
"standard deviations" above the mean each DSP is.

**How to interpret:**
- **Z-score > 1.5** = 🎯 This DSP is an **outlier** — losing much more than peer DSPs. Worth investigating.
- **Z-score 0-1.5** = Within normal range — some variation is expected.
- **Mean & Std:** The average losses per DSP and how spread out they are. A high std means big differences between DSPs.

**Think of it this way:** If the average DSP loses 5 parcels and the standard deviation is 2, a DSP losing 
10 parcels (z-score = 2.5) is unusual enough to warrant investigation.
""")
        otr = df[df["Type"]=="OTR"]
        if len(otr)>=5:
            dc = otr["DSP Name"].dropna().value_counts()
            if len(dc)>=3:
                mu = dc.mean(); sigma = dc.std()
                z = (dc-mu)/sigma if sigma>0 else pd.Series(0,index=dc.index)
                outliers = z[z>1.5]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("DSPs Active", len(dc))
                col2.metric("Avg Losses/DSP", f"{mu:.1f}")
                col3.metric("Std Deviation", f"{sigma:.1f}")
                
                tbl = pd.DataFrame({"DSP":dc.index,"Losses":dc.values,"Z-Score":z.values.round(2)})
                tbl["Flag"] = tbl["Z-Score"].apply(lambda x: "🎯 OUTLIER" if x>1.5 else "✅ Normal")
                tbl.index = range(1,len(tbl)+1)
                st.dataframe(tbl,use_container_width=True)
                
                if len(outliers)>0:
                    st.error(f"🎯 **{len(outliers)} DSP outlier(s) detected** — losing far more than peers:")
                    for dsp,zv in outliers.items():
                        cost = otr[otr["DSP Name"]==dsp]["Cost (£)"].sum()
                        reason = otr[otr["DSP Name"]==dsp]["Loss Reason"].dropna().value_counts()
                        r_str = reason.index[0] if len(reason)>0 else "Unknown"
                        st.markdown(f"- **{dsp}** — {int(dc[dsp])} losses (z={zv:.1f}), costing {fmt_cost(cost)}. Top reason: _{r_str}_")
                        findings.append(f"DSP '{dsp}' is a statistical outlier (z={zv:.1f}): {int(dc[dsp])} losses ({fmt_cost(cost)}). Top reason: {r_str}.")
                    st.markdown("**🔑 Action:** Raise these DSPs in the next DSP performance review. Request root cause from DSP management — is it a specific driver, route, or vehicle issue?")
                else:
                    st.success("✅ **No DSP outliers** — OTR losses are spread fairly evenly across delivery partners.")
                    st.markdown("**🔑 Action:** OTR losses aren't a DSP-specific problem — look at route difficulty, package type, or time-of-day patterns instead.")
            else:
                st.warning("⚠️ Need 3+ DSPs with losses to detect outliers meaningfully.")
        else:
            st.warning(f"⚠️ Need 5+ OTR (on-road) parcels to run this test (currently have {len(otr)}).")

    # 5. DAY-OF-WEEK PATTERN
    with st.expander("📅 Test 5: Day-of-Week Pattern — Do losses spike on certain days?"):
        st.markdown("""
**What this tells you:** Are there days when significantly more parcels go missing? If so, it could 
point to staffing patterns, volume spikes, or handover issues on specific days.

**How to interpret:**
- Same as the shift test — uses a chi-squared test comparing actual losses per day vs. what you'd 
  expect if losses were spread evenly (total ÷ 7).
- **p < 0.05** = 🎯 The day pattern is real, not random.
- **p > 0.05** = Day-to-day variation is just noise.
""")
        if "Day of Week" in df.columns:
            dc2 = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER,fill_value=0); dt = dc2.sum()
            if dt>=14:
                exp = np.array([dt/7]*7); obs = np.array([dc2[d] for d in DAY_ORDER])
                chi2d, pd2 = sp_stats.chisquare(obs,f_exp=exp)
                
                if pd2<0.05:
                    wd = dc2.idxmax(); wn = int(dc2.max())
                    st.error(f"🎯 **Statistically significant** (p = {pd2:.4f}) — day-of-week pattern is real.")
                    st.markdown(f"**{wd}** is the worst day: {wn} losses vs expected {dt/7:.0f}.")
                    findings.append(f"Day variation significant (p={pd2:.3f}). {wd} has {wn} losses vs expected {dt/7:.0f}. Something different on {wd}s (staffing, volume, handover).")
                    st.markdown(f"**🔑 Action:** Investigate what's different on **{wd}s** — volume spike? Different staffing? New process? Handover issues from prior day?")
                else:
                    st.success(f"✅ **Not significant** (p = {pd2:.2f}) — losses are evenly spread across the week.")
                    st.markdown("**🔑 Action:** No day-specific intervention needed.")
                
                tbl = pd.DataFrame({"Day":DAY_ORDER,"Losses":[int(dc2[d]) for d in DAY_ORDER],"Expected":[int(dt/7)]*7})
                tbl["Difference"] = tbl["Losses"] - tbl["Expected"]
                tbl.index = range(1,8); st.dataframe(tbl,use_container_width=True)
            else:
                st.warning(f"⚠️ Need 14+ parcels with dates (currently have {dt}). Upload at least 2 weeks of data.")
        else:
            st.warning("⚠️ No date information available in the data.")

    # 6. SIZE ANALYSIS
    with st.expander("📏 Test 6: Package Size — Are oversized items more likely to be lost?"):
        st.markdown("""
**What this tells you:** Larger parcels may be harder to stow correctly, more likely to fall off 
shelves, or more prone to misplacement. This checks whether oversized items are **over-represented** 
in your loss data compared to what you'd normally expect (~15-20% of volume).

**How to interpret:**
- If oversized parcels make up >30% of losses, there's likely a physical handling/stow problem.
- Compare against your station's overall size mix (from SCC data) if available.
""")
        sc3 = df["Size Category"].value_counts()
        if len(sc3)>=2:
            ov = sc3.get("Small Oversize",0)+sc3.get("Large Oversize",0); ov_pct = round(ov/total*100,1)
            ov_cost = df[df["Size Category"].isin(["Small Oversize","Large Oversize"])]["Cost (£)"].sum()
            
            if ov_pct > 30:
                st.error(f"🎯 **Oversized parcels = {ov_pct}%** of losses — well above the typical 15-20% station mix.")
                st.markdown(f"**{ov} oversized parcels** lost, costing **{fmt_cost(ov_cost)}**")
                findings.append(f"Oversized parcels = {ov_pct}% of losses (above typical ~15-20%). Large items may not fit standard stow, increasing loss risk.")
                st.markdown("**🔑 Action:** Check oversize stow areas — are they full? Damaged? Poorly labelled? Consider dedicated oversize zones or cage assignments.")
            elif ov_pct > 20:
                st.warning(f"⚠️ **Oversized parcels = {ov_pct}%** — slightly elevated. Worth monitoring.")
                st.markdown(f"**{ov} oversized parcels** lost, costing **{fmt_cost(ov_cost)}**")
            else:
                st.success(f"✅ **Oversized parcels = {ov_pct}%** — within normal range.")
            
            stbl = df.groupby("Size Category").agg(Count=("Tracking ID","count"),Cost=("Cost (£)","sum")).sort_values("Count",ascending=False).reset_index()
            stbl["% of Total"] = (stbl["Count"]/total*100).round(1)
            stbl["Cost"] = stbl["Cost"].apply(fmt_cost); stbl.index = range(1,len(stbl)+1)
            st.dataframe(stbl,use_container_width=True)
        else:
            st.warning("⚠️ Need package dimension data in SCC export (Package Length/Width/Height columns).")

    # ─── FINDINGS SUMMARY ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📝 Key Findings & Recommended Actions")
    st.markdown("""
> **These are your statistically-backed conclusions.** Each finding below is supported by the data 
> and is unlikely to be due to random chance. Use these to prioritise your problem-solving.
""")
    if findings:
        for i,f in enumerate(findings,1):
            st.markdown(f"**{i}.** {f}")
        st.markdown("---")
        st.markdown("#### 💡 What To Do Next")
        st.markdown("""
1. **Pick the highest-impact finding** (biggest £ value or easiest to fix)
2. **Go and observe** — walk the area, watch the process, talk to the team
3. **Identify the specific failure point** — where exactly in the process does the parcel get lost?
4. **Implement a countermeasure** — process change, visual management, training, physical fix
5. **Measure again** — upload new data in 1-2 weeks to see if it improved
""")
    else:
        st.info("ℹ️ No statistically significant root causes found with current data. This could mean:")
        st.markdown("""
- Losses are genuinely random (no single pattern dominates)
- The dataset is too small to detect patterns — **upload a larger date range** (2+ weeks recommended)
- Root causes may be at a more granular level than this analysis checks — use the other tabs for manual exploration
""")

def generate_bridge(df, total, dr):
    tc = df["Cost (£)"].sum(); avg = tc/total if total>0 else 0
    otr = df[df["Type"]=="OTR"]; utr = df[df["Type"]=="UTR"]
    sh = df[df["Shift"]!="Unknown"]["Shift"].value_counts(); cl = df["Cluster"].dropna().value_counts()
    sb = df["Sub Bucket"].value_counts()
    lines = [f"LOST PARCELS BRIDGE — DRM2",f"Period: {dr}","",
        f"TOTAL: {total} parcels — {fmt_cost(tc)} (avg {fmt_cost(avg)}/parcel)",
        f"UTR (At Station): {len(utr)} ({round(len(utr)/total*100,1)}%) — {fmt_cost(utr['Cost (£)'].sum())}",
        f"OTR (On Road): {len(otr)} ({round(len(otr)/total*100,1)}%) — {fmt_cost(otr['Cost (£)'].sum())}","",
        "SHIFTS:"]
    for s in SHIFT_ORDER:
        sd = df[df["Shift"]==s]; n = len(sd)
        lines.append(f"  {s}: {n} ({round(n/total*100,1)}%) — {fmt_cost(sd['Cost (£)'].sum())}")
    lines += ["","SUB BUCKETS:"]
    for s,n in sb.head(6).items(): lines.append(f"  {s}: {n} ({round(int(n)/total*100,1)}%) — {fmt_cost(df[df['Sub Bucket']==s]['Cost (£)'].sum())}")
    lines += ["","LOCATIONS (Top 3):"]
    for cn,cv in cl.head(3).items():
        cc = df[df["Cluster"]==cn]["Cost (£)"].sum()
        ta = df[df["Cluster"]==cn]["Aisle"].dropna().value_counts().head(3)
        lines.append(f"  {cn}: {cv} ({round(int(cv)/total*100,1)}%) — {fmt_cost(cc)} — {', '.join([f'{a}({n})' for a,n in ta.items()])}")
    lines += ["","STATISTICAL FINDINGS:"]
    cl_c = df["Cluster"].dropna().value_counts()
    if len(cl_c)>=3:
        top3p = round(cl_c.head(3).sum()/cl_c.sum()*100,1)
        lines.append(f"  Concentration: Top 3 clusters = {top3p}% of losses")
    sc2 = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER,fill_value=0)
    if sc2.sum()>=20:
        exp = np.array([sc2.sum()/4]*4); obs = np.array([sc2[s] for s in SHIFT_ORDER])
        _,p = sp_stats.chisquare(obs,f_exp=exp)
        ws = sc2.idxmax()
        lines.append(f"  Shift test: {ws} dominant (p={p:.4f}, {'significant' if p<0.05 else 'not significant'})")
    if len(df[df["Type"]=="OTR"])>=5:
        dc3 = df[df["Type"]=="OTR"]["DSP Name"].dropna().value_counts()
        if len(dc3)>=3:
            mu = dc3.mean(); sig = dc3.std()
            outs = dc3[(dc3-mu)/sig > 1.5] if sig>0 else pd.Series(dtype=float)
            if len(outs)>0: lines.append(f"  DSP outliers: {', '.join(outs.index.tolist())}")
    return "\n".join(lines)

# ─── MAIN ──────────────────────────────────────────────────────────────────
mode = st.radio("Mode:",["Single Station","Multi-Station"],horizontal=True,key="mode")
with st.expander("📖 How to get data"):
    st.markdown("1. PerfectMile → L&U → Lost → Export\n2. SCC → paste TIDs → Export\n3. Upload both")
if mode == "Single Station":
    c_pm, c_scc = st.columns(2)
    with c_pm: pm_file = st.file_uploader("📊 Perfect Mile",type="csv",key="pm")
    with c_scc: scc_file = st.file_uploader("📋 SCC",type="csv",key="scc")
    if pm_file and scc_file:
        pm_df,scc_df = pd.read_csv(pm_file),pd.read_csv(scc_file)
        pm_miss = [c for c in REQUIRED_PM_COLS if c not in pm_df.columns]
        if pm_miss: st.error(f"❌ PM missing: {pm_miss}"); st.stop()
        scc_miss = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
        if scc_miss: st.error(f"❌ SCC missing: {scc_miss}"); st.stop()
        found = [c for c in SENSITIVE_COLS if c in scc_df.columns]
        if found: st.warning(f"🔒 PII removed: {', '.join(found)}")
        df = merge_data(pm_df,scc_df); total = len(df)
        if total==0: st.stop()
        matched = df["Cluster"].notna().sum(); tc = df["Cost (£)"].sum()
        st.success(f"✅ **{total} parcels** — {fmt_cost(tc)} (PM:{len(pm_df)}, SCC:{len(scc_df)}, Matched:{matched})")
        render_missing_parcels(df,total,matched)
        dr = get_date_range(df)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Lost",total); c2.metric("Cost",fmt_cost(tc)); c3.metric("Cluster",safe_top(df["Cluster"]))
        c4.metric("Aisle",safe_top(df["Aisle"])); c5.metric("DSP",str(safe_top(df["DSP Name"]))[:15])
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]; c6.metric("Shift",safe_top(sk) if len(sk)>0 else "N/A")
        t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs(["📊 Summary","📍 Locations","💡 Shifts","💰 Cost","🔬 Analysis","📅 Day","💾 Export","📋 Bridge"])
        with t1:
            with st.expander("🥧 OTR vs UTR"): st.pyplot(make_pie_otr_utr(df,total,f"OTR vs UTR ({dr})"))
            with st.expander("📍 Clusters"):
                cc = df["Cluster"].dropna().value_counts()
                if len(cc)>0:
                    vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key="cv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(cc,f"Clusters ({dr})"))
                    else: st.dataframe(make_cost_table(df.dropna(subset=["Cluster"]),"Cluster"),use_container_width=True)
            with st.expander("🏷️ Sub Buckets"):
                sb2 = df["Sub Bucket"].value_counts()
                if len(sb2)>0:
                    vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key="sv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(sb2,f"Sub Buckets ({dr})",color="teal"))
                    else: st.dataframe(make_cost_table(df,"Sub Bucket"),use_container_width=True)
        with t2: render_locations_tab(df,total,dr,kp="s_")
        with t3: render_opportunities_tab(df,total,dr,kp="s_")
        with t4: render_cost_tab(df,total,dr,kp="s_")
        with t5: render_analysis_tab(df,total,dr,kp="s_")
        with t6:
            st.caption("Day = date marked lost in Perfect Mile (event_datetime)")
            if "Day of Week" in df.columns:
                dd = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER,fill_value=0)
                with st.expander("📅 Day of Week"):
                    vm = st.radio("View:",["Chart","Table"],horizontal=True,key="dv")
                    if vm=="Chart":
                        fig,ax = plt.subplots(figsize=CHART)
                        ax.plot(dd.index,dd.values,marker="o",color="green",linewidth=2)
                        for i,(d,v) in enumerate(dd.items()): ax.annotate(str(int(v)),xy=(i,v),xytext=(0,8),textcoords="offset points",ha="center",fontsize=8)
                        ax.set_ylabel("Lost",fontsize=8); ax.set_title(f"By Day ({dr})",fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
                    else: st.dataframe(make_table(dd,"Day","Lost"),use_container_width=True)
        with t7:
            exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","Marked Lost DT","shipment_value"]
            ec = [c for c in df.columns if c not in exc]; st.download_button("⬇️ CSV",df[ec].to_csv(index=False),"Lost.csv","text/csv")
        with t8:
            st.text_area("📋 Bridge (auto-generated):",value=generate_bridge(df,total,dr),height=500,key="bridge")
    else: st.info("👆 Upload both files.")
else:
    num = st.slider("Stations:",2,5,2,key="ns"); uploaded = {}
    for i in range(num):
        with st.expander(f"Station {i+1}",expanded=(i<2)):
            a,b = st.columns(2)
            with a: pf = st.file_uploader(f"PM({i+1})",type="csv",key=f"mp{i}")
            with b: sf = st.file_uploader(f"SCC({i+1})",type="csv",key=f"ms{i}")
            if pf and sf: uploaded[i] = (pf,sf)
    if len(uploaded)>=2:
        stations,names = {},[]
        for i,(pf,sf) in uploaded.items():
            pt,s2 = pd.read_csv(pf),pd.read_csv(sf); m = merge_data(pt,s2)
            nm = f"Station {i+1}"
            if "location" in pt.columns and len(pt["location"].dropna())>0: nm = pt["location"].dropna().iloc[0]
            stations[nm] = m; names.append(nm)
        st.success(f"✅ {', '.join(names)}")
        t1,t2,t3,t4,t5 = st.tabs(["📊 Summary","📍 Locations","💰 Cost","🔬 Analysis","💾 Export"])
        with t1:
            for n in names: st.caption(f"{n}: {len(stations[n])} — {fmt_cost(stations[n]['Cost (£)'].sum())}")
        with t2:
            sel = st.selectbox("Station:",names,key="mcl"); render_locations_tab(stations[sel],len(stations[sel]),get_date_range(stations[sel]),kp=f"m{sel}")
        with t3:
            sel = st.selectbox("Station:",names,key="mcc"); render_cost_tab(stations[sel],len(stations[sel]),get_date_range(stations[sel]),kp=f"mc{sel}")
        with t4:
            sel = st.selectbox("Station:",names,key="mca"); render_analysis_tab(stations[sel],len(stations[sel]),get_date_range(stations[sel]),kp=f"ma{sel}")
        with t5:
            for n in names:
                exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","Marked Lost DT","shipment_value"]
                ec = [c for c in stations[n].columns if c not in exc]; st.download_button(f"⬇️ {n}",stations[n][ec].to_csv(index=False),f"{n}.csv","text/csv",key=f"dl{n}")
    elif len(uploaded)==1: st.warning("Need 2+.")
    else: st.info("👆 Upload pairs.")
