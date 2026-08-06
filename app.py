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
                if vm=="Chart": st.pyplot(make_bar_horiz(r,"OTR Reasons",color="crimson"))
                else: st.dataframe(make_table(r,"Reason","Count"),use_container_width=True)
    elif lf == "UTR Only":
        vdf = df[df["Type"]=="UTR"].copy(); st.write(f"**{len(vdf)} UTR** — {fmt_cost(vdf['Cost (£)'].sum())}")
        if len(vdf)==0: return
        with st.expander("🏷️ Sub Buckets"):
            sb = vdf["Sub Bucket"].value_counts()
            if len(sb)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}usb")
                if vm=="Chart": st.pyplot(make_bar_horiz(sb,"UTR Sub Buckets",color="darkorange"))
                else: st.dataframe(make_cost_table(vdf,"Sub Bucket"),use_container_width=True)
        with st.expander("📍 By Cluster"):
            cl = vdf["Cluster"].dropna().value_counts()
            if len(cl)>0:
                vm = st.radio("View:",["Chart","Table + Cost"],horizontal=True,key=f"{kp}ul")
                if vm=="Chart": st.pyplot(make_bar_horiz(cl,"UTR Clusters",color="darkorange"))
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
    """Simple, user-friendly root cause analysis with bridge summary."""
    st.markdown("### 🔬 Analysis & Bridge")

    # ═══════════════════════════════════════════════════════════════════════
    # DISCLAIMER
    # ═══════════════════════════════════════════════════════════════════════
    st.warning(
        "⚠️ This section helps you process the vast amount of data and offers some suggestions, "
        "but **SHOULD NOT BE USED AS A GUIDE, IT IS ONLY AN AID.** "
        "Always use your own judgement and local knowledge when deciding on actions."
    )

    total_cost = df["Cost (£)"].sum()
    findings = []

    # ═══════════════════════════════════════════════════════════════════════
    # DATA SUFFICIENCY
    # ═══════════════════════════════════════════════════════════════════════
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
    ]

    ready = sum(1 for _, s, _ in checks if "✅" in s)
    if ready == len(checks):
        st.success(f"✅ All {len(checks)} tests ready!")
    else:
        st.info(f"{ready}/{len(checks)} tests ready. Upload more data to unlock the rest.")

    check_df = pd.DataFrame(checks, columns=["Question", "Status", "How to fix"])
    st.dataframe(check_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 🎯 What the data suggests")
    st.caption("Look for 🎯 = key finding. These are patterns in the data, not instructions.")

    # ─── 1. WHERE ─────────────────────────────────────────────────────────
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
                st.error(f"🎯 Losses are piling up in a few spots. Top 3 clusters = **{top3_pct}%** of all losses.")
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

    # ─── 2. SHIFT ─────────────────────────────────────────────────────────
    with st.expander("⏰ 2. Is one shift losing more than it should?"):
        shift_counts = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
        assigned = shift_counts.sum()
        if assigned >= 20:
            expected_per_shift = assigned / 4
            observed = np.array([shift_counts[s] for s in SHIFT_ORDER])
            expected = np.array([expected_per_shift] * 4)
            chi2, p = sp_stats.chisquare(observed, f_exp=expected)
            worst_s = shift_counts.idxmax()
            worst_n = int(shift_counts.max())

            tbl = pd.DataFrame({
                "Shift": SHIFT_ORDER,
                "Actual": observed.astype(int),
                "Expected": expected.astype(int),
                "Over/Under": (observed - expected).astype(int)
            })
            tbl.index = range(1, 5)
            st.dataframe(tbl, use_container_width=True)

            if p < 0.05:
                st.error(f"🎯 **{worst_s}** shift has {worst_n} losses vs ~{int(expected_per_shift)} expected. This pattern is unlikely to be random.")
                st.caption(f"Window: {SHIFT_DEFINITIONS[worst_s]}")
                findings.append(f"{worst_s} shift has {worst_n} losses vs expected {int(expected_per_shift)} — not random")
            else:
                st.success(f"✅ Shift differences look like normal variation. No single shift stands out.")
        else:
            st.warning(f"Need 20+ parcels with shift data (have {assigned}).")

    # ─── 3. COST ──────────────────────────────────────────────────────────
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
                st.success("✅ All loss types hit similarly-priced parcels. No standout.")

            display = sb_s[["Sub Bucket", "Count", "Avg £/parcel"]].copy()
            display["Avg £/parcel"] = display["Avg £/parcel"].apply(fmt_cost)
            display.index = range(1, len(display)+1)
            st.dataframe(display, use_container_width=True)
            st.caption(f"Station average: {fmt_cost(overall_avg)} per lost parcel")
        else:
            st.warning("Need cost data and 2+ loss types.")

    # ─── 4. DSP ───────────────────────────────────────────────────────────
    with st.expander("🚚 4. Is any DSP losing way more than others?"):
        otr = df[df["Type"] == "OTR"]
        if len(otr) >= 5:
            dc = otr["DSP Name"].dropna().value_counts()
            if len(dc) >= 3:
                mu = dc.mean(); sigma = dc.std()
                if sigma > 0:
                    z = (dc - mu) / sigma
                else:
                    z = pd.Series(0, index=dc.index)
                outliers = z[z > 1.5]

                col1, col2 = st.columns(2)
                col1.metric("Avg losses/DSP", f"{mu:.1f}")
                col2.metric("DSPs tracked", len(dc))

                tbl = pd.DataFrame({
                    "DSP": dc.index,
                    "Losses": dc.values,
                    "Verdict": ["🎯 Well above average" if x > 1.5 else "✅ Normal" for x in z.values]
                })
                tbl.index = range(1, len(tbl)+1)
                st.dataframe(tbl, use_container_width=True)

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

    # ─── 5. DAY ───────────────────────────────────────────────────────────
    with st.expander("📅 5. Do losses spike on certain days?"):
        if "Day of Week" in df.columns:
            dc2 = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
            dt = dc2.sum()
            if dt >= 14:
                exp = np.array([dt/7]*7)
                obs = np.array([dc2[d] for d in DAY_ORDER])
                chi2d, pd2 = sp_stats.chisquare(obs, f_exp=exp)

                tbl = pd.DataFrame({
                    "Day": DAY_ORDER,
                    "Losses": [int(dc2[d]) for d in DAY_ORDER],
                    "Expected": [int(dt/7)]*7
                })
                tbl.index = range(1, 8)
                st.dataframe(tbl, use_container_width=True)

                if pd2 < 0.05:
                    wd = dc2.idxmax(); wn = int(dc2.max())
                    st.error(f"🎯 **{wd}** is the worst day: {wn} losses vs ~{int(dt/7)} expected.")
                    findings.append(f"{wd} has {wn} losses vs expected {int(dt/7)}")
                else:
                    st.success("✅ No day stands out — losses spread evenly across the week.")
            else:
                st.warning(f"Need 14+ dated parcels (have {dt}).")
        else:
            st.warning("No date data available.")

    # ─── 6. SIZE ──────────────────────────────────────────────────────────
    with st.expander("📏 6. Are big parcels getting lost more?"):
        sc3 = df["Size Category"].value_counts()
        if len(sc3) >= 2:
            ov = sc3.get("Small Oversize", 0) + sc3.get("Large Oversize", 0)
            ov_pct = round(ov / total * 100, 1)
            ov_cost = df[df["Size Category"].isin(["Small Oversize", "Large Oversize"])]["Cost (£)"].sum()

            stbl = df.groupby("Size Category").agg(
                Count=("Tracking ID", "count"), Cost=("Cost (£)", "sum")
            ).sort_values("Count", ascending=False).reset_index()
            stbl["% of losses"] = (stbl["Count"] / total * 100).round(1)
            stbl["Cost"] = stbl["Cost"].apply(fmt_cost)
            stbl.index = range(1, len(stbl)+1)
            st.dataframe(stbl, use_container_width=True)

            if ov_pct > 30:
                st.error(f"🎯 Oversized = **{ov_pct}%** of losses (normal ~15-20%). Costing {fmt_cost(ov_cost)}.")
                findings.append(f"Oversized parcels = {ov_pct}% of losses ({fmt_cost(ov_cost)})")
            elif ov_pct > 20:
                st.warning(f"⚠️ Oversized = {ov_pct}% — slightly elevated.")
            else:
                st.success(f"✅ Oversized = {ov_pct}% — within normal range.")
        else:
            st.warning("Need package dimension data.")

    # ─── 7. REPEAT OFFENDER AISLES ────────────────────────────────────────
    with st.expander("🔁 7. Any repeat offender aisles? (same aisle losing multiple parcels)"):
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
                st.dataframe(repeat_df, use_container_width=True)
                top_aisle = repeat.index[0]
                findings.append(f"Aisle {top_aisle} lost {int(repeat.iloc[0])} parcels — repeat offender")
            else:
                st.success("✅ No single aisle has 3+ losses. Losses are spread out.")
        else:
            st.warning("Not enough aisle data.")

    # ─── 8. UTR vs OTR SPLIT ─────────────────────────────────────────────
    with st.expander("🏠🚚 8. Station (UTR) vs Road (OTR) — where's the bigger problem?"):
        otr_n = len(df[df["Type"]=="OTR"]); utr_n = len(df[df["Type"]=="UTR"])
        otr_cost = df[df["Type"]=="OTR"]["Cost (£)"].sum()
        utr_cost = df[df["Type"]=="UTR"]["Cost (£)"].sum()
        if otr_n + utr_n > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🏠 UTR (At Station)", f"{utr_n} parcels", fmt_cost(utr_cost))
                st.caption(f"{round(utr_n/(otr_n+utr_n)*100,1)}% of losses")
            with col2:
                st.metric("🚚 OTR (On Road)", f"{otr_n} parcels", fmt_cost(otr_cost))
                st.caption(f"{round(otr_n/(otr_n+utr_n)*100,1)}% of losses")

            if utr_n > otr_n * 1.5:
                st.error("🎯 Station losses (UTR) dominate. Focus on in-station processes.")
                findings.append(f"UTR dominates: {utr_n} vs {otr_n} OTR — station processes are the bigger issue")
            elif otr_n > utr_n * 1.5:
                st.error("🎯 Road losses (OTR) dominate. Focus on DSPs and dispatch.")
                findings.append(f"OTR dominates: {otr_n} vs {utr_n} UTR — road/DSP issues are bigger")
            else:
                st.info("ℹ️ Fairly balanced between station and road losses.")
        else:
            st.warning("No type data available.")

    # ═══════════════════════════════════════════════════════════════════════
    # BRIDGE SUMMARY (replaces old Bridge tab)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 📋 Bridge Summary")
    st.caption("Auto-generated overview you can copy/paste into a bridge or handover doc.")

    otr_df = df[df["Type"]=="OTR"]; utr_df = df[df["Type"]=="UTR"]
    cl = df["Cluster"].dropna().value_counts()
    sb = df["Sub Bucket"].value_counts()

    bridge_lines = []
    bridge_lines.append(f"LOST PARCELS — DRM2 | Period: {dr}")
    bridge_lines.append(f"TOTAL: {total} parcels — {fmt_cost(total_cost)} (avg {fmt_cost(total_cost/total if total>0 else 0)}/parcel)")
    bridge_lines.append(f"UTR (Station): {len(utr_df)} ({round(len(utr_df)/total*100,1)}%) — {fmt_cost(utr_df['Cost (£)'].sum())}")
    bridge_lines.append(f"OTR (Road): {len(otr_df)} ({round(len(otr_df)/total*100,1)}%) — {fmt_cost(otr_df['Cost (£)'].sum())}")
    bridge_lines.append("")
    bridge_lines.append("SHIFTS:")
    for s in SHIFT_ORDER:
        sd = df[df["Shift"]==s]; n = len(sd)
        bridge_lines.append(f"  {s}: {n} ({round(n/total*100,1)}%) — {fmt_cost(sd['Cost (£)'].sum())}")
    bridge_lines.append("")
    bridge_lines.append("TOP SUB BUCKETS:")
    for s_name, n in sb.head(5).items():
        bridge_lines.append(f"  {s_name}: {n} ({round(int(n)/total*100,1)}%) — {fmt_cost(df[df['Sub Bucket']==s_name]['Cost (£)'].sum())}")
    bridge_lines.append("")
    bridge_lines.append("TOP 3 CLUSTERS:")
    for cn, cv in cl.head(3).items():
        cc = df[df["Cluster"]==cn]["Cost (£)"].sum()
        ta = df[df["Cluster"]==cn]["Aisle"].dropna().value_counts().head(3)
        bridge_lines.append(f"  {cn}: {cv} ({round(int(cv)/total*100,1)}%) — {fmt_cost(cc)} — Aisles: {', '.join([f'{a}({n})' for a,n in ta.items()])}")
    if findings:
        bridge_lines.append("")
        bridge_lines.append("KEY FINDINGS:")
        for i, f in enumerate(findings, 1):
            bridge_lines.append(f"  {i}. {f}")

    bridge_text = "\n".join(bridge_lines)
    st.text_area("📋 Copy this:", value=bridge_text, height=400, key=f"{kp}bridge_area")

    # ═══════════════════════════════════════════════════════════════════════
    # SUGGESTED ACTIONS (not a guide!)
    # ═══════════════════════════════════════════════════════════════════════
    if findings:
        st.markdown("---")
        st.markdown("#### 💡 Suggested next steps (use your own judgement)")
        st.caption("These are suggestions only — not instructions. You know your station best.")
        for i, f in enumerate(findings, 1):
            st.markdown(f"**{i}.** {f}")


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
        t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📊 Summary","📍 Locations","💡 Shifts","💰 Cost","🔬 Analysis & Bridge","📅 Day","💾 Export"])
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
        t1,t2,t3,t4,t5 = st.tabs(["📊 Summary","📍 Locations","💰 Cost","🔬 Analysis & Bridge","💾 Export"])
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
