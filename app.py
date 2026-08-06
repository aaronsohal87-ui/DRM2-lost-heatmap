import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
SENSITIVE_COLS = ["Driver Id","Holder Name","City","Postal","Province","Ordering Order ID","Order Amount","Receivable Amount","Payment Method","District","Scheduled Delivery End Time"]
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
def clean_scc(df):
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])
    if "Last Scan By" in df.columns:
        df["Associate"] = df["Last Scan By"].astype(str).str.split("@").str[0].replace("nan", pd.NA)
        df = df.drop(columns=["Last Scan By"])
    for col in ["Package Length","Package Width","Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm","").str.replace("cm","")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    dims = ["Package Length","Package Width","Package Height"]
    df["Longest Side"] = df[dims].max(axis=1) if all(c in df.columns for c in dims) else float("nan")
    df["Size Category"] = df["Longest Side"].apply(get_size)
    if "Last Updated Time" in df.columns: df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], dayfirst=True, errors="coerce")
    if "Dispatch Time" in df.columns: df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], dayfirst=True, errors="coerce")
    return df
def merge_data(pm_df, scc_df):
    scc_clean = clean_scc(scc_df.copy())
    pm_keep = ["tracking_id","bucket","sub_bucket","previous_event_datetime","previous_reason","previous_reason_3","event_datetime"]
    pm_cols = pm_df[[c for c in pm_keep if c in pm_df.columns]].copy()
    pm_cols = pm_cols.rename(columns={"tracking_id":"Tracking ID"})
    pm_cols["Prev Event DT"] = pd.to_datetime(pm_cols.get("previous_event_datetime", pd.Series(dtype="object")), format="%d/%m/%Y %H:%M", errors="coerce")
    if "event_datetime" in pm_cols.columns: pm_cols["Marked Lost DT"] = pd.to_datetime(pm_cols["event_datetime"], dayfirst=True, errors="coerce")
    merged = pm_cols.merge(scc_clean, on="Tracking ID", how="left")
    merged["Sub Bucket"] = merged["sub_bucket"]; merged["Bucket"] = merged.get("bucket")
    if "Marked Lost DT" in merged.columns: merged["Day of Week"] = merged["Marked Lost DT"].dt.day_name()
    elif "Dispatch Time" in merged.columns: merged["Day of Week"] = merged["Dispatch Time"].dt.day_name()
    else: merged["Day of Week"] = None
    if "previous_reason" in merged.columns: merged["Loss Reason"] = merged["previous_reason"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else: merged["Loss Reason"] = "Unknown"
    if "previous_reason_3" in merged.columns: merged["UTR Reason"] = merged["previous_reason_3"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else: merged["UTR Reason"] = "Unknown"
    merged["Shift"] = merged.apply(assign_shift, axis=1)
    for col in ["Cluster","Aisle","Sort Zone","DSP Name","Size Category","Associate"]:
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
def safe_top(s):
    c = s.dropna().value_counts(); return c.index[0] if len(c)>0 else "N/A"
def trunc(labels, mx=LABEL_MAX): return [str(l)[:mx]+"..." if len(str(l))>mx else str(l) for l in labels]
def get_detail_cols(df, extra=None):
    base = ["Tracking ID","Cluster","Aisle","Sort Zone","DSP Name","Size Category","Shift","Sub Bucket"]
    if extra: base = extra + [c for c in base if c not in extra]
    return [c for c in base if c in df.columns]
def verify_totals(df, total, label=""):
    if len(df)!=total: st.error(f"⚠️ MISMATCH {label}: Expected {total}, got {len(df)}."); return False
    return True
def make_table(series, c1, c2):
    t = series.reset_index(); t.columns = [c1, c2]; t.index = range(1, len(t)+1); return t
def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    h = max(2, len(data)*0.3); fig, ax = plt.subplots(figsize=(figsize_width, h))
    labs = trunc(data.index, max_label); ax.barh(labs, data.values, color=color); ax.invert_yaxis()
    for i,v in enumerate(data.values): ax.text(v+0.2, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Lost Parcels",fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); return fig
def make_bar_vert(data, xl, yl, title, color="steelblue", figsize=CHART):
    fig, ax = plt.subplots(figsize=figsize); labs = trunc(data.index, LABEL_MAX); ax.bar(labs, data.values, color=color)
    for i,v in enumerate(data.values): ax.text(i, v+0.2, str(int(v)), ha="center", fontsize=7)
    ax.set_xlabel(xl,fontsize=8); ax.set_ylabel(yl,fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); return fig
def make_bar_shift(data, title):
    data = data.reindex(SHIFT_ORDER, fill_value=0); fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER], color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, str(int(b.get_height())), ha="center", fontsize=7)
    ax.set_xlabel("Shift",fontsize=8); ax.set_ylabel("Lost",fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); return fig
def make_pie_otr_utr(df, total, title):
    otr_n = len(df[df["Sub Bucket"].str.contains("Lost On Road", na=False)]); utr_n = len(df[df["Sub Bucket"]=="Lost At Station - UTR Reprocess"]); other_n = total-otr_n-utr_n
    labels,sizes,colors,explode = [],[],[],[]
    if otr_n>0: labels.append(f"OTR ({otr_n})"); sizes.append(otr_n); colors.append("firebrick"); explode.append(0.05)
    if utr_n>0: labels.append(f"UTR ({utr_n})"); sizes.append(utr_n); colors.append("darkorange"); explode.append(0.05)
    if other_n>0: labels.append(f"Other ({other_n})"); sizes.append(other_n); colors.append("steelblue"); explode.append(0)
    fig, ax = plt.subplots(figsize=(3, 2.2))
    ax.pie(sizes, labels=labels, colors=colors, explode=explode, autopct="%1.1f%%", startangle=90, textprops={"fontsize":6})
    ax.set_title(title, fontsize=7); plt.tight_layout(); return fig

def render_locations_tab(df, total, dr, kp=""):
    st.info("💡 **OTR** = DSP + reasons. **UTR** = loss reasons. **All** = cluster/aisle.")
    verify_totals(df, total, "Locations")
    lf = st.radio("Show:", ["All Parcels","OTR Only (Lost On Road)","UTR Reprocess Only"], horizontal=True, key=f"{kp}lf")
    if lf == "OTR Only (Lost On Road)":
        vdf = df[df["Sub Bucket"].str.contains("Lost On Road", na=False)].copy(); st.write(f"**{len(vdf)} OTR parcels**")
        if len(vdf)==0: st.warning("No OTR parcels."); return
        with st.expander("🚚 OTR by DSP"):
            d = vdf["DSP Name"].dropna().value_counts()
            if len(d)>0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}od"); 
                if vm=="Chart": st.pyplot(make_bar_horiz(d, f"OTR by DSP ({dr})", color="firebrick", max_label=DSP_MAX))
                else: st.dataframe(make_table(d,"DSP","Lost"), use_container_width=True)
        with st.expander("🚚 DSP → Reason"):
            dsps = sorted(vdf["DSP Name"].dropna().unique())
            if dsps:
                sd = st.selectbox("DSP:", dsps, key=f"{kp}ods"); ddf = vdf[vdf["DSP Name"]==sd]; st.write(f"**{len(ddf)}** by {sd}")
                r = ddf["Loss Reason"].dropna().value_counts()
                if len(r)>0: st.dataframe(make_table(r,"Reason","Count"), use_container_width=True)
        with st.expander("❓ All OTR Reasons"):
            r = vdf["Loss Reason"].dropna().value_counts()
            if len(r)>0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}or")
                if vm=="Chart": st.pyplot(make_bar_horiz(r, f"OTR Reasons ({dr})", color="crimson"))
                else: st.dataframe(make_table(r,"Reason","Count"), use_container_width=True)
    elif lf == "UTR Reprocess Only":
        vdf = df[df["Sub Bucket"]=="Lost At Station - UTR Reprocess"].copy(); st.write(f"**{len(vdf)} UTR parcels**")
        if len(vdf)==0: st.warning("No UTR parcels."); return
        with st.expander("❓ UTR Reasons"):
            r = vdf["UTR Reason"].dropna().value_counts()
            if len(r)>0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}ur")
                if vm=="Chart": st.pyplot(make_bar_horiz(r, f"UTR Reasons ({dr})", color="darkorange"))
                else: st.dataframe(make_table(r,"Reason","Count"), use_container_width=True)
            else: st.info("No reasons recorded.")
        with st.expander("📍 UTR by Location"):
            cl = vdf["Cluster"].dropna().value_counts()
            if len(cl)>0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}ul")
                if vm=="Chart": st.pyplot(make_bar_horiz(cl, f"UTR Clusters ({dr})", color="darkorange"))
                else: st.dataframe(make_table(cl,"Cluster","Count"), use_container_width=True)
    else:
        vdf = df.copy(); st.write(f"**{len(vdf)} parcels (all)**")
        with st.expander("🏆 Top 10 Locations"):
            rb = st.selectbox("Rank by:", ["Cluster","Aisle","Sort Zone"], key=f"{kp}rb")
            rd = vdf[rb].dropna().value_counts().head(10)
            if len(rd)>0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}lv")
                if vm=="Chart": st.pyplot(make_bar_horiz(rd, f"Top 10 {rb}s ({dr})", color="darkred"))
                else: st.dataframe(make_table(rd, rb, "Lost"), use_container_width=True)
        with st.expander("🔍 Cluster Drill-Down"):
            clusters = sorted(vdf["Cluster"].dropna().unique())
            if clusters:
                sel = st.selectbox("Cluster:", clusters, key=f"{kp}cl"); filt = vdf[vdf["Cluster"]==sel]
                st.write(f"**{len(filt)}** in Cluster {sel}")
                ad = filt["Aisle"].dropna().value_counts()
                if len(ad)>0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}cv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(ad, f"Cluster {sel} Aisles", color="steelblue"))
                    else: st.dataframe(make_table(ad,"Aisle","Lost"), use_container_width=True)
                sc = get_detail_cols(filt, extra=["Tracking ID","Aisle","Sort Zone"])
                det = filt[sc].sort_values("DSP Name").reset_index(drop=True); det.index = range(1,len(det)+1)
                st.dataframe(det, use_container_width=True)

def render_opportunities_tab(df, total, dr, kp=""):
    st.info("💡 Each parcel assigned to the responsible shift via sub-bucket + last scan time.")
    with st.expander("📖 How shift actionable opportunities are assigned"):
        st.markdown("""| Sub Bucket | Shift | Reasoning |
|---|---|---|
| Inducted Not Stowed | **NS** | Inducted during Night Sort but never stowed |
| Stowed Not Picked Up | **AM** | Stowed but never picked for dispatch |
| Debrief Receive(RTS) | **PM** | Driver returned at debrief |
| Lost On Road - * | **OTR** | Lost while with DSP driver |
| PNOV / UTR / Other | **Time-based** | Last scan hour: 0–9→NS, 10–13→AM, 14–23→PM |""")
    cols = st.columns(4)
    for i,(s,d) in enumerate(SHIFT_DEFINITIONS.items()): cols[i].markdown(f"**{s}:** {d}")
    st.markdown("---")
    sc = df[df["Shift"]!="Unknown"]["Shift"].value_counts(); unk = len(df[df["Shift"]=="Unknown"])
    with st.expander("🏆 Shift Opportunity Leaderboard"):
        rows = []
        for s in SHIFT_ORDER:
            n = int(sc.get(s,0)); rows.append({"Shift":s,"Lost":n,"%":f"{round(n/total*100,1)}%","Window":SHIFT_DEFINITIONS[s]})
        rows.sort(key=lambda r: r["Lost"], reverse=True)
        lb = pd.DataFrame(rows); lb.index = range(1,len(lb)+1)
        st.dataframe(lb, use_container_width=True, height=200)
        if len(sc)>0: st.pyplot(make_bar_shift(sc, f"Lost by Shift ({dr})"))
    with st.expander("🔍 Shift Drill-Down"):
        opts = [f"{s} — {int(sc.get(s,0))} parcels" for s in SHIFT_ORDER]
        if unk>0: opts.append(f"Unknown — {unk} parcels")
        sel = st.selectbox("Shift:", opts, key=f"{kp}os"); ss = sel.split(" — ")[0]
        sdf = df[df["Shift"]==ss]; cnt = len(sdf)
        if cnt>0:
            st.markdown(f"**{ss}** — **{cnt} parcels** ({round(cnt/total*100,1)}%)")
            sbc = sdf["Sub Bucket"].value_counts(); sbt = sbc.reset_index(); sbt.columns = ["Sub Bucket","Count"]
            sbt["%"] = (sbt["Count"]/cnt*100).round(1).astype(str)+"%"; sbt.index = range(1,len(sbt)+1)
            vm = st.radio("Display:", ["Table","Chart"], horizontal=True, key=f"{kp}sd")
            if vm=="Chart": st.pyplot(make_bar_horiz(sbc, f"{ss} Sub Buckets", color=SHIFT_COLORS.get(ss,"steelblue")))
            else: st.dataframe(sbt, use_container_width=True)
            sc2 = [c for c in ["Tracking ID","Sub Bucket","Cluster","Aisle","DSP Name","Loss Reason"] if c in df.columns]
            det = sdf[sc2].sort_values("Sub Bucket").reset_index(drop=True); det.index = range(1,len(det)+1)
            st.dataframe(det, use_container_width=True)
    st.markdown("---"); st.caption(f"✅ {len(df[df['Shift']!='Unknown'])} assigned + {unk} unknown = {total}")

def render_associates_tab(df, total, dr, kp=""):
    st.info("💡 Shows the associate (login) who **last scanned** each lost parcel. "
            "This doesn't mean they caused the loss — they were the last to handle it.")
    if "Associate" not in df.columns or df["Associate"].dropna().nunique()==0:
        st.warning("No associate data (SCC 'Last Scan By' not found)."); return
    with st.expander("👤 Associates with Most Lost Parcels"):
        ad = df["Associate"].dropna().value_counts()
        if len(ad)>0:
            vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}av")
            if vm=="Chart": st.pyplot(make_bar_horiz(ad.head(15), f"Last Scanned By ({dr})", color="purple"))
            else: st.dataframe(make_table(ad,"Associate","Parcels"), use_container_width=True)
    with st.expander("👤 Associate Drill-Down"):
        assocs = sorted(df["Associate"].dropna().unique())
        if assocs:
            sel = st.selectbox("Associate:", assocs, key=f"{kp}as"); adf = df[df["Associate"]==sel]
            st.write(f"**{len(adf)} parcels** last scanned by {sel}")
            sb = adf["Sub Bucket"].value_counts()
            if len(sb)>0: st.dataframe(make_table(sb,"Sub Bucket","Count"), use_container_width=True)
            sc2 = [c for c in ["Tracking ID","Sub Bucket","Cluster","Aisle","DSP Name","Shift"] if c in adf.columns]
            out = adf[sc2].sort_values("Sub Bucket").reset_index(drop=True); out.index = range(1,len(out)+1)
            st.dataframe(out, use_container_width=True)

mode = st.radio("Mode:", ["Single Station","Multi-Station Compare"], horizontal=True, key="mode")
with st.expander("📖 How to get your data"):
    st.markdown("**1.** PerfectMile → L&U → Lost → Export CSV\n**2.** SCC → paste TIDs → Select All → Export CSV\n**3.** Upload both. PM = source of truth.")
if mode == "Single Station":
    st.subheader("Upload Data")
    col_pm, col_scc = st.columns(2)
    with col_pm: pm_file = st.file_uploader("📊 Perfect Mile (.csv)", type="csv", key="pm_up")
    with col_scc: scc_file = st.file_uploader("📋 SCC (.csv)", type="csv", key="scc_up")
    if pm_file and scc_file:
        pm_df, scc_df = pd.read_csv(pm_file), pd.read_csv(scc_file)
        pm_miss = [c for c in REQUIRED_PM_COLS if c not in pm_df.columns]
        if pm_miss: st.error(f"❌ PM missing: {pm_miss}"); st.stop()
        scc_miss = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
        if scc_miss: st.error(f"❌ SCC missing: {scc_miss}"); st.stop()
        found = [c for c in SENSITIVE_COLS if c in scc_df.columns]
        if found: st.warning(f"🔒 PII removed: {', '.join(found)}")
        df = merge_data(pm_df, scc_df); total = len(df)
        if total==0: st.warning("No data."); st.stop()
        pm_total = len(pm_df); matched = df["Cluster"].notna().sum()
        st.success(f"✅ **{total} lost parcels** (PM:{pm_total}, SCC:{len(scc_df)}, Matched:{matched})")
        if total-matched > 0: st.info(f"ℹ️ {total-matched} parcel(s) in Perfect Mile had no matching row in SCC — included but with limited detail (no cluster/aisle/DSP info).")
        if total != pm_total: st.error(f"🚨 MISMATCH: {total} vs PM {pm_total}")
        dr = get_date_range(df)
        st.subheader(f"Quick Summary ({dr})")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Lost", total); c2.metric("Worst Cluster", safe_top(df["Cluster"]))
        c3.metric("Worst Aisle", safe_top(df["Aisle"])); c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]; c5.metric("Worst Shift", safe_top(sk) if len(sk)>0 else "N/A")
        t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📊 Summary","📍 Lost Locations","💡 Opportunities","👤 Associates","📅 Day of Week","💾 Export","📋 Bridge"])
        with t1:
            verify_totals(df, total, "Summary")
            with st.expander("🥧 OTR & UTR"): st.pyplot(make_pie_otr_utr(df, total, f"OTR & UTR vs Other ({dr})"))
            with st.expander("📏 Parcel Size Breakdown"):
                sc = df["Size Category"].value_counts()
                if len(sc)>0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="sz")
                    if vm=="Chart": st.pyplot(make_bar_vert(sc,"Size","Lost",f"Parcel Size ({dr})", color=["green","orange","red","darkred","grey"][:len(sc)]))
                    else: st.dataframe(make_table(sc,"Size","Count"), use_container_width=True)
            with st.expander("📍 Cluster Breakdown"):
                cc = df["Cluster"].dropna().value_counts()
                if len(cc)>0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="cv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(cc, f"By Cluster ({dr})"))
                    else: st.dataframe(make_table(cc,"Cluster","Count"), use_container_width=True)
            with st.expander("🏷️ Lost Sub Bucket Breakdown"):
                sb = df["Sub Bucket"].value_counts()
                if len(sb)>0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="sv")
                    if vm=="Chart": st.pyplot(make_bar_horiz(sb, f"Sub Bucket ({dr})", color="teal"))
                    else: st.dataframe(make_table(sb,"Sub Bucket","Count"), use_container_width=True)
        with t2: render_locations_tab(df, total, dr, kp="s_")
        with t3: render_opportunities_tab(df, total, dr, kp="s_")
        with t4: render_associates_tab(df, total, dr, kp="s_")
        with t5:
            verify_totals(df, total, "Day")
            st.markdown("""**📌 How Day of Week is calculated:**
Each parcel has an `event_datetime` in Perfect Mile — this is the exact date and time the system **marked that parcel as lost**.
We take the day of the week from that date. For example, if a parcel was marked lost on `28/07/2026` (a Monday), it counts as Monday.
This gives a natural spread across the week because parcels are flagged on different days as they're identified as missing.""")
            if "Day of Week" in df.columns:
                dd = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                with st.expander("📅 Day of Week"):
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="dv")
                    if vm=="Chart":
                        fig, ax = plt.subplots(figsize=CHART)
                        ax.plot(dd.index, dd.values, marker="o", color="green", linewidth=2, markersize=6)
                        for i,(d,v) in enumerate(dd.items()): ax.annotate(str(int(v)), xy=(i,v), xytext=(0,8), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
                        ax.set_xlabel("Day",fontsize=8); ax.set_ylabel("Lost",fontsize=8); ax.set_title(f"Lost by Day Marked ({dr})",fontsize=9)
                        ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig)
                    else: st.dataframe(make_table(dd,"Day","Lost"), use_container_width=True)
                    st.caption(f"ℹ️ {df['Day of Week'].notna().sum()}/{total} parcels have date data.")
        with t6:
            exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","Marked Lost DT"]
            ec = [c for c in df.columns if c not in exc]; st.download_button("⬇️ Download CSV", df[ec].to_csv(index=False), "Lost_Merged.csv", "text/csv")
        with t7:
            cl_c = df["Cluster"].dropna().value_counts(); sb_c = df["Sub Bucket"].value_counts()
            sh_c = df[df["Shift"]!="Unknown"]["Shift"].value_counts()
            sl = "\n".join([f"  {s}: {int(sh_c.get(s,0))} ({round(int(sh_c.get(s,0))/total*100,1)}%)" for s in SHIFT_ORDER])
            sbl = "\n".join([f"  {sb}: {n} ({round(int(n)/total*100,1)}%)" for sb,n in sb_c.head(6).items()])
            cdet = ""
            for cn,cv in cl_c.head(3).items():
                ta = df[df["Cluster"]==cn]["Aisle"].dropna().value_counts().head(3)
                cdet += f"  {cn}: {cv} ({round(int(cv)/total*100,1)}%) — {', '.join([f'{a}({n})' for a,n in ta.items()])}\n"
            st.text_area("✏️ Bridge:", value=f"Lost Parcels Bridge — DRM2\n{dr}\nTOTAL: {total}\nSHIFTS:\n{sl}\nSUB BUCKETS:\n{sbl}\nLOCATIONS:\n{cdet}", height=300, key="bridge")
    elif pm_file: st.info("👆 Upload SCC.")
    elif scc_file: st.info("👆 Upload PM.")
    else: st.info("👆 Upload both files above.")
else:
    st.subheader("Upload Station Data"); num = st.slider("Stations:", 2, 5, 2, key="ns")
    uploaded = {}
    for i in range(num):
        with st.expander(f"Station {i+1}", expanded=(i<2)):
            ca,cb = st.columns(2)
            with ca: pf = st.file_uploader(f"PM ({i+1})", type="csv", key=f"mp{i}")
            with cb: sf = st.file_uploader(f"SCC ({i+1})", type="csv", key=f"ms{i}")
            if pf and sf: uploaded[i] = (pf, sf)
    if len(uploaded)>=2:
        stations, names = {}, []
        for i,(pf,sf) in uploaded.items():
            pt,s2 = pd.read_csv(pf), pd.read_csv(sf); m = merge_data(pt, s2)
            if "Station" in s2.columns and len(s2["Station"].dropna())>0: nm = s2["Station"].dropna().iloc[0]
            elif "location" in pt.columns and len(pt["location"].dropna())>0: nm = pt["location"].dropna().iloc[0]
            else: nm = f"Station {i+1}"
            stations[nm] = m; names.append(nm)
        st.success(f"✅ {', '.join(names)}")
        t1,t2,t3,t4,t5,t6 = st.tabs(["📊 Summary","📍 Locations","💡 Opportunities","👤 Associates","📅 Day","💾 Export"])
        with t1:
            with st.expander("📊 Total Lost"):
                fig, ax = plt.subplots(figsize=CHART)
                bars = ax.bar(names, [len(stations[n]) for n in names], color=STATION_COLORS[:len(names)])
                for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, str(int(b.get_height())), ha="center", fontsize=8)
                ax.set_ylabel("Lost",fontsize=8); ax.set_title("Total by Station",fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
            with st.expander("🥧 OTR & UTR"):
                sp = st.selectbox("Station:", names, key="mcp"); st.pyplot(make_pie_otr_utr(stations[sp], len(stations[sp]), f"{sp} OTR & UTR"))
        with t2:
            sel = st.selectbox("Station:", names, key="mcl"); render_locations_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc_{sel}_")
        with t3:
            sel = st.selectbox("Station:", names, key="mco"); render_opportunities_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc_{sel}_")
        with t4:
            sel = st.selectbox("Station:", names, key="mca"); render_associates_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc_{sel}_")
        with t5:
            with st.expander("📅 Day of Week"):
                st.caption("Day = date parcel was marked as lost in Perfect Mile.")
                fig, ax = plt.subplots(figsize=CHART)
                for i,n in enumerate(names):
                    if "Day of Week" in stations[n].columns:
                        dd = stations[n]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        ax.plot(dd.index, dd.values, marker="o", label=n, color=STATION_COLORS[i], linewidth=2)
                ax.set_ylabel("Lost",fontsize=8); ax.set_title("By Day Marked",fontsize=9); ax.tick_params(labelsize=7); ax.legend(fontsize=7); plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig)
        with t6:
            for n in names:
                exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","Marked Lost DT"]
                ec = [c for c in stations[n].columns if c not in exc]; st.download_button(f"⬇️ {n}", stations[n][ec].to_csv(index=False), f"Lost_{n}.csv", "text/csv", key=f"dl{n}")
    elif len(uploaded)==1: st.warning("Need ≥2 stations.")
    else: st.info("👆 Upload file pairs.")
