import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DRM2 Lost Heatmap", page_icon="📦", layout="wide")
st.title("📦 DRM2 Lost Parcel Heatmap")
st.markdown("---")

STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SHIFT_ORDER = ["NS", "AM", "PM", "OTR"]
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen", "OTR": "firebrick"}
SHIFT_DEFINITIONS = {"NS": "00:00 – 09:59 (Night Sort — stow)", "AM": "10:00 – 13:59 (Pick, stage, dispatch)", "PM": "14:00 – 23:59 (Dispatch, RELO)", "OTR": "On The Road (DSP responsibility)"}
SHIFT_HOUR_MAP = {0:"NS",1:"NS",2:"NS",3:"NS",4:"NS",5:"NS",6:"NS",7:"NS",8:"NS",9:"NS",10:"AM",11:"AM",12:"AM",13:"AM",14:"PM",15:"PM",16:"PM",17:"PM",18:"PM",19:"PM",20:"PM",21:"PM",22:"PM",23:"PM"}
SUB_BUCKET_SHIFT_MAP = {"Lost At Station - Inducted Not Stowed":"NS","Lost At Station - Stowed Not Picked Up":"AM","Lost At Station - Debrief Receive(RTS)":"PM","Lost On Road - Attempted":"OTR","Lost On Road - Damage":"OTR","Lost On Road - No Further Status":"OTR"}
SENSITIVE_COLS = ["Last Scan By","Driver Id","Holder Name","City","Postal","Province","Ordering Order ID","Order Amount","Receivable Amount","Payment Method","District","Scheduled Delivery End Time"]
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
    for col in ["Package Length","Package Width","Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm","").str.replace("cm","")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    dims = ["Package Length","Package Width","Package Height"]
    df["Longest Side"] = df[dims].max(axis=1) if all(c in df.columns for c in dims) else float("nan")
    df["Size Category"] = df["Longest Side"].apply(get_size)
    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], dayfirst=True, errors="coerce")
    if "Dispatch Time" in df.columns:
        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], dayfirst=True, errors="coerce")
    return df

def merge_data(pm_df, scc_df):
    """Merge PM (master) + SCC (detail). Day of Week = PM event_datetime (date marked as lost)."""
    scc_clean = clean_scc(scc_df.copy())
    pm_keep = ["tracking_id","bucket","sub_bucket","previous_event_datetime","previous_reason","previous_reason_3","event_datetime"]
    pm_cols = pm_df[[c for c in pm_keep if c in pm_df.columns]].copy()
    pm_cols = pm_cols.rename(columns={"tracking_id": "Tracking ID"})
    pm_cols["Prev Event DT"] = pd.to_datetime(pm_cols.get("previous_event_datetime", pd.Series(dtype="object")), format="%d/%m/%Y %H:%M", errors="coerce")
    if "event_datetime" in pm_cols.columns:
        pm_cols["Marked Lost DT"] = pd.to_datetime(pm_cols["event_datetime"], dayfirst=True, errors="coerce")
    merged = pm_cols.merge(scc_clean, on="Tracking ID", how="left")
    merged["Sub Bucket"] = merged["sub_bucket"]
    merged["Bucket"] = merged.get("bucket")
    if "Marked Lost DT" in merged.columns:
        merged["Day of Week"] = merged["Marked Lost DT"].dt.day_name()
    elif "Dispatch Time" in merged.columns:
        merged["Day of Week"] = merged["Dispatch Time"].dt.day_name()
    else:
        merged["Day of Week"] = None
    if "previous_reason" in merged.columns:
        merged["Loss Reason"] = merged["previous_reason"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else:
        merged["Loss Reason"] = "Unknown"
    if "previous_reason_3" in merged.columns:
        merged["UTR Reason"] = merged["previous_reason_3"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else:
        merged["UTR Reason"] = "Unknown"
    merged["Shift"] = merged.apply(assign_shift, axis=1)
    for col in ["Cluster","Aisle","Sort Zone","DSP Name","Size Category"]:
        if col not in merged.columns: merged[col] = None
    return merged

def get_date_range(df):
    for col in ["Marked Lost DT","Dispatch Time","Last Updated Time"]:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                s, e = valid.min().strftime("%d %b %Y"), valid.max().strftime("%d %b %Y")
                return s if s == e else f"{s} – {e}"
    return ""

def safe_top(series):
    c = series.dropna().value_counts()
    return c.index[0] if len(c) > 0 else "N/A"

def trunc(labels, max_len=LABEL_MAX):
    return [str(l)[:max_len]+"..." if len(str(l))>max_len else str(l) for l in labels]

def get_detail_cols(df, extra=None):
    base = ["Tracking ID","Cluster","Aisle","Sort Zone","DSP Name","Size Category","Shift","Sub Bucket"]
    if extra: base = extra + [c for c in base if c not in extra]
    return [c for c in base if c in df.columns]

def verify_totals(df, total, label=""):
    if len(df) != total: st.error(f"⚠️ MISMATCH {label}: Expected {total}, got {len(df)}."); return False
    return True

def make_table(series, c1, c2):
    t = series.reset_index(); t.columns = [c1, c2]; t.index = range(1, len(t)+1); return t

def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    h = max(2, len(data)*0.3)
    fig, ax = plt.subplots(figsize=(figsize_width, h))
    labs = trunc(data.index, max_label)
    ax.barh(labs, data.values, color=color); ax.invert_yaxis()
    for i, v in enumerate(data.values): ax.text(v+0.2, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Lost Parcels",fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); return fig

def make_bar_vert(data, xl, yl, title, color="steelblue", figsize=CHART):
    fig, ax = plt.subplots(figsize=figsize)
    labs = trunc(data.index, LABEL_MAX); ax.bar(labs, data.values, color=color)
    for i, v in enumerate(data.values): ax.text(i, v+0.2, str(int(v)), ha="center", fontsize=7)
    ax.set_xlabel(xl,fontsize=8); ax.set_ylabel(yl,fontsize=8); ax.set_title(title,fontsize=9)
    ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); return fig

def make_bar_shift(data, title):
    data = data.reindex(SHIFT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER], color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, str(int(b.get_height())), ha="center", fontsize=7)
    ax.set_xlabel("Shift",fontsize=8); ax.set_ylabel("Lost",fontsize=8); ax.set_title(title,fontsize=9)
    ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); return fig

def make_pie_otr_utr(df, total, title, figsize=(4, 3)):
    otr_n = len(df[df["Sub Bucket"].str.contains("Lost On Road", na=False)])
    utr_n = len(df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"])
    other_n = total - otr_n - utr_n
    labels, sizes, colors, explode = [], [], [], []
    if otr_n > 0: labels.append(f"OTR ({otr_n})"); sizes.append(otr_n); colors.append("firebrick"); explode.append(0.05)
    if utr_n > 0: labels.append(f"UTR ({utr_n})"); sizes.append(utr_n); colors.append("darkorange"); explode.append(0.05)
    if other_n > 0: labels.append(f"Other ({other_n})"); sizes.append(other_n); colors.append("steelblue"); explode.append(0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(sizes, labels=labels, colors=colors, explode=explode, autopct="%1.1f%%", startangle=90, textprops={"fontsize":7})
    ax.set_title(title, fontsize=8); plt.tight_layout(); return fig

def render_locations_tab(df, total, dr, key_prefix=""):
    st.info("💡 **OTR** = DSP breakdown + loss reasons. **UTR** = loss reasons. **All** = cluster/aisle.")
    verify_totals(df, total, "Locations")
    loc_filter = st.radio("Show:", ["All Parcels","OTR Only (Lost On Road)","UTR Reprocess Only"], horizontal=True, key=f"{key_prefix}lf")
    if loc_filter == "OTR Only (Lost On Road)":
        view_df = df[df["Sub Bucket"].str.contains("Lost On Road", na=False)].copy()
        st.write(f"**{len(view_df)} OTR parcels**")
        if len(view_df) == 0: st.warning("No OTR parcels."); return
        with st.expander("🚚 OTR by DSP"):
            dsp_data = view_df["DSP Name"].dropna().value_counts()
            if len(dsp_data) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{key_prefix}otr_d_v")
                if vm == "Chart": st.pyplot(make_bar_horiz(dsp_data, f"OTR by DSP ({dr})", color="firebrick", max_label=DSP_MAX))
                else: st.dataframe(make_table(dsp_data,"DSP","Lost"), use_container_width=True)
        with st.expander("🚚 DSP → Reason Drill-Down"):
            dsps = sorted(view_df["DSP Name"].dropna().unique())
            if dsps:
                sel_dsp = st.selectbox("Select DSP:", dsps, key=f"{key_prefix}otr_dsp_sel")
                dsp_df = view_df[view_df["DSP Name"] == sel_dsp]
                st.write(f"**{len(dsp_df)} parcels** by {sel_dsp}")
                dsp_reasons = dsp_df["Loss Reason"].dropna().value_counts()
                if len(dsp_reasons) > 0: st.dataframe(make_table(dsp_reasons,"Reason Given","Count"), use_container_width=True)
                show_cols = [c for c in ["Tracking ID","Sub Bucket","Loss Reason","Cluster","Aisle"] if c in dsp_df.columns]
                out = dsp_df[show_cols].reset_index(drop=True); out.index = range(1, len(out)+1)
                st.dataframe(out, use_container_width=True)
        with st.expander("❓ All OTR Reasons"):
            reason_data = view_df["Loss Reason"].dropna().value_counts()
            if len(reason_data) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{key_prefix}otr_r_v")
                if vm == "Chart": st.pyplot(make_bar_horiz(reason_data, f"OTR Reasons ({dr})", color="crimson"))
                else: st.dataframe(make_table(reason_data,"Reason","Count"), use_container_width=True)
    elif loc_filter == "UTR Reprocess Only":
        view_df = df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"].copy()
        st.write(f"**{len(view_df)} UTR parcels**")
        if len(view_df) == 0: st.warning("No UTR parcels."); return
        with st.expander("❓ UTR Loss Reasons"):
            reason_data = view_df["UTR Reason"].dropna().value_counts()
            if len(reason_data) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{key_prefix}utr_r_v")
                if vm == "Chart": st.pyplot(make_bar_horiz(reason_data, f"UTR Reasons ({dr})", color="darkorange"))
                else: st.dataframe(make_table(reason_data,"Reason","Count"), use_container_width=True)
            else: st.info("No specific reasons recorded.")
        with st.expander("📍 UTR by Location"):
            cl_data = view_df["Cluster"].dropna().value_counts()
            if len(cl_data) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{key_prefix}utr_l_v")
                if vm == "Chart": st.pyplot(make_bar_horiz(cl_data, f"UTR by Cluster ({dr})", color="darkorange"))
                else: st.dataframe(make_table(cl_data,"Cluster","Count"), use_container_width=True)
        with st.expander("📦 All UTR parcels"):
            show_cols = [c for c in ["Tracking ID","Cluster","Aisle","DSP Name","UTR Reason","Shift"] if c in view_df.columns]
            out = view_df[show_cols].sort_values("Cluster").reset_index(drop=True); out.index = range(1, len(out)+1)
            st.dataframe(out, use_container_width=True)
    else:
        view_df = df.copy()
        st.write(f"**{len(view_df)} parcels (all)**")
        with st.expander("🏆 Top 10 Worst Locations"):
            rank_by = st.selectbox("Rank by:", ["Cluster","Aisle","Sort Zone"], key=f"{key_prefix}rb")
            rank_data = view_df[rank_by].dropna().value_counts().head(10)
            if len(rank_data) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{key_prefix}loc_v")
                if vm == "Chart": st.pyplot(make_bar_horiz(rank_data, f"Top 10 {rank_by}s ({dr})", color="darkred"))
                else: st.dataframe(make_table(rank_data, rank_by, "Lost"), use_container_width=True)
        with st.expander("🔍 Cluster Drill-Down"):
            clusters = sorted(view_df["Cluster"].dropna().unique())
            if clusters:
                sel = st.selectbox("Cluster:", clusters, key=f"{key_prefix}cl")
                filt = view_df[view_df["Cluster"] == sel]
                st.write(f"**{len(filt)} parcels** in Cluster {sel}")
                drill_data = filt["Aisle"].dropna().value_counts()
                if len(drill_data) > 0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{key_prefix}cl_v")
                    if vm == "Chart": st.pyplot(make_bar_horiz(drill_data, f"Cluster {sel} — Aisles", color="steelblue"))
                    else: st.dataframe(make_table(drill_data,"Aisle","Lost"), use_container_width=True)
                show_cols = get_detail_cols(filt, extra=["Tracking ID","Aisle","Sort Zone"])
                detail = filt[show_cols].sort_values("DSP Name").reset_index(drop=True)
                detail.index = range(1, len(detail)+1); st.dataframe(detail, use_container_width=True)

def render_opportunities_tab(df, total, dr, key_prefix=""):
    st.info("💡 Each parcel assigned to the responsible shift via sub-bucket + last scan time.")
    with st.expander("📖 How shift actionable opportunities are assigned"):
        st.markdown("""
| Sub Bucket | Shift | Reasoning |
|---|---|---|
| Inducted Not Stowed | **NS** | Inducted during Night Sort but never stowed |
| Stowed Not Picked Up | **AM** | Stowed but never picked for dispatch |
| Debrief Receive(RTS) | **PM** | Driver returned at debrief |
| Lost On Road - * | **OTR** | Lost while with DSP driver |
| PNOV / UTR / Other | **Time-based** | Last scan hour: 0–9→NS, 10–13→AM, 14–23→PM |
""")
    cols = st.columns(4)
    for i, (s, d) in enumerate(SHIFT_DEFINITIONS.items()): cols[i].markdown(f"**{s}:** {d}")
    st.markdown("---")
    shift_counts = df[df["Shift"] != "Unknown"]["Shift"].value_counts()
    unk_count = len(df[df["Shift"] == "Unknown"])
    with st.expander("🏆 Shift Responsibility Leaderboard"):
        rows = []
        for s in SHIFT_ORDER:
            n = int(shift_counts.get(s, 0)); pct = round(n/total*100, 1) if total > 0 else 0
            rows.append({"Shift": s, "Lost": n, "% Total": f"{pct}%", "Window": SHIFT_DEFINITIONS[s]})
        rows.sort(key=lambda r: r["Lost"], reverse=True)
        lb = pd.DataFrame(rows); lb.index = range(1, len(lb)+1)
        st.dataframe(lb, use_container_width=True, height=200)
        if len(shift_counts) > 0: st.pyplot(make_bar_shift(shift_counts, f"Lost by Shift ({dr})"))
    with st.expander("🔍 Shift Drill-Down"):
        st.caption("Select a shift to see sub-bucket breakdown and parcels.")
        shift_options = [f"{s} — {int(shift_counts.get(s,0))} parcels" for s in SHIFT_ORDER]
        if unk_count > 0: shift_options.append(f"Unknown — {unk_count} parcels")
        selected = st.selectbox("Shift:", shift_options, key=f"{key_prefix}opp_sel")
        selected_shift = selected.split(" — ")[0]
        s_df = df[df["Shift"] == selected_shift]; count = len(s_df)
        if count > 0:
            st.markdown(f"**{selected_shift}** — **{count} parcels** ({round(count/total*100,1)}%)")
            sb_counts = s_df["Sub Bucket"].value_counts()
            sb_tbl = sb_counts.reset_index(); sb_tbl.columns = ["Sub Bucket","Count"]
            sb_tbl["% of Shift"] = (sb_tbl["Count"]/count*100).round(1).astype(str)+"%"
            sb_tbl["% of Total"] = (sb_tbl["Count"]/total*100).round(1).astype(str)+"%"
            sb_tbl.index = range(1, len(sb_tbl)+1)
            vm = st.radio("Display:", ["Table","Chart"], horizontal=True, key=f"{key_prefix}sd_v")
            if vm == "Chart": st.pyplot(make_bar_horiz(sb_counts, f"{selected_shift} Sub Buckets", color=SHIFT_COLORS.get(selected_shift,"steelblue")))
            else: st.dataframe(sb_tbl, use_container_width=True)
            show_cols = [c for c in ["Tracking ID","Sub Bucket","Cluster","Aisle","DSP Name","Size Category","Loss Reason"] if c in df.columns]
            detail = s_df[show_cols].sort_values("Sub Bucket").reset_index(drop=True)
            detail.index = range(1, len(detail)+1); st.dataframe(detail, use_container_width=True)
        else: st.success(f"✅ No parcels for {selected_shift}.")
    st.markdown("---")
    assigned = len(df[df["Shift"] != "Unknown"])
    st.caption(f"✅ Verification: {assigned} + {unk_count} unknown = {assigned+unk_count} (Total: {total})")

mode = st.radio("Mode:", ["Single Station","Multi-Station Compare"], horizontal=True, key="mode")
with st.expander("📖 How to get your data"):
    st.markdown("**1.** PerfectMile → L&U → Lost → Export CSV\n**2.** SCC → paste TIDs → Select All → Export CSV\n**3.** Upload both. PM = source of truth.\n\n**Day of Week** = date parcel was **marked as lost** (PM `event_datetime`).")

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
        if total == 0: st.warning("No data."); st.stop()
        pm_total = len(pm_df); matched = df["Cluster"].notna().sum()
        st.success(f"✅ **{total} lost parcels** (PM:{pm_total}, SCC:{len(scc_df)}, Matched:{matched})")
        if total-matched > 0: st.info(f"ℹ️ {total-matched} in PM not in SCC.")
        if total != pm_total: st.error(f"🚨 MISMATCH: {total} vs PM {pm_total}")
        dr = get_date_range(df)
        st.subheader(f"Quick Summary ({dr})")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Lost", total); c2.metric("Worst Cluster", safe_top(df["Cluster"]))
        c3.metric("Worst Aisle", safe_top(df["Aisle"])); c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]; c5.metric("Worst Shift", safe_top(sk) if len(sk)>0 else "N/A")
        t1,t2,t3,t4,t5,t6 = st.tabs(["📊 Summary","📍 Lost Locations","💡 Suggested Opportunities","📅 Day of Week","💾 Export","📋 Bridge"])
        with t1:
            verify_totals(df, total, "Summary")
            with st.expander("🥧 OTR & UTR Breakdown"):
                st.pyplot(make_pie_otr_utr(df, total, f"OTR & UTR vs Other ({dr})"))
            with st.expander("📏 Parcel Size Breakdown"):
                sc = df["Size Category"].value_counts()
                if len(sc) > 0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="sz_v")
                    if vm == "Chart": st.pyplot(make_bar_vert(sc,"Size","Lost",f"Parcel Size ({dr})", color=["green","orange","red","darkred","grey"][:len(sc)]))
                    else: st.dataframe(make_table(sc,"Size","Count"), use_container_width=True)
            with st.expander("📍 Cluster Breakdown"):
                cc = df["Cluster"].dropna().value_counts()
                if len(cc) > 0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="cl_v")
                    if vm == "Chart": st.pyplot(make_bar_horiz(cc, f"By Cluster ({dr})"))
                    else: st.dataframe(make_table(cc,"Cluster","Count"), use_container_width=True)
            with st.expander("🏷️ Lost Sub Bucket Breakdown"):
                sb = df["Sub Bucket"].value_counts()
                if len(sb) > 0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="sb_v")
                    if vm == "Chart": st.pyplot(make_bar_horiz(sb, f"Sub Bucket ({dr})", color="teal"))
                    else: st.dataframe(make_table(sb,"Sub Bucket","Count"), use_container_width=True)
        with t2: render_locations_tab(df, total, dr, key_prefix="s_")
        with t3: render_opportunities_tab(df, total, dr, key_prefix="s_")
        with t4:
            verify_totals(df, total, "Day")
            st.caption("📌 Day = date parcel was **marked as lost** (PM `event_datetime`).")
            if "Day of Week" in df.columns:
                day_data = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                with st.expander("📅 Day of Week"):
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="day_v")
                    if vm == "Chart":
                        fig, ax = plt.subplots(figsize=CHART)
                        ax.plot(day_data.index, day_data.values, marker="o", color="green", linewidth=2, markersize=6)
                        for i,(d,v) in enumerate(day_data.items()):
                            ax.annotate(str(int(v)), xy=(i,v), xytext=(0,8), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
                        ax.set_xlabel("Day",fontsize=8); ax.set_ylabel("Lost",fontsize=8); ax.set_title(f"Lost by Day Marked ({dr})",fontsize=9)
                        ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig)
                    else: st.dataframe(make_table(day_data,"Day","Lost"), use_container_width=True)
                    st.caption(f"ℹ️ {df['Day of Week'].notna().sum()}/{total} parcels have date data.")
            else: st.warning("No date data available.")
        with t5:
            verify_totals(df, total, "Export")
            exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","Marked Lost DT"]
            ec = [c for c in df.columns if c not in exc]
            st.download_button("⬇️ Download CSV", df[ec].to_csv(index=False), "Lost_Merged.csv", "text/csv")
        with t6:
            verify_totals(df, total, "Bridge")
            cl_c = df["Cluster"].dropna().value_counts(); sb_c = df["Sub Bucket"].value_counts()
            sh_c = df[df["Shift"]!="Unknown"]["Shift"].value_counts()
            sl = "\n".join([f"  {s}: {int(sh_c.get(s,0))} ({round(int(sh_c.get(s,0))/total*100,1)}%)" for s in SHIFT_ORDER])
            sb_lines = "\n".join([f"  {sb}: {n} ({round(int(n)/total*100,1)}%)" for sb,n in sb_c.head(6).items()])
            cdet = ""
            for cn,cv in cl_c.head(3).items():
                ta = df[df["Cluster"]==cn]["Aisle"].dropna().value_counts().head(3)
                cdet += f"  {cn}: {cv} ({round(int(cv)/total*100,1)}%) — {', '.join([f'{a}({n})' for a,n in ta.items()])}\n"
            bridge = f"Lost Parcels Bridge — DRM2\n{dr}\nTOTAL: {total}\nSHIFTS:\n{sl}\nSUB BUCKETS:\n{sb_lines}\nLOCATIONS:\n{cdet}"
            st.text_area("✏️ Bridge:", value=bridge, height=300, key="bridge")
    elif pm_file: st.info("👆 Upload SCC.")
    elif scc_file: st.info("👆 Upload PM.")
    else: st.info("👆 Upload both files above.")
else:
    st.subheader("Upload Station Data")
    num = st.slider("Stations:", 2, 5, 2, key="num_st")
    uploaded = {}
    for i in range(num):
        with st.expander(f"Station {i+1}", expanded=(i<2)):
            ca,cb = st.columns(2)
            with ca: pf = st.file_uploader(f"PM ({i+1})", type="csv", key=f"mp_{i}")
            with cb: sf = st.file_uploader(f"SCC ({i+1})", type="csv", key=f"ms_{i}")
            if pf and sf: uploaded[i] = (pf, sf)
    if len(uploaded) >= 2:
        stations, names = {}, []
        for i,(pf,sf) in uploaded.items():
            pt, st2 = pd.read_csv(pf), pd.read_csv(sf); m = merge_data(pt, st2)
            if "Station" in st2.columns and len(st2["Station"].dropna())>0: nm = st2["Station"].dropna().iloc[0]
            elif "location" in pt.columns and len(pt["location"].dropna())>0: nm = pt["location"].dropna().iloc[0]
            else: nm = f"Station {i+1}"
            stations[nm] = m; names.append(nm)
        st.success(f"✅ {', '.join(names)}")
        t1,t2,t3,t4,t5 = st.tabs(["📊 Summary","📍 Locations","💡 Opportunities","📅 Day","💾 Export"])
        with t1:
            with st.expander("📊 Total Lost Comparison"):
                fig, ax = plt.subplots(figsize=CHART)
                bars = ax.bar(names, [len(stations[n]) for n in names], color=STATION_COLORS[:len(names)])
                for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, str(int(b.get_height())), ha="center", fontsize=8)
                ax.set_ylabel("Lost",fontsize=8); ax.set_title("Total by Station",fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
            with st.expander("🥧 OTR & UTR Breakdown"):
                sel_pie = st.selectbox("Station:", names, key="mc_pie")
                st.pyplot(make_pie_otr_utr(stations[sel_pie], len(stations[sel_pie]), f"{sel_pie} — OTR & UTR"))
        with t2:
            sel = st.selectbox("Station:", names, key="mc_loc")
            render_locations_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), key_prefix=f"mc_{sel}_")
        with t3:
            sel = st.selectbox("Station:", names, key="mc_opp")
            render_opportunities_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), key_prefix=f"mc_{sel}_")
        with t4:
            with st.expander("📅 Day of Week Comparison"):
                st.caption("📌 Day = date parcel was marked as lost.")
                fig, ax = plt.subplots(figsize=CHART)
                for i,n in enumerate(names):
                    if "Day of Week" in stations[n].columns:
                        dd = stations[n]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        ax.plot(dd.index, dd.values, marker="o", label=n, color=STATION_COLORS[i], linewidth=2)
                ax.set_ylabel("Lost",fontsize=8); ax.set_title("Lost by Day Marked",fontsize=9)
                ax.tick_params(labelsize=7); ax.legend(fontsize=7); plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig)
        with t5:
            for n in names:
                exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","Marked Lost DT"]
                ec = [c for c in stations[n].columns if c not in exc]
                st.download_button(f"⬇️ {n}", stations[n][ec].to_csv(index=False), f"Lost_{n}.csv", "text/csv", key=f"dl_{n}")
    elif len(uploaded)==1: st.warning("Need ≥2 stations.")
    else: st.info("👆 Upload file pairs.")
