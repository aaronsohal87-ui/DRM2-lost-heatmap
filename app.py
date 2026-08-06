import streamlit as st  # Streamlit UI framework
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Charting

# ─────────────────────────────────────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DRM2 Lost Heatmap", page_icon="📦", layout="wide")
st.title("📦 DRM2 Lost Parcel Heatmap")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SHIFT_ORDER = ["NS", "AM", "PM", "OTR"]
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen", "OTR": "firebrick"}
SHIFT_DEFINITIONS = {
    "NS": "00:00 – 09:59 (Night Sort — stow)",
    "AM": "10:00 – 13:59 (Pick, stage, dispatch)",
    "PM": "14:00 – 23:59 (Dispatch, RELO)",
    "OTR": "On The Road (DSP responsibility)"
}
SHIFT_HOUR_MAP = {
    0: "NS", 1: "NS", 2: "NS", 3: "NS", 4: "NS",
    5: "NS", 6: "NS", 7: "NS", 8: "NS", 9: "NS",
    10: "AM", 11: "AM", 12: "AM", 13: "AM",
    14: "PM", 15: "PM", 16: "PM", 17: "PM", 18: "PM",
    19: "PM", 20: "PM", 21: "PM", 22: "PM", 23: "PM"
}
SUB_BUCKET_SHIFT_MAP = {
    "Lost At Station - Inducted Not Stowed": "NS",
    "Lost At Station - Stowed Not Picked Up": "AM",
    "Lost At Station - Debrief Receive(RTS)": "PM",
    "Lost On Road - Attempted": "OTR",
    "Lost On Road - Damage": "OTR",
    "Lost On Road - No Further Status": "OTR",
}
SENSITIVE_COLS = [
    "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
    "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
    "Payment Method", "District", "Scheduled Delivery End Time"
]
REQUIRED_SCC_COLS = [
    "Tracking ID", "Sort Zone", "Aisle", "Cluster",
    "Package Length", "Package Width", "Package Height",
    "DSP Name", "Assigned Cycle", "Last Updated Time"
]
REQUIRED_PM_COLS = ["tracking_id", "sub_bucket"]
CHART = (7, 2.5)
CHART_SM = (6, 2)
DSP_MAX = 20
LABEL_MAX = 25
DETAIL_COLS = ["Tracking ID", "Cluster", "Aisle", "Sort Zone",
               "DSP Name", "Size Category", "Shift", "Sub Bucket"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


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
    for col in ["Package Length", "Package Width", "Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm", "").str.replace("cm", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    dims = ["Package Length", "Package Width", "Package Height"]
    if all(c in df.columns for c in dims):
        df["Longest Side"] = df[dims].max(axis=1)
    else:
        df["Longest Side"] = float("nan")
    df["Size Category"] = df["Longest Side"].apply(get_size)
    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], dayfirst=True, errors="coerce")
        df["Day of Week"] = df["Last Updated Time"].dt.day_name()
    if "Dispatch Time" in df.columns:
        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], dayfirst=True, errors="coerce")
    return df


def merge_data(pm_df, scc_df):
    scc_clean = clean_scc(scc_df.copy())
    pm_cols = pm_df[["tracking_id", "bucket", "sub_bucket", "previous_event_datetime"]].copy()
    pm_cols = pm_cols.rename(columns={"tracking_id": "Tracking ID"})
    pm_cols["Prev Event DT"] = pd.to_datetime(pm_cols["previous_event_datetime"], format="%d/%m/%Y %H:%M", errors="coerce")
    merged = pm_cols.merge(scc_clean, on="Tracking ID", how="left")
    merged["Sub Bucket"] = merged["sub_bucket"]
    merged["Bucket"] = merged["bucket"]
    merged["Shift"] = merged.apply(assign_shift, axis=1)
    for col in ["Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"]:
        if col not in merged.columns: merged[col] = None
    return merged


def get_date_range(df):
    if "Last Updated Time" not in df.columns: return ""
    valid = df["Last Updated Time"].dropna()
    if len(valid) == 0: return ""
    s = valid.min().strftime("%d %b %Y")
    e = valid.max().strftime("%d %b %Y")
    return s if s == e else f"{s} – {e}"


def safe_top(series):
    c = series.dropna().value_counts()
    return c.index[0] if len(c) > 0 else "N/A"


def trunc(labels, max_len=LABEL_MAX):
    return [str(l)[:max_len] + "..." if len(str(l)) > max_len else str(l) for l in labels]


def get_detail_cols(df, extra=None):
    base = list(DETAIL_COLS)
    if extra: base = extra + [c for c in base if c not in extra]
    return [c for c in base if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    h = max(2, len(data) * 0.3)
    fig, ax = plt.subplots(figsize=(figsize_width, h))
    labs = trunc(data.index, max_label)
    ax.barh(labs, data.values, color=color)
    ax.invert_yaxis()
    for i, v in enumerate(data.values):
        ax.text(v + 0.2, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def make_bar_vert(data, xl, yl, title, color="steelblue", figsize=CHART):
    fig, ax = plt.subplots(figsize=figsize)
    labs = trunc(data.index, LABEL_MAX)
    ax.bar(labs, data.values, color=color)
    for i, v in enumerate(data.values):
        ax.text(i, v + 0.2, str(int(v)), ha="center", fontsize=7)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def make_bar_shift(data, title):
    order = SHIFT_ORDER
    data = data.reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(order, [data[s] for s in order], color=[SHIFT_COLORS[s] for s in order])
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.2, str(int(h)), ha="center", fontsize=7)
    ax.set_xlabel("Shift", fontsize=8)
    ax.set_ylabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def make_pie_otr_utr(df, total, title):
    otr_count = len(df[df["Sub Bucket"].str.contains("Lost On Road", na=False)])
    utr_count = len(df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"])
    other_count = total - otr_count - utr_count
    labels, sizes, colors, explode = [], [], [], []
    if otr_count > 0:
        labels.append(f"OTR ({otr_count})"); sizes.append(otr_count)
        colors.append("firebrick"); explode.append(0.05)
    if utr_count > 0:
        labels.append(f"UTR Reprocess ({utr_count})"); sizes.append(utr_count)
        colors.append("darkorange"); explode.append(0.05)
    if other_count > 0:
        labels.append(f"Other Lost ({other_count})"); sizes.append(other_count)
        colors.append("steelblue"); explode.append(0)
    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                                       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    for at in autotexts: at.set_fontsize(8); at.set_fontweight("bold")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    return fig


def make_table(series, c1, c2):
    t = series.reset_index()
    t.columns = [c1, c2]
    t.index = range(1, len(t) + 1)
    return t


def verify_totals(df, total, label=""):
    if len(df) != total:
        st.error(f"⚠️ COUNT MISMATCH {label}: Expected {total}, got {len(df)}.")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SUGGESTED OPPORTUNITIES TAB
# ─────────────────────────────────────────────────────────────────────────────


def render_opportunities_tab(df, total, dr, key_prefix=""):
    st.info("💡 **What's here:** Each lost parcel is assigned to the shift most likely "
            "responsible based on its **sub-bucket** and **time of last scan**.")
    with st.expander("📖 How shifts are assigned"):
        st.markdown("""
| Sub Bucket | Assigned Shift | Reasoning |
|---|---|---|
| Inducted Not Stowed | **NS** | Parcel inducted during Night Sort but never stowed |
| Stowed Not Picked Up | **AM** | Stowed but never picked for dispatch |
| Debrief Receive(RTS) | **PM** | Driver returned parcel at debrief (PM) |
| Lost On Road - * | **OTR** | Lost while with DSP driver |
| PNOV / UTR / Other | **Time-based** | Uses last scan time before EoD to assign shift |
""")
        st.caption("Time-based: hour 0–9 → NS, 10–13 → AM, 14–23 → PM.")
    cols = st.columns(4)
    for i, (s, d) in enumerate(SHIFT_DEFINITIONS.items()):
        cols[i].markdown(f"**{s}:** {d}")
    st.markdown("---")
    st.subheader("🏆 Shift Responsibility Leaderboard")
    shift_counts = df[df["Shift"] != "Unknown"]["Shift"].value_counts()
    rows = []
    for s in SHIFT_ORDER:
        n = int(shift_counts.get(s, 0))
        pct = round(n / total * 100, 1) if total > 0 else 0
        rows.append({"Shift": s, "Lost Parcels": n, "% of Total": f"{pct}%", "Time Window": SHIFT_DEFINITIONS[s]})
    rows.sort(key=lambda r: r["Lost Parcels"], reverse=True)
    lb = pd.DataFrame(rows)
    lb.index = range(1, len(lb) + 1)
    st.dataframe(lb, use_container_width=True, height=200)
    if len(shift_counts) > 0:
        st.pyplot(make_bar_shift(shift_counts, f"Lost by Responsible Shift ({dr})"))
    st.markdown("---")
    st.subheader("🥧 OTR & UTR Breakdown")
    st.caption("Proportion of parcels lost On The Road (OTR) vs UTR Reprocess vs other.")
    st.pyplot(make_pie_otr_utr(df, total, f"OTR & UTR vs Other ({dr})"))
    st.markdown("---")
    st.subheader("🔍 Shift Drill-Down")
    st.caption("Select a shift to see its sub-bucket breakdown and all parcels.")
    shift_options = []
    for s in SHIFT_ORDER:
        n = int(shift_counts.get(s, 0))
        shift_options.append(f"{s} — {n} parcels")
    unk_count = len(df[df["Shift"] == "Unknown"])
    if unk_count > 0:
        shift_options.append(f"Unknown — {unk_count} parcels")
    selected = st.selectbox("Select Shift:", shift_options, key=f"{key_prefix}opp_shift_sel")
    selected_shift = selected.split(" — ")[0]
    s_df = df[df["Shift"] == selected_shift].copy()
    count = len(s_df)
    if count > 0:
        pct = round(count / total * 100, 1)
        st.markdown(f"**{selected_shift}** — **{count} parcels** ({pct}% of total) — "
                    f"*{SHIFT_DEFINITIONS.get(selected_shift, 'N/A')}*")
        sb_counts = s_df["Sub Bucket"].value_counts()
        sb_tbl = sb_counts.reset_index()
        sb_tbl.columns = ["Sub Bucket", "Count"]
        sb_tbl["% of Shift"] = (sb_tbl["Count"] / count * 100).round(1).astype(str) + "%"
        sb_tbl["% of Total"] = (sb_tbl["Count"] / total * 100).round(1).astype(str) + "%"
        sb_tbl.index = range(1, len(sb_tbl) + 1)
        st.dataframe(sb_tbl, use_container_width=True)
        with st.expander(f"📦 All {count} parcels for {selected_shift} shift"):
            show_cols = [c for c in ["Tracking ID", "Sub Bucket", "Cluster", "Aisle",
                                     "Sort Zone", "DSP Name", "Size Category"] if c in df.columns]
            detail = s_df[show_cols].sort_values("Sub Bucket").reset_index(drop=True)
            detail.index = range(1, len(detail) + 1)
            st.dataframe(detail, use_container_width=True)
    else:
        st.success(f"✅ No parcels assigned to {selected_shift} shift.")
    st.markdown("---")
    assigned = len(df[df["Shift"] != "Unknown"])
    st.caption(f"✅ Verification: {assigned} assigned + {unk_count} unknown = {assigned + unk_count} (Total: {total})")


# ─────────────────────────────────────────────────────────────────────────────
# LOST LOCATIONS TAB
# ─────────────────────────────────────────────────────────────────────────────


def render_locations_tab(df, total, dr, key_prefix=""):
    st.info("💡 **What's here:** Find where parcels are being lost. "
            "Filter by OTR or UTR to see those locations specifically.")
    verify_totals(df, total, "Locations Tab")
    filter_options = ["All Parcels", "OTR Only (Lost On Road)", "UTR Reprocess Only"]
    loc_filter = st.radio("Show:", filter_options, horizontal=True, key=f"{key_prefix}loc_filter")
    if loc_filter == "OTR Only (Lost On Road)":
        view_df = df[df["Sub Bucket"].str.contains("Lost On Road", na=False)].copy()
        filter_label = "OTR"
    elif loc_filter == "UTR Reprocess Only":
        view_df = df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"].copy()
        filter_label = "UTR"
    else:
        view_df = df.copy()
        filter_label = "All"
    view_count = len(view_df)
    st.write(f"**Showing: {view_count} parcels** ({filter_label})")
    if view_count == 0:
        st.warning("No parcels in this category.")
        return
    st.subheader(f"🏆 Top 10 Worst Locations ({filter_label})")
    rank_by = st.selectbox("Rank by:", ["Cluster", "Aisle", "Sort Zone"], key=f"{key_prefix}rb")
    rank_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"{key_prefix}rv")
    rank_data = view_df[rank_by].dropna().value_counts().head(10)
    if len(rank_data) > 0:
        if rank_view == "Chart":
            st.pyplot(make_bar_horiz(rank_data, f"Top 10 {rank_by}s — {filter_label} ({dr})", color="darkred"))
        else:
            st.dataframe(make_table(rank_data, rank_by, "Lost Parcels"))
    else:
        st.info("No location data available for this filter.")
    st.markdown("---")
    st.subheader(f"🔍 Cluster Drill-Down ({filter_label})")
    clusters = sorted(view_df["Cluster"].dropna().unique())
    if clusters:
        sel_cluster = st.selectbox("Select Cluster:", clusters, key=f"{key_prefix}cl_sel")
        filtered = view_df[view_df["Cluster"] == sel_cluster]
        st.write(f"**{len(filtered)} parcels** in Cluster {sel_cluster}")
        drill_by = st.selectbox("View by:", ["Aisle", "Sort Zone"], key=f"{key_prefix}drill")
        drill_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"{key_prefix}dv")
        drill_data = filtered[drill_by].dropna().value_counts()
        if len(drill_data) > 0:
            if drill_view == "Chart":
                st.pyplot(make_bar_horiz(drill_data, f"Cluster {sel_cluster} — {drill_by}s ({filter_label})", color="steelblue"))
            else:
                st.dataframe(make_table(drill_data, drill_by, "Lost Parcels"))
        with st.expander(f"📦 All parcels in Cluster {sel_cluster} ({filter_label})"):
            show_cols = get_detail_cols(filtered, extra=["Tracking ID", "Aisle", "Sort Zone"])
            detail = filtered[show_cols].sort_values("DSP Name").reset_index(drop=True)
            detail.index = range(1, len(detail) + 1)
            st.dataframe(detail, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

mode = st.radio("Mode:", ["Single Station", "Multi-Station Compare"], horizontal=True, key="mode")

with st.expander("📖 How to get your data"):
    st.markdown("""
**Step 1 — PerfectMile:** Concessions Control Tower → L&U → Lost → Export CSV
**Step 2 — SCC:** Paste Tracking IDs → View Options → Select All → Export CSV
**Step 3 — Upload both below.** Perfect Mile = source of truth for total count.
    """)

if mode == "Single Station":
    st.subheader("Upload Data")
    col_pm, col_scc = st.columns(2)
    with col_pm:
        pm_file = st.file_uploader("📊 Perfect Mile (.csv)", type="csv", key="pm_upload")
    with col_scc:
        scc_file = st.file_uploader("📋 SCC (.csv)", type="csv", key="scc_upload")

    if pm_file is not None and scc_file is not None:
        pm_df = pd.read_csv(pm_file)
        scc_df = pd.read_csv(scc_file)
        pm_missing = [c for c in REQUIRED_PM_COLS if c not in pm_df.columns]
        if pm_missing: st.error(f"❌ PM missing: {', '.join(pm_missing)}"); st.stop()
        scc_missing = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
        if scc_missing: st.error(f"❌ SCC missing: {', '.join(scc_missing)}"); st.stop()
        found = [c for c in SENSITIVE_COLS if c in scc_df.columns]
        if found: st.warning(f"🔒 PII auto-removed: {', '.join(found)}")
        df = merge_data(pm_df, scc_df)
        total = len(df)
        if total == 0: st.warning("⚠️ No data."); st.stop()
        pm_total = len(pm_df)
        matched = df["Cluster"].notna().sum()
        unmatched = total - matched
        st.success(f"✅ **{total} lost parcels** (PM: {pm_total}, SCC: {len(scc_df)}, Matched: {matched})")
        if unmatched > 0: st.info(f"ℹ️ {unmatched} in PM not in SCC — included with limited detail.")
        if total != pm_total: st.error(f"🚨 MISMATCH: {total} vs PM {pm_total}")
        dr = get_date_range(df)
        st.subheader(f"Quick Summary ({dr})")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Lost", total)
        c2.metric("Worst Cluster", safe_top(df["Cluster"]))
        c3.metric("Worst Aisle", safe_top(df["Aisle"]))
        c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]
        c5.metric("Worst Shift", safe_top(sk) if len(sk) > 0 else "N/A")
        cl_r = df["Cluster"].dropna().value_counts()
        if len(cl_r) > 0:
            parts = [f"#{i+1} {cl} — {n} ({round(n/total*100,1)}%)" for i, (cl, n) in enumerate(cl_r.head(3).items())]
            st.info("🎯 **Cluster Priority:** " + " → ".join(parts))

        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📊 Summary", "📍 Lost Locations", "🚚 DSP",
            "💡 Suggested Opportunities", "📅 Day of Week", "💾 Export", "📋 Bridge"
        ])

        with t1:
            st.info("💡 Breakdown by size, cluster, and sub-bucket.")
            verify_totals(df, total, "Summary")
            view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="sum_v")
            if view == "Chart":
                sc = df["Size Category"].value_counts()
                if len(sc) > 0:
                    st.pyplot(make_bar_vert(sc, "Size", "Lost", f"Lost by Size ({dr})",
                              color=["green","orange","red","darkred","grey"][:len(sc)]))
                cc = df["Cluster"].dropna().value_counts()
                if len(cc) > 0: st.pyplot(make_bar_horiz(cc, f"Lost by Cluster ({dr})"))
                sb_c = df["Sub Bucket"].value_counts()
                if len(sb_c) > 0: st.pyplot(make_bar_horiz(sb_c, f"Lost by Sub Bucket ({dr})", color="teal"))
            else:
                st.dataframe(make_table(df["Size Category"].value_counts(), "Size", "Count"))
                pivot = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0)
                pivot["Total"] = pivot.sum(axis=1)
                st.dataframe(pivot)
                st.dataframe(make_table(df["Sub Bucket"].value_counts(), "Sub Bucket", "Count"), use_container_width=True)

        with t2:
            render_locations_tab(df, total, dr, key_prefix="single_")

        with t3:
            st.info("💡 Which DSPs are losing the most.")
            verify_totals(df, total, "DSP")
            dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dsp_v")
            dsp_data = df["DSP Name"].dropna().value_counts()
            if dsp_view == "Chart":
                if len(dsp_data) > 0: st.pyplot(make_bar_horiz(dsp_data, f"Lost by DSP ({dr})", color="orange", max_label=DSP_MAX))
            else:
                if len(dsp_data) > 0: st.dataframe(make_table(dsp_data.sort_index(), "DSP", "Lost"), use_container_width=True)
            with st.expander("📦 All parcels by DSP"):
                show_cols = get_detail_cols(df, extra=["Tracking ID", "DSP Name"])
                out = df[show_cols].sort_values("DSP Name").reset_index(drop=True)
                out.index = range(1, len(out) + 1)
                st.dataframe(out, use_container_width=True)

        with t4:
            render_opportunities_tab(df, total, dr, key_prefix="single_")

        with t5:
            st.info("💡 Which days have the most lost parcels.")
            verify_totals(df, total, "Day")
            if "Day of Week" in df.columns:
                day_data = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                day_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="day_v")
                if day_view == "Chart":
                    fig, ax = plt.subplots(figsize=CHART)
                    ax.plot(day_data.index, day_data.values, marker="o", color="green", linewidth=2, markersize=6)
                    for i, (d, v) in enumerate(day_data.items()):
                        ax.annotate(str(int(v)), xy=(i, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
                    ax.set_xlabel("Day", fontsize=8); ax.set_ylabel("Lost", fontsize=8)
                    ax.set_title(f"Lost by Day ({dr})", fontsize=9); ax.tick_params(labelsize=7)
                    plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig)
                else:
                    st.dataframe(make_table(day_data, "Day", "Lost"))
            else:
                st.warning("No day data available.")

        with t6:
            verify_totals(df, total, "Export")
            export_cols = [c for c in df.columns if c not in ["Prev Event DT", "previous_event_datetime", "bucket", "sub_bucket"]]
            st.download_button("⬇️ Download CSV", df[export_cols].to_csv(index=False), "Lost_Merged.csv", "text/csv")

        with t7:
            verify_totals(df, total, "Bridge")
            cl_c = df["Cluster"].dropna().value_counts()
            dsp_c = df["DSP Name"].dropna().value_counts()
            sb_c = df["Sub Bucket"].value_counts()
            sh_c = df[df["Shift"] != "Unknown"]["Shift"].value_counts()
            wc = cl_c.index[0] if len(cl_c) > 0 else "N/A"
            wc_n = int(cl_c.values[0]) if len(cl_c) > 0 else 0
            wd = dsp_c.index[0] if len(dsp_c) > 0 else "N/A"
            wd_n = int(dsp_c.values[0]) if len(dsp_c) > 0 else 0
            avg_d = dsp_c.mean() if len(dsp_c) > 0 else 1
            dm = round(wd_n / avg_d, 1) if avg_d > 0 else 1.0
            wsh = sh_c.index[0] if len(sh_c) > 0 else "N/A"
            wsh_n = int(sh_c.values[0]) if len(sh_c) > 0 else 0
            sl = "\n".join([f"  {s}: {int(sh_c.get(s,0))} ({round(int(sh_c.get(s,0))/total*100,1)}%)" for s in SHIFT_ORDER])
            sb_lines = "\n".join([f"  {sb}: {n} ({round(int(n)/total*100,1)}%)" for sb, n in sb_c.head(6).items()])
            cdet = ""
            for cn, cv in cl_c.head(3).items():
                ta = df[df["Cluster"]==cn]["Aisle"].dropna().value_counts().head(3)
                al = ", ".join([f"{a}({n})" for a, n in ta.items()])
                cdet += f"  {cn}: {cv} ({round(int(cv)/total*100,1)}%) — {al}\n"
            dlines = "\n".join([f"  {d}: {n}" for d, n in dsp_c.head(3).items()])
            acts = []
            def ac(t): acts.append(f"AC{len(acts)+1}: {t}")
            ac(f"PS focus on Cluster {wc} ({round(wc_n/total*100,1)}% of losts).")
            if dm >= 1.5: ac(f"DSP {wd} briefing — {dm}x avg.")
            ins = int(sb_c.get("Lost At Station - Inducted Not Stowed", 0))
            if ins > total*0.15: ac(f"NS stow audit — {ins} parcels ({round(ins/total*100,1)}%).")
            pnov = int(sb_c.get("Lost At Station - PNOV", 0))
            if pnov > total*0.3: ac(f"PNOV deep-dive — {pnov} parcels ({round(pnov/total*100,1)}%).")
            utr = int(sb_c.get("Lost At Station - UTR Reprocess", 0))
            if utr > total*0.1: ac(f"UTR review — {utr} parcels ({round(utr/total*100,1)}%).")
            otr_t = sum(int(sb_c.get(k,0)) for k in sb_c.index if "Lost On Road" in k)
            if otr_t > total*0.05: ac(f"OTR driver briefing — {otr_t} parcels ({round(otr_t/total*100,1)}%).")
            bridge = f"""Lost Parcels Bridge — DRM2
{dr}
TOTAL: {total} (Perfect Mile)
SHIFTS:\n{sl}
SUB BUCKETS:\n{sb_lines}
LOCATIONS:\n{cdet}
DSPs:\n{dlines}
ACTIONS:\n{chr(10).join(acts)}"""
            st.text_area("✏️ Edit bridge:", value=bridge, height=350, key="bridge_edit")

    elif pm_file is not None: st.info("👆 Upload SCC to complete.")
    elif scc_file is not None: st.info("👆 Upload Perfect Mile to complete.")
    else: st.info("👆 Upload both files above.")

else:
    st.subheader("Upload Station Data")
    st.caption("Upload PM + SCC pairs per station (2–5).")
    num = st.slider("Stations:", 2, 5, 2, key="num_stations")
    uploaded = {}
    for i in range(num):
        with st.expander(f"Station {i+1}", expanded=(i < 2)):
            col_a, col_b = st.columns(2)
            with col_a: pm_f = st.file_uploader(f"PM ({i+1})", type="csv", key=f"mc_pm_{i}")
            with col_b: scc_f = st.file_uploader(f"SCC ({i+1})", type="csv", key=f"mc_scc_{i}")
            if pm_f and scc_f: uploaded[i] = (pm_f, scc_f)

    if len(uploaded) >= 2:
        stations, names = {}, []
        for i, (pm_f, scc_f) in uploaded.items():
            pm_tmp, scc_tmp = pd.read_csv(pm_f), pd.read_csv(scc_f)
            merged_tmp = merge_data(pm_tmp, scc_tmp)
            if "Station" in scc_tmp.columns and len(scc_tmp["Station"].dropna()) > 0:
                name = scc_tmp["Station"].dropna().iloc[0]
            elif "location" in pm_tmp.columns and len(pm_tmp["location"].dropna()) > 0:
                name = pm_tmp["location"].dropna().iloc[0]
            else: name = f"Station {i+1}"
            stations[name] = merged_tmp; names.append(name)
        st.success(f"✅ Loaded: **{', '.join(names)}**")
        mc = st.columns(len(names))
        for i, n in enumerate(names):
            sk = stations[n][stations[n]["Shift"].isin(SHIFT_ORDER)]["Shift"]
            mc[i].metric(n, f"{len(stations[n])} lost")
            mc[i].caption(f"Worst shift: {safe_top(sk) if len(sk)>0 else 'N/A'}")

        t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Summary", "📍 Locations", "🚚 DSP", "💡 Opportunities", "📅 Day", "💾 Export"])
        with t1:
            fig, ax = plt.subplots(figsize=CHART)
            bars = ax.bar(names, [len(stations[n]) for n in names], color=STATION_COLORS[:len(names)])
            for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, str(int(b.get_height())), ha="center", fontsize=8)
            ax.set_ylabel("Lost", fontsize=8); ax.set_title("Total Lost by Station", fontsize=9); ax.tick_params(labelsize=7)
            plt.tight_layout(); st.pyplot(fig)
        with t2:
            sel = st.selectbox("Station:", names, key="mc_loc_stn")
            render_locations_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), key_prefix=f"mc_{sel}_")
        with t3:
            for i, n in enumerate(names):
                dsp = stations[n]["DSP Name"].dropna().value_counts().head(10)
                if len(dsp) > 0: st.pyplot(make_bar_horiz(dsp, f"{n} — DSPs", color=STATION_COLORS[i], max_label=DSP_MAX))
        with t4:
            sel = st.selectbox("Station:", names, key="mc_opp_stn")
            render_opportunities_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), key_prefix=f"mc_{sel}_")
        with t5:
            fig, ax = plt.subplots(figsize=CHART)
            for i, n in enumerate(names):
                if "Day of Week" in stations[n].columns:
                    dd = stations[n]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                    ax.plot(dd.index, dd.values, marker="o", label=n, color=STATION_COLORS[i], linewidth=2)
            ax.set_ylabel("Lost", fontsize=8); ax.set_title("Lost by Day", fontsize=9)
            ax.tick_params(labelsize=7); ax.legend(fontsize=7); plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig)
        with t6:
            for n in names:
                ec = [c for c in stations[n].columns if c not in ["Prev Event DT","previous_event_datetime","bucket","sub_bucket"]]
                st.download_button(f"⬇️ {n}", stations[n][ec].to_csv(index=False), f"Lost_{n}.csv", "text/csv", key=f"dl_{n}")
            combined = pd.concat([stations[n].assign(Station=n) for n in names], ignore_index=True)
            ec = [c for c in combined.columns if c not in ["Prev Event DT","previous_event_datetime","bucket","sub_bucket"]]
            st.download_button("⬇️ All", combined[ec].to_csv(index=False), "Lost_All.csv", "text/csv", key="dl_all")
    elif len(uploaded) == 1: st.warning("⚠️ Need at least 2 stations.")
    else: st.info("👆 Upload file pairs above.")
