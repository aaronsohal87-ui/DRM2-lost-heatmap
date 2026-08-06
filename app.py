import streamlit as st  # Streamlit web app framework
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Charts

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DRM2 Lost Heatmap", page_icon="📦", layout="wide")
st.title("📦 DRM2 Lost Parcel Heatmap")
st.markdown("---")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]  # Colors for multi-station charts
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # Fixed day order
SHIFT_ORDER = ["NS", "AM", "PM", "OTR"]  # Shift display order
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen", "OTR": "firebrick"}  # Shift chart colors
SHIFT_DEFINITIONS = {"NS": "00:00 – 09:59 (Night Sort — stow)", "AM": "10:00 – 13:59 (Pick, stage, dispatch)", "PM": "14:00 – 23:59 (Dispatch, RELO)", "OTR": "On The Road (DSP responsibility)"}
SHIFT_HOUR_MAP = {0:"NS",1:"NS",2:"NS",3:"NS",4:"NS",5:"NS",6:"NS",7:"NS",8:"NS",9:"NS",10:"AM",11:"AM",12:"AM",13:"AM",14:"PM",15:"PM",16:"PM",17:"PM",18:"PM",19:"PM",20:"PM",21:"PM",22:"PM",23:"PM"}
# Fixed shift assignment: sub-bucket directly tells us which shift was responsible
SUB_BUCKET_SHIFT_MAP = {"Lost At Station - Inducted Not Stowed":"NS","Lost At Station - Stowed Not Picked Up":"AM","Lost At Station - Debrief Receive(RTS)":"PM","Lost On Road - Attempted":"OTR","Lost On Road - Damage":"OTR","Lost On Road - No Further Status":"OTR"}
# PII columns to remove from SCC (Driver Id kept for OTR, Last Scan By kept for UTR)
SENSITIVE_COLS = ["Holder Name","City","Postal","Province","Ordering Order ID","Order Amount","Receivable Amount","Payment Method","District","Scheduled Delivery End Time"]
REQUIRED_SCC_COLS = ["Tracking ID","Sort Zone","Aisle","Cluster","Package Length","Package Width","Package Height","DSP Name","Assigned Cycle","Last Updated Time"]
REQUIRED_PM_COLS = ["tracking_id","sub_bucket"]
CHART = (7, 2.5)  # Default chart size (width, height)
DSP_MAX = 20  # Max chars for DSP name labels
LABEL_MAX = 25  # Max chars for general labels
EOD_SCRUB_HOUR_START = 22  # EoD scrub window start hour (inclusive)
EOD_SCRUB_HOUR_END = 23  # EoD scrub window end hour (inclusive) — scans in 22:00-23:59 are scrub operators

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_size(val):
    """Categorise parcel by longest side (cm)."""
    if pd.isna(val): return "Unknown"
    if val <= 35: return "Small"
    if val <= 45: return "Medium"
    if val <= 61: return "Small Oversize"
    return "Large Oversize"

def hour_to_shift(hour):
    """Map hour (0-23) to shift name."""
    if pd.isna(hour): return "Unknown"
    return SHIFT_HOUR_MAP.get(int(hour), "Unknown")

def assign_shift(row):
    """Assign responsibility shift to a parcel.
    Priority: 1) Sub-bucket (deterministic) → 2) previous_event_datetime hour
    → 3) Dispatch Time hour → 4) Assigned Cycle text → 5) Unknown."""
    sb = row.get("Sub Bucket", "")
    # Step 1: Check if sub-bucket directly maps to a shift
    if sb in SUB_BUCKET_SHIFT_MAP: return SUB_BUCKET_SHIFT_MAP[sb]
    # Step 2: Check previous_event_datetime (last event before loss)
    prev_dt = row.get("Prev Event DT")
    if pd.notna(prev_dt): return hour_to_shift(prev_dt.hour)
    # Step 3: Check Dispatch Time from SCC
    disp_dt = row.get("Dispatch Time")
    if pd.notna(disp_dt): return hour_to_shift(disp_dt.hour)
    # Step 4: Check Assigned Cycle text from SCC
    cyc = row.get("Assigned Cycle", "")
    if pd.notna(cyc):
        u = str(cyc).upper()
        if "NS" in u or "NIGHT" in u: return "NS"
        if "PM" in u or "RELO" in u or "C2" in u: return "PM"
        if "AM" in u or "C1" in u: return "AM"
    # Step 5: Could not determine
    return "Unknown"

def is_eod_scrub(row):
    """Detect if the Last Scan By is likely an EoD scrub operator.
    EoD scrub scans happen in a late-night window (22:00-23:59).
    The person running the scrub is NOT the person who lost the parcel —
    they just flagged it during the end-of-day audit.
    General solution: any scan in the scrub window is excluded from associate analysis."""
    lut = row.get("Last Updated Time")
    if pd.notna(lut):
        try:
            hour = lut.hour
            if EOD_SCRUB_HOUR_START <= hour <= EOD_SCRUB_HOUR_END:
                return True
        except (AttributeError, TypeError):
            pass
    return False

def clean_scc(df):
    """Clean SCC data: remove PII, extract associate + driver, parse dimensions."""
    # Remove sensitive PII columns
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])
    # Extract associate alias from Last Scan By email (for UTR analysis)
    if "Last Scan By" in df.columns:
        df["Associate"] = df["Last Scan By"].astype(str).str.split("@").str[0].replace("nan", pd.NA)
        df = df.drop(columns=["Last Scan By"])  # Drop full email, keep alias only
    # Keep Driver Id for OTR analysis
    if "Driver Id" in df.columns:
        df["Driver"] = df["Driver Id"].astype(str).replace("nan", pd.NA)
        df = df.drop(columns=["Driver Id"])
    # Parse package dimensions (strip "cm" text, convert to numeric)
    for col in ["Package Length","Package Width","Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm","").str.replace("cm","")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Calculate longest side and size category
    dims = ["Package Length","Package Width","Package Height"]
    df["Longest Side"] = df[dims].max(axis=1) if all(c in df.columns for c in dims) else float("nan")
    df["Size Category"] = df["Longest Side"].apply(get_size)
    # Parse datetime columns
    if "Last Updated Time" in df.columns: df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], dayfirst=True, errors="coerce")
    if "Dispatch Time" in df.columns: df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], dayfirst=True, errors="coerce")
    return df

def merge_data(pm_df, scc_df):
    """Merge Perfect Mile (master, source of truth) with SCC (detail).
    LEFT JOIN so every PM parcel is kept even if not in SCC.
    Day of Week = PM event_datetime (date parcel was marked as lost)."""
    scc_clean = clean_scc(scc_df.copy())
    # Select relevant PM columns
    pm_keep = ["tracking_id","bucket","sub_bucket","previous_event_datetime","previous_reason","previous_reason_3","event_datetime"]
    pm_cols = pm_df[[c for c in pm_keep if c in pm_df.columns]].copy()
    pm_cols = pm_cols.rename(columns={"tracking_id":"Tracking ID"})
    # Parse previous_event_datetime (some may be corrupt format MM:SS.s — coerce handles those)
    pm_cols["Prev Event DT"] = pd.to_datetime(pm_cols.get("previous_event_datetime", pd.Series(dtype="object")), format="%d/%m/%Y %H:%M", errors="coerce")
    # Parse event_datetime = when parcel was marked as lost
    if "event_datetime" in pm_cols.columns:
        pm_cols["Marked Lost DT"] = pd.to_datetime(pm_cols["event_datetime"], dayfirst=True, errors="coerce")
    # LEFT JOIN: PM is master, SCC adds detail
    merged = pm_cols.merge(scc_clean, on="Tracking ID", how="left")
    merged["Sub Bucket"] = merged["sub_bucket"]
    merged["Bucket"] = merged.get("bucket")
    # Day of Week from the date the parcel was MARKED AS LOST (not dispatch, not EoD scrub)
    if "Marked Lost DT" in merged.columns:
        merged["Day of Week"] = merged["Marked Lost DT"].dt.day_name()
    elif "Dispatch Time" in merged.columns:
        merged["Day of Week"] = merged["Dispatch Time"].dt.day_name()
    else:
        merged["Day of Week"] = None
    # Loss reasons from PM
    if "previous_reason" in merged.columns:
        merged["Loss Reason"] = merged["previous_reason"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else:
        merged["Loss Reason"] = "Unknown"
    if "previous_reason_3" in merged.columns:
        merged["UTR Reason"] = merged["previous_reason_3"].replace({"NOREASON":"No Reason","NONE":"No Reason"}).fillna("Unknown")
    else:
        merged["UTR Reason"] = "Unknown"
    # Assign shift responsibility
    merged["Shift"] = merged.apply(assign_shift, axis=1)
    # Flag EoD scrub scans (associate is scrub operator, not the person who lost it)
    merged["Is EoD Scrub"] = merged.apply(is_eod_scrub, axis=1)
    # Ensure all expected columns exist
    for col in ["Cluster","Aisle","Sort Zone","DSP Name","Size Category","Associate","Driver"]:
        if col not in merged.columns: merged[col] = None
    return merged

def get_date_range(df):
    """Get earliest-latest date string from available datetime columns."""
    for col in ["Marked Lost DT","Dispatch Time","Last Updated Time"]:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                s, e = valid.min().strftime("%d %b %Y"), valid.max().strftime("%d %b %Y")
                return s if s == e else f"{s} – {e}"
    return ""

def safe_top(s):
    """Get the most common value in a series (or N/A if empty)."""
    c = s.dropna().value_counts(); return c.index[0] if len(c) > 0 else "N/A"

def trunc(labels, mx=LABEL_MAX):
    """Truncate long labels for chart readability."""
    return [str(l)[:mx]+"..." if len(str(l))>mx else str(l) for l in labels]

def get_detail_cols(df, extra=None):
    """Get a sensible default set of columns to show in detail tables."""
    base = ["Tracking ID","Cluster","Aisle","Sort Zone","DSP Name","Size Category","Shift","Sub Bucket"]
    if extra: base = extra + [c for c in base if c not in extra]
    return [c for c in base if c in df.columns]

def verify_totals(df, total, label=""):
    """Check that dataframe length matches expected total — shows error if mismatch."""
    if len(df) != total: st.error(f"⚠️ MISMATCH {label}: Expected {total}, got {len(df)}."); return False
    return True

def make_table(series, c1, c2):
    """Convert a value_counts series into a nice display table."""
    t = series.reset_index(); t.columns = [c1, c2]; t.index = range(1, len(t)+1); return t

# ─── CHARTS ───────────────────────────────────────────────────────────────────
def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    """Horizontal bar chart (sorted high to low)."""
    h = max(2, len(data)*0.3); fig, ax = plt.subplots(figsize=(figsize_width, h))
    labs = trunc(data.index, max_label); ax.barh(labs, data.values, color=color); ax.invert_yaxis()
    for i, v in enumerate(data.values): ax.text(v+0.2, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Lost Parcels",fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); return fig

def make_bar_vert(data, xl, yl, title, color="steelblue", figsize=CHART):
    """Vertical bar chart."""
    fig, ax = plt.subplots(figsize=figsize); labs = trunc(data.index, LABEL_MAX); ax.bar(labs, data.values, color=color)
    for i, v in enumerate(data.values): ax.text(i, v+0.2, str(int(v)), ha="center", fontsize=7)
    ax.set_xlabel(xl,fontsize=8); ax.set_ylabel(yl,fontsize=8); ax.set_title(title,fontsize=9)
    ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); return fig

def make_bar_shift(data, title):
    """Bar chart with shift-specific colors."""
    data = data.reindex(SHIFT_ORDER, fill_value=0); fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER], color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, str(int(b.get_height())), ha="center", fontsize=7)
    ax.set_xlabel("Shift",fontsize=8); ax.set_ylabel("Lost",fontsize=8); ax.set_title(title,fontsize=9)
    ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); return fig

def make_pie_otr_utr(df, total, title):
    """Small pie chart showing OTR vs UTR vs Other split."""
    otr_n = len(df[df["Sub Bucket"].str.contains("Lost On Road", na=False)])
    utr_n = len(df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"])
    other_n = total - otr_n - utr_n
    labels, sizes, colors, explode = [], [], [], []
    if otr_n > 0: labels.append(f"OTR ({otr_n})"); sizes.append(otr_n); colors.append("firebrick"); explode.append(0.05)
    if utr_n > 0: labels.append(f"UTR ({utr_n})"); sizes.append(utr_n); colors.append("darkorange"); explode.append(0.05)
    if other_n > 0: labels.append(f"Other ({other_n})"); sizes.append(other_n); colors.append("steelblue"); explode.append(0)
    fig, ax = plt.subplots(figsize=(3, 2.2))  # Intentionally small
    ax.pie(sizes, labels=labels, colors=colors, explode=explode, autopct="%1.1f%%", startangle=90, textprops={"fontsize":6})
    ax.set_title(title, fontsize=7); plt.tight_layout(); return fig

# ─── MISSING PARCELS VIEWER ──────────────────────────────────────────────────
def render_missing_parcels(df, total, matched):
    """Show parcels in PM that had no SCC match, with a selectbox to inspect each one."""
    missing_count = total - matched
    if missing_count > 0:
        st.info(f"ℹ️ **{missing_count} parcel(s)** in Perfect Mile had no matching row in SCC — "
                "they are included in totals but have no cluster/aisle/DSP detail because SCC didn't have them.")
        missing_df = df[df["Cluster"].isna()].copy()
        if len(missing_df) > 0:
            with st.expander(f"🔍 View {len(missing_df)} Missing Parcel(s)"):
                sel_tid = st.selectbox("Select parcel:", missing_df["Tracking ID"].tolist(), key="miss_sel")
                row = missing_df[missing_df["Tracking ID"] == sel_tid].iloc[0]
                st.markdown(f"**Tracking ID:** {sel_tid}")
                st.markdown(f"**Sub Bucket:** {row.get('Sub Bucket', 'N/A')}")
                st.markdown(f"**Shift:** {row.get('Shift', 'N/A')}")
                st.markdown(f"**Loss Reason:** {row.get('Loss Reason', 'N/A')}")
                st.markdown(f"**Day Marked Lost:** {row.get('Day of Week', 'N/A')}")
                st.caption("This parcel exists in Perfect Mile but has no corresponding row in SCC — "
                          "it may not have been scanned into SCC, or the Tracking ID format differs.")

# ─── PEOPLE TAB (OTR → Drivers, UTR → Associates) ────────────────────────────
def render_people_tab(df, total, dr, kp=""):
    """Shows who was last responsible before loss:
    - OTR parcels → Driver Id + DSP (the driver who had it on road)
    - UTR parcels → Associate login (last person to scan in station, excluding EoD scrub)
    This builds trends over time to identify repeat patterns."""
    st.markdown("### 👤 People — OTR Drivers & UTR Associates")
    st.info("💡 **Purpose:** Build trends over time to spot repeat patterns.\n\n"
            "- **OTR (Lost On Road):** The **Driver Id** from SCC tells us which driver had the parcel when it went missing. "
            "We also show which DSP they belong to. Tracking this over time helps identify drivers who may need coaching.\n\n"
            "- **UTR (Lost At Station):** The **Last Scan By** from SCC tells us which associate last scanned the parcel "
            "before it was marked lost. **EoD scrub operators are excluded** — if the last scan was during the EoD scrub window "
            f"({EOD_SCRUB_HOUR_START}:00–{EOD_SCRUB_HOUR_END}:59), that person was just running the audit, not the one who lost it.\n\n"
            "⚠️ This is about **identifying trends for improvement**, not blame. One appearance means nothing — "
            "repeated appearances across multiple weeks suggest a coaching opportunity.")
    st.markdown("---")

    # ─── OTR SECTION: DRIVERS + DSP ──────────────────────────────────────────
    st.markdown("#### 🚚 OTR — Driver Trend")
    otr_df = df[df["Sub Bucket"].str.contains("Lost On Road", na=False)].copy()
    if len(otr_df) == 0:
        st.success("No OTR parcels — no driver issues this period.")
    else:
        st.write(f"**{len(otr_df)} OTR parcels**")
        if "Driver" in otr_df.columns and otr_df["Driver"].dropna().nunique() > 0:
            # Build a Driver → DSP mapping so we can show DSP alongside driver
            driver_dsp = otr_df[["Driver","DSP Name"]].dropna().drop_duplicates().set_index("Driver")["DSP Name"].to_dict()
            with st.expander("🚚 Drivers with Most OTR Losses"):
                drv_data = otr_df["Driver"].dropna().value_counts()
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}drv_v")
                if vm == "Chart":
                    # Show Driver (DSP) on chart labels
                    chart_labels = pd.Series(drv_data.values, index=[f"{d} ({driver_dsp.get(d,'?')})" for d in drv_data.index])
                    st.pyplot(make_bar_horiz(chart_labels.head(15), f"OTR by Driver ({dr})", color="firebrick", max_label=35))
                else:
                    # Table with Driver Id, DSP, Count
                    tbl = drv_data.reset_index(); tbl.columns = ["Driver Id", "OTR Parcels Lost"]
                    tbl["DSP"] = tbl["Driver Id"].map(driver_dsp).fillna("Unknown")
                    tbl = tbl[["Driver Id","DSP","OTR Parcels Lost"]]
                    tbl.index = range(1, len(tbl)+1)
                    st.dataframe(tbl, use_container_width=True)
            with st.expander("🚚 Driver Drill-Down"):
                drivers = sorted(otr_df["Driver"].dropna().unique())
                if drivers:
                    # Show driver with DSP in selectbox
                    driver_labels = [f"{d} ({driver_dsp.get(d,'?')})" for d in drivers]
                    sel_idx = st.selectbox("Select Driver:", range(len(drivers)), format_func=lambda i: driver_labels[i], key=f"{kp}drv_sel")
                    sel = drivers[sel_idx]
                    d_df = otr_df[otr_df["Driver"] == sel]
                    st.write(f"**{len(d_df)} OTR parcels** by Driver {sel}")
                    st.markdown(f"**DSP:** {driver_dsp.get(sel, 'Unknown')}")
                    # Show reasons breakdown
                    reasons = d_df["Loss Reason"].dropna().value_counts()
                    if len(reasons) > 0:
                        st.markdown("**Loss Reasons:**")
                        st.dataframe(make_table(reasons, "Reason", "Count"), use_container_width=True)
                    # Detail table
                    show_cols = [c for c in ["Tracking ID","Sub Bucket","Loss Reason","DSP Name","Day of Week"] if c in d_df.columns]
                    out = d_df[show_cols].reset_index(drop=True); out.index = range(1, len(out)+1)
                    st.dataframe(out, use_container_width=True)
        else:
            st.warning("No Driver Id data available in SCC for OTR parcels.")

    st.markdown("---")

    # ─── UTR SECTION: ASSOCIATES (excluding EoD scrub) ────────────────────────
    st.markdown("#### 🏭 UTR — Associate Trend")
    st.caption(f"ℹ️ Scans in the EoD scrub window ({EOD_SCRUB_HOUR_START}:00–{EOD_SCRUB_HOUR_END}:59) are excluded — "
               "those associates were running the audit, not handling the parcel operationally.")
    utr_df = df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"].copy()
    if len(utr_df) == 0:
        st.success("No UTR parcels — no associate issues this period.")
    else:
        # Exclude EoD scrub operators — their scan is the audit, not the loss
        utr_operational = utr_df[~utr_df["Is EoD Scrub"]].copy()
        scrub_excluded = len(utr_df) - len(utr_operational)
        st.write(f"**{len(utr_df)} UTR parcels** ({scrub_excluded} excluded as EoD scrub scans)")
        if "Associate" in utr_operational.columns and utr_operational["Associate"].dropna().nunique() > 0:
            with st.expander("👤 Associates with Most UTR Losses"):
                assoc_data = utr_operational["Associate"].dropna().value_counts()
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}assoc_v")
                if vm == "Chart":
                    st.pyplot(make_bar_horiz(assoc_data.head(15), f"UTR by Associate ({dr})", color="purple"))
                else:
                    # Table with associate, count, and last TID they scanned
                    tbl_rows = []
                    for assoc, count in assoc_data.items():
                        # Get the most recent UTR parcel this associate scanned
                        a_parcels = utr_operational[utr_operational["Associate"] == assoc]
                        # Sort by Last Updated Time to get most recent, fallback to first row
                        if "Last Updated Time" in a_parcels.columns:
                            sorted_p = a_parcels.sort_values("Last Updated Time", ascending=False, na_position="last")
                        else:
                            sorted_p = a_parcels
                        last_tid = sorted_p["Tracking ID"].iloc[0] if len(sorted_p) > 0 else "N/A"
                        tbl_rows.append({"Associate": assoc, "UTR Parcels": int(count), "Last UTR Scan TID": last_tid})
                    tbl = pd.DataFrame(tbl_rows); tbl.index = range(1, len(tbl)+1)
                    st.dataframe(tbl, use_container_width=True)
            with st.expander("👤 Associate Drill-Down"):
                assocs = sorted(utr_operational["Associate"].dropna().unique())
                if assocs:
                    sel = st.selectbox("Select Associate:", assocs, key=f"{kp}assoc_sel")
                    a_df = utr_operational[utr_operational["Associate"] == sel]
                    st.write(f"**{len(a_df)} UTR parcels** last scanned by {sel}")
                    # Show most recent TID
                    if "Last Updated Time" in a_df.columns:
                        sorted_a = a_df.sort_values("Last Updated Time", ascending=False, na_position="last")
                    else:
                        sorted_a = a_df
                    last_tid = sorted_a["Tracking ID"].iloc[0] if len(sorted_a) > 0 else "N/A"
                    st.markdown(f"**Most recent UTR scan:** `{last_tid}`")
                    # Show UTR reasons
                    reasons = a_df["UTR Reason"].dropna().value_counts()
                    if len(reasons) > 0:
                        st.markdown("**UTR Reasons:**")
                        st.dataframe(make_table(reasons, "Reason", "Count"), use_container_width=True)
                    # Show locations
                    clusters = a_df["Cluster"].dropna().value_counts()
                    if len(clusters) > 0:
                        st.markdown("**Locations:**")
                        st.dataframe(make_table(clusters, "Cluster", "Count"), use_container_width=True)
                    # Detail table
                    show_cols = [c for c in ["Tracking ID","Sub Bucket","UTR Reason","Cluster","Aisle","Day of Week"] if c in a_df.columns]
                    out = a_df[show_cols].reset_index(drop=True); out.index = range(1, len(out)+1)
                    st.dataframe(out, use_container_width=True)
        else:
            st.warning("No Associate data available in SCC for UTR parcels (after excluding EoD scrub).")

# ─── LOCATIONS TAB ────────────────────────────────────────────────────────────
def render_locations_tab(df, total, dr, kp=""):
    """Location analysis: All parcels (cluster/aisle), OTR (DSP), UTR (reasons)."""
    st.info("💡 **OTR** = DSP + reasons. **UTR** = loss reasons + locations. **All** = cluster/aisle drill-down.")
    verify_totals(df, total, "Locations")
    lf = st.radio("Show:", ["All Parcels","OTR Only (Lost On Road)","UTR Reprocess Only"], horizontal=True, key=f"{kp}lf")
    if lf == "OTR Only (Lost On Road)":
        vdf = df[df["Sub Bucket"].str.contains("Lost On Road", na=False)].copy()
        st.write(f"**{len(vdf)} OTR parcels**")
        if len(vdf) == 0: st.warning("No OTR parcels."); return
        with st.expander("🚚 OTR by DSP"):
            d = vdf["DSP Name"].dropna().value_counts()
            if len(d) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}od")
                if vm == "Chart": st.pyplot(make_bar_horiz(d, f"OTR by DSP ({dr})", color="firebrick", max_label=DSP_MAX))
                else: st.dataframe(make_table(d, "DSP", "Lost"), use_container_width=True)
        with st.expander("🚚 DSP → Reason"):
            dsps = sorted(vdf["DSP Name"].dropna().unique())
            if dsps:
                sd = st.selectbox("DSP:", dsps, key=f"{kp}ods")
                ddf = vdf[vdf["DSP Name"] == sd]; st.write(f"**{len(ddf)}** by {sd}")
                r = ddf["Loss Reason"].dropna().value_counts()
                if len(r) > 0: st.dataframe(make_table(r, "Reason", "Count"), use_container_width=True)
        with st.expander("❓ All OTR Reasons"):
            r = vdf["Loss Reason"].dropna().value_counts()
            if len(r) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}or")
                if vm == "Chart": st.pyplot(make_bar_horiz(r, f"OTR Reasons ({dr})", color="crimson"))
                else: st.dataframe(make_table(r, "Reason", "Count"), use_container_width=True)
    elif lf == "UTR Reprocess Only":
        vdf = df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"].copy()
        st.write(f"**{len(vdf)} UTR parcels**")
        if len(vdf) == 0: st.warning("No UTR parcels."); return
        with st.expander("❓ UTR Reasons"):
            r = vdf["UTR Reason"].dropna().value_counts()
            if len(r) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}ur")
                if vm == "Chart": st.pyplot(make_bar_horiz(r, f"UTR Reasons ({dr})", color="darkorange"))
                else: st.dataframe(make_table(r, "Reason", "Count"), use_container_width=True)
            else: st.info("No reasons recorded.")
        with st.expander("📍 UTR by Location"):
            cl = vdf["Cluster"].dropna().value_counts()
            if len(cl) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}ul")
                if vm == "Chart": st.pyplot(make_bar_horiz(cl, f"UTR Clusters ({dr})", color="darkorange"))
                else: st.dataframe(make_table(cl, "Cluster", "Count"), use_container_width=True)
    else:
        vdf = df.copy(); st.write(f"**{len(vdf)} parcels (all)**")
        with st.expander("🏆 Top 10 Locations"):
            rb = st.selectbox("Rank by:", ["Cluster","Aisle","Sort Zone"], key=f"{kp}rb")
            rd = vdf[rb].dropna().value_counts().head(10)
            if len(rd) > 0:
                vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}lv")
                if vm == "Chart": st.pyplot(make_bar_horiz(rd, f"Top 10 {rb}s ({dr})", color="darkred"))
                else: st.dataframe(make_table(rd, rb, "Lost"), use_container_width=True)
        with st.expander("🔍 Cluster Drill-Down"):
            clusters = sorted(vdf["Cluster"].dropna().unique())
            if clusters:
                sel = st.selectbox("Cluster:", clusters, key=f"{kp}cl")
                filt = vdf[vdf["Cluster"] == sel]; st.write(f"**{len(filt)}** in Cluster {sel}")
                ad = filt["Aisle"].dropna().value_counts()
                if len(ad) > 0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key=f"{kp}cv")
                    if vm == "Chart": st.pyplot(make_bar_horiz(ad, f"Cluster {sel} Aisles", color="steelblue"))
                    else: st.dataframe(make_table(ad, "Aisle", "Lost"), use_container_width=True)
                sc = get_detail_cols(filt, extra=["Tracking ID","Aisle","Sort Zone"])
                det = filt[sc].sort_values("DSP Name").reset_index(drop=True); det.index = range(1, len(det)+1)
                st.dataframe(det, use_container_width=True)

# ─── OPPORTUNITIES TAB ────────────────────────────────────────────────────────
def render_opportunities_tab(df, total, dr, kp=""):
    """Shift responsibility analysis — who should action each lost parcel."""
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
    sc = df[df["Shift"] != "Unknown"]["Shift"].value_counts()
    unk = len(df[df["Shift"] == "Unknown"])
    with st.expander("🏆 Shift Opportunity Leaderboard"):
        rows = []
        for s in SHIFT_ORDER:
            n = int(sc.get(s, 0)); rows.append({"Shift": s, "Lost": n, "%": f"{round(n/total*100,1)}%", "Window": SHIFT_DEFINITIONS[s]})
        rows.sort(key=lambda r: r["Lost"], reverse=True)
        lb = pd.DataFrame(rows); lb.index = range(1, len(lb)+1)
        st.dataframe(lb, use_container_width=True, height=200)
        if len(sc) > 0: st.pyplot(make_bar_shift(sc, f"Lost by Shift ({dr})"))
    with st.expander("🔍 Shift Drill-Down"):
        opts = [f"{s} — {int(sc.get(s,0))} parcels" for s in SHIFT_ORDER]
        if unk > 0: opts.append(f"Unknown — {unk} parcels")
        sel = st.selectbox("Shift:", opts, key=f"{kp}os"); ss = sel.split(" — ")[0]
        sdf = df[df["Shift"] == ss]; cnt = len(sdf)
        if cnt > 0:
            st.markdown(f"**{ss}** — **{cnt} parcels** ({round(cnt/total*100,1)}%)")
            sbc = sdf["Sub Bucket"].value_counts(); sbt = sbc.reset_index(); sbt.columns = ["Sub Bucket","Count"]
            sbt["%"] = (sbt["Count"]/cnt*100).round(1).astype(str)+"%"; sbt.index = range(1, len(sbt)+1)
            vm = st.radio("Display:", ["Table","Chart"], horizontal=True, key=f"{kp}sd")
            if vm == "Chart": st.pyplot(make_bar_horiz(sbc, f"{ss} Sub Buckets", color=SHIFT_COLORS.get(ss,"steelblue")))
            else: st.dataframe(sbt, use_container_width=True)
            sc2 = [c for c in ["Tracking ID","Sub Bucket","Cluster","Aisle","DSP Name","Loss Reason"] if c in df.columns]
            det = sdf[sc2].sort_values("Sub Bucket").reset_index(drop=True); det.index = range(1, len(det)+1)
            st.dataframe(det, use_container_width=True)
    st.markdown("---")
    st.caption(f"✅ Verification: {len(df[df['Shift']!='Unknown'])} assigned + {unk} unknown = {total}")

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
mode = st.radio("Mode:", ["Single Station","Multi-Station Compare"], horizontal=True, key="mode")
with st.expander("📖 How to get your data"):
    st.markdown("**1.** PerfectMile → L&U → Lost → Export CSV\n**2.** SCC → paste TIDs → Select All → Export CSV\n**3.** Upload both. PM = source of truth for total count.")

if mode == "Single Station":
    st.subheader("Upload Data")
    col_pm, col_scc = st.columns(2)
    with col_pm: pm_file = st.file_uploader("📊 Perfect Mile (.csv)", type="csv", key="pm_up")
    with col_scc: scc_file = st.file_uploader("📋 SCC (.csv)", type="csv", key="scc_up")
    if pm_file and scc_file:
        pm_df, scc_df = pd.read_csv(pm_file), pd.read_csv(scc_file)
        # Validate required columns
        pm_miss = [c for c in REQUIRED_PM_COLS if c not in pm_df.columns]
        if pm_miss: st.error(f"❌ PM missing: {pm_miss}"); st.stop()
        scc_miss = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
        if scc_miss: st.error(f"❌ SCC missing: {scc_miss}"); st.stop()
        # Warn about PII removal
        found = [c for c in SENSITIVE_COLS if c in scc_df.columns]
        if found: st.warning(f"🔒 PII removed: {', '.join(found)}")
        # Merge and process
        df = merge_data(pm_df, scc_df); total = len(df)
        if total == 0: st.warning("No data."); st.stop()
        pm_total = len(pm_df); matched = df["Cluster"].notna().sum()
        st.success(f"✅ **{total} lost parcels** (PM:{pm_total}, SCC:{len(scc_df)}, Matched:{matched})")
        # Show missing parcels viewer (PM parcels not in SCC)
        render_missing_parcels(df, total, matched)
        # Check for total mismatch
        if total != pm_total: st.error(f"🚨 MISMATCH: {total} vs PM {pm_total}")
        dr = get_date_range(df)
        # Quick summary metrics
        st.subheader(f"Quick Summary ({dr})")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Lost", total); c2.metric("Worst Cluster", safe_top(df["Cluster"]))
        c3.metric("Worst Aisle", safe_top(df["Aisle"])); c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]; c5.metric("Worst Shift", safe_top(sk) if len(sk) > 0 else "N/A")
        # Main tabs
        t1,t2,t3,t4,t5,t6,t7 = st.tabs(["📊 Summary","📍 Lost Locations","💡 Opportunities","👤 People","📅 Day of Week","💾 Export","📋 Bridge"])
        with t1:
            verify_totals(df, total, "Summary")
            with st.expander("🥧 OTR & UTR"): st.pyplot(make_pie_otr_utr(df, total, f"OTR & UTR vs Other ({dr})"))
            with st.expander("📏 Parcel Size Breakdown"):
                sc = df["Size Category"].value_counts()
                if len(sc) > 0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="sz")
                    if vm == "Chart": st.pyplot(make_bar_vert(sc,"Size","Lost",f"Parcel Size ({dr})", color=["green","orange","red","darkred","grey"][:len(sc)]))
                    else: st.dataframe(make_table(sc,"Size","Count"), use_container_width=True)
            with st.expander("📍 Cluster Breakdown"):
                cc = df["Cluster"].dropna().value_counts()
                if len(cc) > 0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="cv")
                    if vm == "Chart": st.pyplot(make_bar_horiz(cc, f"By Cluster ({dr})"))
                    else: st.dataframe(make_table(cc,"Cluster","Count"), use_container_width=True)
            with st.expander("🏷️ Lost Sub Bucket Breakdown"):
                sb = df["Sub Bucket"].value_counts()
                if len(sb) > 0:
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="sv")
                    if vm == "Chart": st.pyplot(make_bar_horiz(sb, f"Sub Bucket ({dr})", color="teal"))
                    else: st.dataframe(make_table(sb,"Sub Bucket","Count"), use_container_width=True)
        with t2: render_locations_tab(df, total, dr, kp="s_")
        with t3: render_opportunities_tab(df, total, dr, kp="s_")
        with t4: render_people_tab(df, total, dr, kp="s_")
        with t5:
            verify_totals(df, total, "Day")
            st.markdown("""**📌 How Day of Week is calculated:**

Each parcel has an `event_datetime` in Perfect Mile — this is the exact date and time the system **marked that parcel as lost**.
We take the day of the week from that date. For example, if a parcel was marked lost on 28/07/2026 (a Tuesday), it counts as Tuesday.

This gives a natural spread across the week because parcels are flagged on different days as they're identified missing.
We do NOT use the EoD scrub time (which would bunch everything on Sun/Mon) or the dispatch date (only available for dispatched parcels).""")
            if "Day of Week" in df.columns:
                dd = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                with st.expander("📅 Day of Week"):
                    vm = st.radio("Display:", ["Chart","Table"], horizontal=True, key="dv")
                    if vm == "Chart":
                        fig, ax = plt.subplots(figsize=CHART)
                        ax.plot(dd.index, dd.values, marker="o", color="green", linewidth=2, markersize=6)
                        for i,(d,v) in enumerate(dd.items()):
                            ax.annotate(str(int(v)), xy=(i,v), xytext=(0,8), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
                        ax.set_xlabel("Day",fontsize=8); ax.set_ylabel("Lost",fontsize=8); ax.set_title(f"Lost by Day Marked ({dr})",fontsize=9)
                        ax.tick_params(labelsize=7); plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig)
                    else: st.dataframe(make_table(dd,"Day","Lost"), use_container_width=True)
                    st.caption(f"ℹ️ {df['Day of Week'].notna().sum()}/{total} parcels have date data.")
        with t6:
            verify_totals(df, total, "Export")
            exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","Marked Lost DT","Is EoD Scrub"]
            ec = [c for c in df.columns if c not in exc]
            st.download_button("⬇️ Download CSV", df[ec].to_csv(index=False), "Lost_Merged.csv", "text/csv")
        with t7:
            verify_totals(df, total, "Bridge")
            cl_c = df["Cluster"].dropna().value_counts(); sb_c = df["Sub Bucket"].value_counts()
            sh_c = df[df["Shift"]!="Unknown"]["Shift"].value_counts()
            sl = "\n".join([f"  {s}: {int(sh_c.get(s,0))} ({round(int(sh_c.get(s,0))/total*100,1)}%)" for s in SHIFT_ORDER])
            sbl = "\n".join([f"  {sb}: {n} ({round(int(n)/total*100,1)}%)" for sb,n in sb_c.head(6).items()])
            cdet = ""
            for cn,cv in cl_c.head(3).items():
                ta = df[df["Cluster"]==cn]["Aisle"].dropna().value_counts().head(3)
                cdet += f"  {cn}: {cv} ({round(int(cv)/total*100,1)}%) — {', '.join([f'{a}({n})' for a,n in ta.items()])}\n"
            bridge = f"Lost Parcels Bridge — DRM2\n{dr}\nTOTAL: {total}\nSHIFTS:\n{sl}\nSUB BUCKETS:\n{sbl}\nLOCATIONS:\n{cdet}"
            st.text_area("✏️ Bridge:", value=bridge, height=300, key="bridge")
    elif pm_file: st.info("👆 Upload SCC.")
    elif scc_file: st.info("👆 Upload PM.")
    else: st.info("👆 Upload both files above.")

else:
    # ─── MULTI-STATION MODE ───────────────────────────────────────────────────
    st.subheader("Upload Station Data")
    num = st.slider("Stations:", 2, 5, 2, key="ns")
    uploaded = {}
    for i in range(num):
        with st.expander(f"Station {i+1}", expanded=(i<2)):
            ca, cb = st.columns(2)
            with ca: pf = st.file_uploader(f"PM ({i+1})", type="csv", key=f"mp{i}")
            with cb: sf = st.file_uploader(f"SCC ({i+1})", type="csv", key=f"ms{i}")
            if pf and sf: uploaded[i] = (pf, sf)
    if len(uploaded) >= 2:
        stations, names = {}, []
        for i,(pf,sf) in uploaded.items():
            pt, s2 = pd.read_csv(pf), pd.read_csv(sf); m = merge_data(pt, s2)
            if "Station" in s2.columns and len(s2["Station"].dropna()) > 0: nm = s2["Station"].dropna().iloc[0]
            elif "location" in pt.columns and len(pt["location"].dropna()) > 0: nm = pt["location"].dropna().iloc[0]
            else: nm = f"Station {i+1}"
            stations[nm] = m; names.append(nm)
        st.success(f"✅ {', '.join(names)}")
        t1,t2,t3,t4,t5,t6 = st.tabs(["📊 Summary","📍 Locations","💡 Opportunities","👤 People","📅 Day","💾 Export"])
        with t1:
            with st.expander("📊 Total Lost"):
                fig, ax = plt.subplots(figsize=CHART)
                bars = ax.bar(names, [len(stations[n]) for n in names], color=STATION_COLORS[:len(names)])
                for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, str(int(b.get_height())), ha="center", fontsize=8)
                ax.set_ylabel("Lost",fontsize=8); ax.set_title("Total by Station",fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
            with st.expander("🥧 OTR & UTR"):
                sp = st.selectbox("Station:", names, key="mcp")
                st.pyplot(make_pie_otr_utr(stations[sp], len(stations[sp]), f"{sp} OTR & UTR"))
        with t2:
            sel = st.selectbox("Station:", names, key="mcl")
            render_locations_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc_{sel}_")
        with t3:
            sel = st.selectbox("Station:", names, key="mco")
            render_opportunities_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc_{sel}_")
        with t4:
            sel = st.selectbox("Station:", names, key="mca")
            render_people_tab(stations[sel], len(stations[sel]), get_date_range(stations[sel]), kp=f"mc_{sel}_")
        with t5:
            with st.expander("📅 Day of Week"):
                st.caption("Day = date parcel was marked as lost in Perfect Mile.")
                fig, ax = plt.subplots(figsize=CHART)
                for i,n in enumerate(names):
                    if "Day of Week" in stations[n].columns:
                        dd = stations[n]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        ax.plot(dd.index, dd.values, marker="o", label=n, color=STATION_COLORS[i], linewidth=2)
                ax.set_ylabel("Lost",fontsize=8); ax.set_title("By Day Marked",fontsize=9)
                ax.tick_params(labelsize=7); ax.legend(fontsize=7); plt.xticks(rotation=0); plt.tight_layout(); st.pyplot(fig)
        with t6:
            for n in names:
                exc = ["Prev Event DT","previous_event_datetime","bucket","sub_bucket","previous_reason","previous_reason_3","event_datetime","Marked Lost DT","Is EoD Scrub"]
                ec = [c for c in stations[n].columns if c not in exc]
                st.download_button(f"⬇️ {n}", stations[n][ec].to_csv(index=False), f"Lost_{n}.csv", "text/csv", key=f"dl{n}")
    elif len(uploaded) == 1: st.warning("Need ≥2 stations.")
    else: st.info("👆 Upload file pairs.")
