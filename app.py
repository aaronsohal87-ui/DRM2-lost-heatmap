import streamlit as st  # Streamlit web framework
import pandas as pd  # Data manipulation library
import matplotlib.pyplot as plt  # Charting library

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="DRM2 Lost Heatmap", page_icon="📦", layout="wide")  # Wide layout
st.title("📦 DRM2 Lost Parcel Heatmap")  # Page title
st.markdown("---")  # Horizontal divider

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]  # Colours for multi-station
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]  # Size tiers
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # Week order
SHIFT_ORDER = ["NS", "AM", "PM", "OTR"]  # Shift display order
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen", "OTR": "firebrick"}  # Shift colours
SHIFT_DEFINITIONS = {  # Time windows per shift
    "NS": "00:00 – 09:59 (Night Sort — stow)",  # Night shift
    "AM": "10:00 – 13:59 (Pick, stage, dispatch)",  # Morning shift
    "PM": "14:00 – 23:59 (Dispatch, RELO)",  # Afternoon/evening shift
    "OTR": "On The Road (DSP responsibility)"  # Out on road with driver
}
SHIFT_HOUR_MAP = {  # Hour → shift lookup (NS extended to 10AM)
    0: "NS", 1: "NS", 2: "NS", 3: "NS", 4: "NS",
    5: "NS", 6: "NS", 7: "NS", 8: "NS", 9: "NS",
    10: "AM", 11: "AM", 12: "AM", 13: "AM",
    14: "PM", 15: "PM", 16: "PM", 17: "PM", 18: "PM",
    19: "PM", 20: "PM", 21: "PM", 22: "PM", 23: "PM"
}
SUB_BUCKET_SHIFT_MAP = {  # Deterministic sub-bucket → shift assignments
    "Lost At Station - Inducted Not Stowed": "NS",  # Inducted during NS, never stowed
    "Lost At Station - Stowed Not Picked Up": "AM",  # Stowed but never picked (AM job)
    "Lost At Station - Debrief Receive(RTS)": "PM",  # Returned at debrief (PM)
    "Lost On Road - Attempted": "OTR",  # Lost with driver
    "Lost On Road - Damage": "OTR",  # Damaged on road
    "Lost On Road - No Further Status": "OTR",  # No update from driver
}
SENSITIVE_COLS = [  # PII columns to auto-remove from SCC
    "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
    "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
    "Payment Method", "District", "Scheduled Delivery End Time"
]
REQUIRED_SCC_COLS = ["Tracking ID", "Sort Zone", "Aisle", "Cluster",  # Required SCC columns
    "Package Length", "Package Width", "Package Height", "DSP Name", "Assigned Cycle", "Last Updated Time"]
REQUIRED_PM_COLS = ["tracking_id", "sub_bucket"]  # Required PM columns
CHART = (7, 2.5)  # Standard chart size
DSP_MAX = 20  # Max chars for DSP labels
LABEL_MAX = 25  # Max chars for general labels
DETAIL_COLS = ["Tracking ID", "Cluster", "Aisle", "Sort Zone",  # Default detail table columns
               "DSP Name", "Size Category", "Shift", "Sub Bucket"]


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def get_size(val):
    """Classify parcel by longest side dimension."""
    if pd.isna(val): return "Unknown"  # No dimension data
    if val <= 35: return "Small"  # ≤35cm
    if val <= 45: return "Medium"  # 36-45cm
    if val <= 61: return "Small Oversize"  # 46-61cm
    return "Large Oversize"  # >61cm


def hour_to_shift(hour):
    """Convert hour (0-23) to responsible shift."""
    if pd.isna(hour): return "Unknown"  # No hour available
    return SHIFT_HOUR_MAP.get(int(hour), "Unknown")  # Lookup in map


def assign_shift(row):
    """Assign shift using sub-bucket (primary) + time (confirmatory).
    Priority: 1) Fixed sub-bucket map → 2) PM previous_event_datetime hour
    → 3) SCC Dispatch Time hour → 4) Assigned Cycle text → 5) Unknown
    """
    sb = row.get("Sub Bucket", "")  # Get sub-bucket from PM
    if sb in SUB_BUCKET_SHIFT_MAP: return SUB_BUCKET_SHIFT_MAP[sb]  # Fixed assignment
    prev_dt = row.get("Prev Event DT")  # Last scan time from PM
    if pd.notna(prev_dt): return hour_to_shift(prev_dt.hour)  # Use PM time
    disp_dt = row.get("Dispatch Time")  # Dispatch time from SCC
    if pd.notna(disp_dt): return hour_to_shift(disp_dt.hour)  # Use SCC time
    cyc = row.get("Assigned Cycle", "")  # Cycle text fallback
    if pd.notna(cyc):  # Parse cycle name
        u = str(cyc).upper()  # Uppercase for matching
        if "NS" in u or "NIGHT" in u: return "NS"  # Night sort
        if "PM" in u or "RELO" in u or "C2" in u: return "PM"  # PM/RELO
        if "AM" in u or "C1" in u: return "AM"  # AM/Cycle 1
    return "Unknown"  # Could not determine


def clean_scc(df):
    """Clean SCC export: remove PII, parse dimensions and dates."""
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])  # Strip PII
    for col in ["Package Length", "Package Width", "Package Height"]:  # Parse dimensions
        if col in df.columns:  # Only if column exists
            df[col] = df[col].astype(str).str.replace(" cm", "").str.replace("cm", "")  # Remove units
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numbers
    dims = ["Package Length", "Package Width", "Package Height"]  # Dimension columns
    if all(c in df.columns for c in dims):  # All dims available
        df["Longest Side"] = df[dims].max(axis=1)  # Get longest side
    else:
        df["Longest Side"] = float("nan")  # No data
    df["Size Category"] = df["Longest Side"].apply(get_size)  # Classify size
    if "Last Updated Time" in df.columns:  # Parse last updated (EoD scrub time)
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], dayfirst=True, errors="coerce")
    if "Dispatch Time" in df.columns:  # Parse dispatch time (actual operational day)
        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], dayfirst=True, errors="coerce")
        df["Day of Week"] = df["Dispatch Time"].dt.day_name()  # Use dispatch day, NOT last updated
    return df


def merge_data(pm_df, scc_df):
    """Merge PM (master) + SCC (detail) on Tracking ID. PM = source of truth."""
    scc_clean = clean_scc(scc_df.copy())  # Clean SCC data
    # Keep key PM columns including reasons for OTR/UTR analysis
    pm_keep = ["tracking_id", "bucket", "sub_bucket", "previous_event_datetime",
               "previous_reason", "previous_reason_3", "event_datetime"]  # PM columns to keep
    pm_cols = pm_df[[c for c in pm_keep if c in pm_df.columns]].copy()  # Only existing cols
    pm_cols = pm_cols.rename(columns={"tracking_id": "Tracking ID"})  # Standardise name
    # Parse previous_event_datetime (many PNOV rows are corrupt MM:SS.s format)
    pm_cols["Prev Event DT"] = pd.to_datetime(
        pm_cols.get("previous_event_datetime", pd.Series(dtype="object")),
        format="%d/%m/%Y %H:%M", errors="coerce")  # Parse valid ones only
    # Parse event_datetime from PM as fallback for day-of-week
    if "event_datetime" in pm_cols.columns:  # If PM has event timestamp
        pm_cols["PM Event DT"] = pd.to_datetime(
            pm_cols["event_datetime"], dayfirst=True, errors="coerce")  # Parse it
    merged = pm_cols.merge(scc_clean, on="Tracking ID", how="left")  # Left join (PM is master)
    merged["Sub Bucket"] = merged["sub_bucket"]  # Friendly column name
    merged["Bucket"] = merged.get("bucket")  # Top-level bucket
    # Create Loss Reason column (from PM previous_reason)
    if "previous_reason" in merged.columns:  # If reason available
        merged["Loss Reason"] = merged["previous_reason"].replace(  # Clean up uninformative values
            {"NOREASON": "No Reason", "NONE": "No Reason"}).fillna("Unknown")
    else:
        merged["Loss Reason"] = "Unknown"  # No reason data
    # Create UTR Reason column (from PM previous_reason_3 - deeper cause)
    if "previous_reason_3" in merged.columns:  # UTR-specific reason
        merged["UTR Reason"] = merged["previous_reason_3"].replace(
            {"NOREASON": "No Reason", "NONE": "No Reason"}).fillna("Unknown")
    else:
        merged["UTR Reason"] = "Unknown"  # No data
    merged["Shift"] = merged.apply(assign_shift, axis=1)  # Assign shift responsibility
    # Fill Day of Week from PM event_datetime where SCC Dispatch Time was missing
    if "Day of Week" not in merged.columns:  # If SCC didn't create it
        merged["Day of Week"] = pd.NaT  # Placeholder
    if "PM Event DT" in merged.columns:  # Fill gaps with PM date
        mask = merged["Day of Week"].isna() & merged["PM Event DT"].notna()  # Where missing
        merged.loc[mask, "Day of Week"] = merged.loc[mask, "PM Event DT"].dt.day_name()  # Fill
    # Ensure key columns exist even if SCC didn't have them
    for col in ["Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"]:
        if col not in merged.columns: merged[col] = None  # Add empty column
    return merged  # Return merged dataset


def get_date_range(df):
    """Get date range string from Dispatch Time (operational day)."""
    for col in ["Dispatch Time", "Last Updated Time"]:  # Preferred order
        if col in df.columns:  # If column exists
            valid = df[col].dropna()  # Non-null values
            if len(valid) > 0:  # Has data
                s = valid.min().strftime("%d %b %Y")  # Start date
                e = valid.max().strftime("%d %b %Y")  # End date
                return s if s == e else f"{s} – {e}"  # Format range
    return ""  # No date data


def safe_top(series):
    """Get most common value safely."""
    c = series.dropna().value_counts()  # Count values
    return c.index[0] if len(c) > 0 else "N/A"  # Top or N/A


def trunc(labels, max_len=LABEL_MAX):
    """Truncate long labels for charts."""
    return [str(l)[:max_len] + "..." if len(str(l)) > max_len else str(l) for l in labels]


def get_detail_cols(df, extra=None):
    """Get available detail columns for tables."""
    base = list(DETAIL_COLS)  # Start with defaults
    if extra: base = extra + [c for c in base if c not in extra]  # Prepend extras
    return [c for c in base if c in df.columns]  # Only existing columns


def verify_totals(df, total, label=""):
    """Check row count matches expected total."""
    if len(df) != total:  # Mismatch detected
        st.error(f"⚠️ MISMATCH {label}: Expected {total}, got {len(df)}.")  # Show error
        return False  # Failed
    return True  # Passed


def make_table(series, c1, c2):
    """Convert value_counts Series to display table."""
    t = series.reset_index()  # Index to column
    t.columns = [c1, c2]  # Rename
    t.index = range(1, len(t) + 1)  # 1-based index
    return t  # Return table


# ─── CHART FUNCTIONS ──────────────────────────────────────────────────────────

def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    """Horizontal bar chart with auto-scaled height."""
    h = max(2, len(data) * 0.3)  # Scale height (min 2in)
    fig, ax = plt.subplots(figsize=(figsize_width, h))  # Create figure
    labs = trunc(data.index, max_label)  # Truncate labels
    ax.barh(labs, data.values, color=color)  # Draw bars
    ax.invert_yaxis()  # Highest at top
    for i, v in enumerate(data.values):  # Add value labels
        ax.text(v + 0.2, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Lost Parcels", fontsize=8)  # X label
    ax.set_title(title, fontsize=9)  # Title
    ax.tick_params(labelsize=7)  # Tick size
    plt.tight_layout()  # Fit layout
    return fig  # Return figure


def make_bar_vert(data, xl, yl, title, color="steelblue", figsize=CHART):
    """Vertical bar chart."""
    fig, ax = plt.subplots(figsize=figsize)  # Create figure
    labs = trunc(data.index, LABEL_MAX)  # Truncate labels
    ax.bar(labs, data.values, color=color)  # Draw bars
    for i, v in enumerate(data.values):  # Add value labels
        ax.text(i, v + 0.2, str(int(v)), ha="center", fontsize=7)
    ax.set_xlabel(xl, fontsize=8)  # X axis label
    ax.set_ylabel(yl, fontsize=8)  # Y axis label
    ax.set_title(title, fontsize=9)  # Title
    ax.tick_params(labelsize=7)  # Tick size
    plt.xticks(rotation=0, ha="center")  # No rotation
    plt.tight_layout()  # Fit layout
    return fig  # Return figure


def make_bar_shift(data, title):
    """Shift bar chart with fixed order and colours."""
    data = data.reindex(SHIFT_ORDER, fill_value=0)  # Ensure all shifts shown
    fig, ax = plt.subplots(figsize=CHART)  # Create figure
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER],  # Draw colour-coded bars
                  color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    for b in bars:  # Add value labels on each bar
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                str(int(b.get_height())), ha="center", fontsize=7)
    ax.set_xlabel("Shift", fontsize=8)  # X label
    ax.set_ylabel("Lost Parcels", fontsize=8)  # Y label
    ax.set_title(title, fontsize=9)  # Title
    ax.tick_params(labelsize=7)  # Tick size
    plt.xticks(rotation=0)  # No rotation
    plt.tight_layout()  # Fit
    return fig  # Return figure


def make_pie_otr_utr(df, total, title):
    """Pie chart: OTR vs UTR vs Other."""
    otr_n = len(df[df["Sub Bucket"].str.contains("Lost On Road", na=False)])  # Count OTR
    utr_n = len(df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"])  # Count UTR
    other_n = total - otr_n - utr_n  # Everything else
    labels, sizes, colors, explode = [], [], [], []  # Build pie data
    if otr_n > 0:  # OTR slice
        labels.append(f"OTR ({otr_n})"); sizes.append(otr_n)
        colors.append("firebrick"); explode.append(0.05)
    if utr_n > 0:  # UTR slice
        labels.append(f"UTR ({utr_n})"); sizes.append(utr_n)
        colors.append("darkorange"); explode.append(0.05)
    if other_n > 0:  # Other slice
        labels.append(f"Other ({other_n})"); sizes.append(other_n)
        colors.append("steelblue"); explode.append(0)
    fig, ax = plt.subplots(figsize=(5, 4))  # Create figure
    ax.pie(sizes, labels=labels, colors=colors, explode=explode,  # Draw pie
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title(title, fontsize=9)  # Title
    plt.tight_layout()  # Fit
    return fig  # Return figure


# ─── LOST LOCATIONS TAB ──────────────────────────────────────────────────────

def render_locations_tab(df, total, dr, key_prefix=""):
    """Render Lost Locations: All=clusters/aisles, OTR=DSP+reason, UTR=reasons."""
    st.info("💡 Find where parcels are lost. OTR shows DSP + loss reason. UTR shows reasons.")
    verify_totals(df, total, "Locations")  # Count check
    filter_options = ["All Parcels", "OTR Only (Lost On Road)", "UTR Reprocess Only"]  # Options
    loc_filter = st.radio("Show:", filter_options, horizontal=True, key=f"{key_prefix}lf")  # Radio

    if loc_filter == "OTR Only (Lost On Road)":  # ── OTR VIEW ──
        view_df = df[df["Sub Bucket"].str.contains("Lost On Road", na=False)].copy()  # Filter OTR
        st.write(f"**Showing: {len(view_df)} OTR parcels**")  # Count header
        if len(view_df) == 0: st.warning("No OTR parcels."); return  # Guard
        with st.expander("🚚 OTR by DSP"):  # DSP breakdown
            dsp_data = view_df["DSP Name"].dropna().value_counts()  # Count per DSP
            if len(dsp_data) > 0:  # Has data
                view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"{key_prefix}otr_dsp_v")
                if view_mode == "Chart":  # Chart view
                    st.pyplot(make_bar_horiz(dsp_data, f"OTR by DSP ({dr})", color="firebrick", max_label=DSP_MAX))
                else:  # Table view
                    st.dataframe(make_table(dsp_data, "DSP", "Lost Parcels"), use_container_width=True)
        with st.expander("❓ OTR Loss Reasons"):  # Reason breakdown
            reason_data = view_df["Loss Reason"].dropna().value_counts()  # Count per reason
            if len(reason_data) > 0:  # Has data
                view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"{key_prefix}otr_r_v")
                if view_mode == "Chart":  # Chart view
                    st.pyplot(make_bar_horiz(reason_data, f"OTR Loss Reasons ({dr})", color="crimson"))
                else:  # Table view
                    st.dataframe(make_table(reason_data, "Reason", "Count"), use_container_width=True)
        with st.expander("📦 All OTR parcels"):  # Full detail
            show_cols = [c for c in ["Tracking ID", "Sub Bucket", "DSP Name", "Loss Reason",
                                     "Cluster", "Aisle"] if c in view_df.columns]  # Available cols
            out = view_df[show_cols].sort_values("DSP Name").reset_index(drop=True)  # Sort by DSP
            out.index = range(1, len(out) + 1)  # 1-based
            st.dataframe(out, use_container_width=True)  # Display

    elif loc_filter == "UTR Reprocess Only":  # ── UTR VIEW ──
        view_df = df[df["Sub Bucket"] == "Lost At Station - UTR Reprocess"].copy()  # Filter UTR
        st.write(f"**Showing: {len(view_df)} UTR parcels**")  # Count header
        if len(view_df) == 0: st.warning("No UTR parcels."); return  # Guard
        with st.expander("❓ UTR Loss Reasons"):  # Reason breakdown
            reason_data = view_df["UTR Reason"].dropna().value_counts()  # Count reasons
            if len(reason_data) > 0:  # Has data
                view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"{key_prefix}utr_r_v")
                if view_mode == "Chart":  # Chart view
                    st.pyplot(make_bar_horiz(reason_data, f"UTR Reasons ({dr})", color="darkorange"))
                else:  # Table view
                    st.dataframe(make_table(reason_data, "Reason", "Count"), use_container_width=True)
            else:  # No reasons recorded
                st.info("No specific reasons recorded.")
        with st.expander("📍 UTR by Location"):  # Location breakdown
            cl_data = view_df["Cluster"].dropna().value_counts()  # Clusters
            if len(cl_data) > 0:  # Has data
                view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"{key_prefix}utr_l_v")
                if view_mode == "Chart":
                    st.pyplot(make_bar_horiz(cl_data, f"UTR by Cluster ({dr})", color="darkorange"))
                else:
                    st.dataframe(make_table(cl_data, "Cluster", "Count"), use_container_width=True)
        with st.expander("📦 All UTR parcels"):  # Full detail
            show_cols = [c for c in ["Tracking ID", "Cluster", "Aisle", "DSP Name",
                                     "UTR Reason", "Shift"] if c in view_df.columns]  # Cols
            out = view_df[show_cols].sort_values("Cluster").reset_index(drop=True)  # Sort
            out.index = range(1, len(out) + 1)  # 1-based
            st.dataframe(out, use_container_width=True)  # Display

    else:  # ── ALL PARCELS VIEW ──
        view_df = df.copy()  # No filter
        st.write(f"**Showing: {len(view_df)} parcels (all)**")  # Count
        with st.expander("🏆 Top 10 Worst Locations"):  # Top locations
            rank_by = st.selectbox("Rank by:", ["Cluster", "Aisle", "Sort Zone"], key=f"{key_prefix}rb")
            rank_data = view_df[rank_by].dropna().value_counts().head(10)  # Top 10
            if len(rank_data) > 0:  # Has data
                view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"{key_prefix}loc_v")
                if view_mode == "Chart":  # Chart view
                    st.pyplot(make_bar_horiz(rank_data, f"Top 10 {rank_by}s ({dr})", color="darkred"))
                else:  # Table view
                    st.dataframe(make_table(rank_data, rank_by, "Lost Parcels"), use_container_width=True)
        with st.expander("🔍 Cluster Drill-Down"):  # Drill into a cluster
            clusters = sorted(view_df["Cluster"].dropna().unique())  # Available clusters
            if clusters:  # Has clusters
                sel = st.selectbox("Cluster:", clusters, key=f"{key_prefix}cl")  # Selector
                filt = view_df[view_df["Cluster"] == sel]  # Filter to cluster
                st.write(f"**{len(filt)} parcels** in Cluster {sel}")  # Count
                drill_data = filt["Aisle"].dropna().value_counts()  # Aisle breakdown
                if len(drill_data) > 0:  # Has aisles
                    view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"{key_prefix}cl_v")
                    if view_mode == "Chart":
                        st.pyplot(make_bar_horiz(drill_data, f"Cluster {sel} — Aisles", color="steelblue"))
                    else:
                        st.dataframe(make_table(drill_data, "Aisle", "Lost Parcels"), use_container_width=True)
                show_cols = get_detail_cols(filt, extra=["Tracking ID", "Aisle", "Sort Zone"])
                detail = filt[show_cols].sort_values("DSP Name").reset_index(drop=True)  # Sort
                detail.index = range(1, len(detail) + 1)  # 1-based
                st.dataframe(detail, use_container_width=True)  # Display


# ─── SUGGESTED OPPORTUNITIES TAB ─────────────────────────────────────────────

def render_opportunities_tab(df, total, dr, key_prefix=""):
    """Render Suggested Opportunities: shift leaderboard, pie, drill-down."""
    st.info("💡 Each parcel assigned to the responsible shift via sub-bucket + last scan time.")
    with st.expander("📖 How shifts are assigned"):  # Assignment rules
        st.markdown("""
| Sub Bucket | Shift | Reasoning |
|---|---|---|
| Inducted Not Stowed | **NS** | Inducted during Night Sort but never stowed |
| Stowed Not Picked Up | **AM** | Stowed but never picked for dispatch |
| Debrief Receive(RTS) | **PM** | Driver returned at debrief |
| Lost On Road - * | **OTR** | Lost while with DSP driver |
| PNOV / UTR / Other | **Time-based** | Last scan hour: 0–9→NS, 10–13→AM, 14–23→PM |
""")
    cols = st.columns(4)  # 4 columns for 4 shifts
    for i, (s, d) in enumerate(SHIFT_DEFINITIONS.items()):  # Display each
        cols[i].markdown(f"**{s}:** {d}")
    st.markdown("---")  # Divider

    shift_counts = df[df["Shift"] != "Unknown"]["Shift"].value_counts()  # Count per shift
    unk_count = len(df[df["Shift"] == "Unknown"])  # Unknown count

    with st.expander("🏆 Shift Responsibility Leaderboard"):  # Leaderboard
        rows = []  # Build table rows
        for s in SHIFT_ORDER:  # Each shift
            n = int(shift_counts.get(s, 0))  # Count (0 if none)
            pct = round(n / total * 100, 1) if total > 0 else 0  # Percentage
            rows.append({"Shift": s, "Lost": n, "% Total": f"{pct}%", "Window": SHIFT_DEFINITIONS[s]})
        rows.sort(key=lambda r: r["Lost"], reverse=True)  # Worst first
        lb = pd.DataFrame(rows)  # Create DataFrame
        lb.index = range(1, len(lb) + 1)  # 1-based index
        st.dataframe(lb, use_container_width=True, height=200)  # Display table
        if len(shift_counts) > 0:  # Has shift data
            st.pyplot(make_bar_shift(shift_counts, f"Lost by Shift ({dr})"))  # Chart

    with st.expander("🥧 OTR & UTR Breakdown"):  # Pie chart
        st.pyplot(make_pie_otr_utr(df, total, f"OTR & UTR vs Other ({dr})"))  # Show pie

    with st.expander("🔍 Shift Drill-Down"):  # Select a shift to inspect
        st.caption("Select a shift to see its sub-bucket breakdown and all parcels.")
        shift_options = [f"{s} — {int(shift_counts.get(s, 0))} parcels" for s in SHIFT_ORDER]
        if unk_count > 0: shift_options.append(f"Unknown — {unk_count} parcels")  # Add unknown
        selected = st.selectbox("Select Shift:", shift_options, key=f"{key_prefix}opp_sel")  # Selector
        selected_shift = selected.split(" — ")[0]  # Extract shift name
        s_df = df[df["Shift"] == selected_shift]  # Filter to shift
        count = len(s_df)  # Count
        if count > 0:  # Has parcels
            pct = round(count / total * 100, 1)  # Percentage
            st.markdown(f"**{selected_shift}** — **{count} parcels** ({pct}%)")  # Header
            sb_counts = s_df["Sub Bucket"].value_counts()  # Sub-bucket breakdown
            sb_tbl = sb_counts.reset_index()  # To table
            sb_tbl.columns = ["Sub Bucket", "Count"]  # Rename
            sb_tbl["% of Shift"] = (sb_tbl["Count"] / count * 100).round(1).astype(str) + "%"  # Pct
            sb_tbl["% of Total"] = (sb_tbl["Count"] / total * 100).round(1).astype(str) + "%"  # Total pct
            sb_tbl.index = range(1, len(sb_tbl) + 1)  # 1-based
            view_mode = st.radio("Display:", ["Table", "Chart"], horizontal=True, key=f"{key_prefix}sd_v")
            if view_mode == "Chart":  # Chart view
                st.pyplot(make_bar_horiz(sb_counts, f"{selected_shift} — Sub Buckets",
                          color=SHIFT_COLORS.get(selected_shift, "steelblue")))
            else:  # Table view
                st.dataframe(sb_tbl, use_container_width=True)  # Show table
            show_cols = [c for c in ["Tracking ID", "Sub Bucket", "Cluster", "Aisle",
                                     "DSP Name", "Size Category", "Loss Reason"] if c in df.columns]
            detail = s_df[show_cols].sort_values("Sub Bucket").reset_index(drop=True)  # Sort
            detail.index = range(1, len(detail) + 1)  # 1-based
            st.dataframe(detail, use_container_width=True)  # Display
        else:  # No parcels
            st.success(f"✅ No parcels for {selected_shift}.")

    st.markdown("---")  # Divider
    assigned = len(df[df["Shift"] != "Unknown"])  # Count assigned
    st.caption(f"✅ Verification: {assigned} assigned + {unk_count} unknown = "
               f"{assigned + unk_count} (Total: {total})")  # Verification line


# ─── MAIN APP ─────────────────────────────────────────────────────────────────

mode = st.radio("Mode:", ["Single Station", "Multi-Station Compare"], horizontal=True, key="mode")

with st.expander("📖 How to get your data"):  # Instructions
    st.markdown("""
**Step 1 — PerfectMile:** Concessions Control Tower → L&U → Lost → Export CSV
**Step 2 — SCC:** Paste Tracking IDs → View Options → Select All → Export CSV
**Step 3 — Upload both below.** Perfect Mile = source of truth for total count.
    """)

# ─── SINGLE STATION MODE ─────────────────────────────────────────────────────

if mode == "Single Station":
    st.subheader("Upload Data")  # Section header
    col_pm, col_scc = st.columns(2)  # Two upload columns
    with col_pm: pm_file = st.file_uploader("📊 Perfect Mile (.csv)", type="csv", key="pm_up")
    with col_scc: scc_file = st.file_uploader("📋 SCC (.csv)", type="csv", key="scc_up")

    if pm_file and scc_file:  # Both uploaded
        pm_df, scc_df = pd.read_csv(pm_file), pd.read_csv(scc_file)  # Load both
        pm_miss = [c for c in REQUIRED_PM_COLS if c not in pm_df.columns]  # Missing cols
        if pm_miss: st.error(f"❌ PM missing: {pm_miss}"); st.stop()  # Halt
        scc_miss = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]  # Missing cols
        if scc_miss: st.error(f"❌ SCC missing: {scc_miss}"); st.stop()  # Halt
        found = [c for c in SENSITIVE_COLS if c in scc_df.columns]  # PII found
        if found: st.warning(f"🔒 PII removed: {', '.join(found)}")  # Warn user
        df = merge_data(pm_df, scc_df)  # PM + SCC merge
        total = len(df)  # Total parcels (from PM)
        if total == 0: st.warning("No data."); st.stop()  # Guard
        pm_total = len(pm_df)  # PM row count
        matched = df["Cluster"].notna().sum()  # Matched with SCC
        st.success(f"✅ **{total} lost parcels** (PM:{pm_total}, SCC:{len(scc_df)}, Matched:{matched})")
        if total - matched > 0:  # Some unmatched
            st.info(f"ℹ️ {total - matched} in PM not in SCC — included with limited detail.")
        if total != pm_total:  # Total mismatch
            st.error(f"🚨 MISMATCH: {total} vs PM {pm_total}")

        dr = get_date_range(df)  # Date range string
        st.subheader(f"Quick Summary ({dr})")  # Summary header
        c1, c2, c3, c4, c5 = st.columns(5)  # 5 metric cards
        c1.metric("Total Lost", total)  # Total
        c2.metric("Worst Cluster", safe_top(df["Cluster"]))  # Worst cluster
        c3.metric("Worst Aisle", safe_top(df["Aisle"]))  # Worst aisle
        c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])  # Worst DSP
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]  # Valid shifts only
        c5.metric("Worst Shift", safe_top(sk) if len(sk) > 0 else "N/A")  # Worst shift

        t1, t2, t3, t4, t5, t6, t7 = st.tabs(["📊 Summary", "📍 Lost Locations", "🚚 DSP",
            "💡 Suggested Opportunities", "📅 Day of Week", "💾 Export", "📋 Bridge"])

        with t1:  # ═══ SUMMARY ═══
            verify_totals(df, total, "Summary")  # Count check
            with st.expander("📏 Size Breakdown"):  # Size section
                sc = df["Size Category"].value_counts()  # Count per size
                if len(sc) > 0:  # Has data
                    view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="sz_v")
                    if view_mode == "Chart":
                        st.pyplot(make_bar_vert(sc, "Size", "Lost", f"By Size ({dr})",
                            color=["green", "orange", "red", "darkred", "grey"][:len(sc)]))
                    else:
                        st.dataframe(make_table(sc, "Size", "Count"), use_container_width=True)
            with st.expander("📍 Cluster Breakdown"):  # Cluster section
                cc = df["Cluster"].dropna().value_counts()  # Count per cluster
                if len(cc) > 0:  # Has data
                    view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="cl_v")
                    if view_mode == "Chart":
                        st.pyplot(make_bar_horiz(cc, f"By Cluster ({dr})"))
                    else:
                        st.dataframe(make_table(cc, "Cluster", "Count"), use_container_width=True)
            with st.expander("🏷️ Sub Bucket Breakdown"):  # Sub bucket section
                sb = df["Sub Bucket"].value_counts()  # Count per sub-bucket
                if len(sb) > 0:  # Has data
                    view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="sb_v")
                    if view_mode == "Chart":
                        st.pyplot(make_bar_horiz(sb, f"By Sub Bucket ({dr})", color="teal"))
                    else:
                        st.dataframe(make_table(sb, "Sub Bucket", "Count"), use_container_width=True)

        with t2: render_locations_tab(df, total, dr, key_prefix="s_")  # ═══ LOCATIONS ═══

        with t3:  # ═══ DSP ═══
            verify_totals(df, total, "DSP")  # Count check
            with st.expander("🚚 DSP Breakdown"):  # DSP chart/table
                dsp_data = df["DSP Name"].dropna().value_counts()  # Count per DSP
                if len(dsp_data) > 0:  # Has data
                    view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dsp_v")
                    if view_mode == "Chart":
                        st.pyplot(make_bar_horiz(dsp_data, f"By DSP ({dr})", color="orange", max_label=DSP_MAX))
                    else:
                        st.dataframe(make_table(dsp_data.sort_index(), "DSP", "Lost"), use_container_width=True)
            with st.expander("📦 All parcels by DSP"):  # Full parcel list
                show_cols = get_detail_cols(df, extra=["Tracking ID", "DSP Name"])  # Columns
                out = df[show_cols].sort_values("DSP Name").reset_index(drop=True)  # Sort
                out.index = range(1, len(out) + 1)  # 1-based
                st.dataframe(out, use_container_width=True)  # Display

        with t4: render_opportunities_tab(df, total, dr, key_prefix="s_")  # ═══ OPPORTUNITIES ═══

        with t5:  # ═══ DAY OF WEEK ═══
            verify_totals(df, total, "Day")  # Count check
            st.caption("📌 Uses **Dispatch Time** (actual operational day), not EoD scrub time.")
            if "Day of Week" in df.columns:  # Has day data
                day_data = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                with st.expander("📅 Day of Week"):  # Day section
                    view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="day_v")
                    if view_mode == "Chart":  # Line chart
                        fig, ax = plt.subplots(figsize=CHART)  # Create figure
                        ax.plot(day_data.index, day_data.values, marker="o", color="green",
                                linewidth=2, markersize=6)  # Line plot
                        for i, (d, v) in enumerate(day_data.items()):  # Annotate points
                            ax.annotate(str(int(v)), xy=(i, v), xytext=(0, 8),
                                        textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
                        ax.set_xlabel("Day", fontsize=8)  # X label
                        ax.set_ylabel("Lost Parcels", fontsize=8)  # Y label
                        ax.set_title(f"Lost by Day of Week ({dr})", fontsize=9)  # Title
                        ax.tick_params(labelsize=7)  # Ticks
                        plt.xticks(rotation=0)  # No rotation
                        plt.tight_layout()  # Fit
                        st.pyplot(fig)  # Display
                    else:  # Table view
                        st.dataframe(make_table(day_data, "Day", "Lost Parcels"), use_container_width=True)
                    has_day = df["Day of Week"].notna().sum()  # Count with day info
                    st.caption(f"ℹ️ {has_day}/{total} parcels have Dispatch Time data.")
            else:  # No day data
                st.warning("No Dispatch Time data available.")

        with t6:  # ═══ EXPORT ═══
            verify_totals(df, total, "Export")  # Count check
            exc = ["Prev Event DT", "previous_event_datetime", "bucket", "sub_bucket",
                   "previous_reason", "previous_reason_3", "event_datetime", "PM Event DT"]
            ec = [c for c in df.columns if c not in exc]  # Export columns
            st.download_button("⬇️ Download CSV", df[ec].to_csv(index=False), "Lost_Merged.csv", "text/csv")

        with t7:  # ═══ BRIDGE ═══
            verify_totals(df, total, "Bridge")  # Count check
            cl_c = df["Cluster"].dropna().value_counts()  # Cluster counts
            dsp_c = df["DSP Name"].dropna().value_counts()  # DSP counts
            sb_c = df["Sub Bucket"].value_counts()  # Sub-bucket counts
            sh_c = df[df["Shift"] != "Unknown"]["Shift"].value_counts()  # Shift counts
            sl = "\n".join([f"  {s}: {int(sh_c.get(s, 0))} ({round(int(sh_c.get(s, 0))/total*100, 1)}%)"
                           for s in SHIFT_ORDER])  # Shift lines
            sb_lines = "\n".join([f"  {sb}: {n} ({round(int(n)/total*100, 1)}%)"
                                 for sb, n in sb_c.head(6).items()])  # Sub-bucket lines
            cdet = ""  # Cluster detail
            for cn, cv in cl_c.head(3).items():  # Top 3 clusters
                ta = df[df["Cluster"] == cn]["Aisle"].dropna().value_counts().head(3)  # Top aisles
                al = ", ".join([f"{a}({n})" for a, n in ta.items()])  # Format
                cdet += f"  {cn}: {cv} ({round(int(cv)/total*100, 1)}%) — {al}\n"  # Add line
            bridge = f"""Lost Parcels Bridge — DRM2\n{dr}\nTOTAL: {total}\nSHIFTS:\n{sl}\nSUB BUCKETS:\n{sb_lines}\nLOCATIONS:\n{cdet}"""
            st.text_area("✏️ Bridge:", value=bridge, height=300, key="bridge")  # Editable

    elif pm_file: st.info("👆 Upload SCC.")  # Only PM uploaded
    elif scc_file: st.info("👆 Upload PM.")  # Only SCC uploaded
    else: st.info("👆 Upload both files above.")  # Nothing uploaded

# ─── MULTI-STATION COMPARE ───────────────────────────────────────────────────

else:
    st.subheader("Upload Station Data")  # Section header
    num = st.slider("Stations:", 2, 5, 2, key="num_st")  # Station count slider
    uploaded = {}  # Store file pairs
    for i in range(num):  # Upload expanders
        with st.expander(f"Station {i + 1}", expanded=(i < 2)):  # Expand first 2
            ca, cb = st.columns(2)  # Two columns
            with ca: pf = st.file_uploader(f"PM ({i + 1})", type="csv", key=f"mp_{i}")  # PM upload
            with cb: sf = st.file_uploader(f"SCC ({i + 1})", type="csv", key=f"ms_{i}")  # SCC upload
            if pf and sf: uploaded[i] = (pf, sf)  # Store pair

    if len(uploaded) >= 2:  # At least 2 stations
        stations, names = {}, []  # Station data and names
        for i, (pf, sf) in uploaded.items():  # Process each pair
            pt, st2 = pd.read_csv(pf), pd.read_csv(sf)  # Load files
            m = merge_data(pt, st2)  # Merge
            if "Station" in st2.columns and len(st2["Station"].dropna()) > 0:
                nm = st2["Station"].dropna().iloc[0]  # From SCC
            elif "location" in pt.columns and len(pt["location"].dropna()) > 0:
                nm = pt["location"].dropna().iloc[0]  # From PM
            else:
                nm = f"Station {i + 1}"  # Fallback
            stations[nm] = m; names.append(nm)  # Store
        st.success(f"✅ Loaded: **{', '.join(names)}**")  # Confirm

        t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Summary", "📍 Locations", "🚚 DSP",
                                            "💡 Opportunities", "📅 Day", "💾 Export"])
        with t1:  # Summary comparison
            with st.expander("📊 Total Lost Comparison"):
                fig, ax = plt.subplots(figsize=CHART)  # Create figure
                bars = ax.bar(names, [len(stations[n]) for n in names],
                              color=STATION_COLORS[:len(names)])  # Bar per station
                for b in bars:  # Value labels
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                            str(int(b.get_height())), ha="center", fontsize=8)
                ax.set_ylabel("Lost", fontsize=8); ax.set_title("Total by Station", fontsize=9)
                ax.tick_params(labelsize=7); plt.tight_layout()
                st.pyplot(fig)  # Display

        with t2:  # Locations per station
            sel = st.selectbox("Station:", names, key="mc_loc")  # Station selector
            render_locations_tab(stations[sel], len(stations[sel]),
                                get_date_range(stations[sel]), key_prefix=f"mc_{sel}_")

        with t3:  # DSP per station
            for i, n in enumerate(names):  # Each station
                with st.expander(f"🚚 {n} — DSPs"):
                    dsp = stations[n]["DSP Name"].dropna().value_counts().head(10)  # Top DSPs
                    if len(dsp) > 0:
                        view_mode = st.radio("Display:", ["Chart", "Table"], horizontal=True, key=f"mc_dsp_{n}")
                        if view_mode == "Chart":
                            st.pyplot(make_bar_horiz(dsp, f"{n} DSPs", color=STATION_COLORS[i], max_label=DSP_MAX))
                        else:
                            st.dataframe(make_table(dsp, "DSP", "Lost"), use_container_width=True)

        with t4:  # Opportunities per station
            sel = st.selectbox("Station:", names, key="mc_opp")  # Station selector
            render_opportunities_tab(stations[sel], len(stations[sel]),
                                    get_date_range(stations[sel]), key_prefix=f"mc_{sel}_")

        with t5:  # Day of Week comparison
            with st.expander("📅 Day of Week Comparison"):
                fig, ax = plt.subplots(figsize=CHART)  # Create figure
                for i, n in enumerate(names):  # Each station
                    if "Day of Week" in stations[n].columns:  # Has day data
                        dd = stations[n]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                        ax.plot(dd.index, dd.values, marker="o", label=n,
                                color=STATION_COLORS[i], linewidth=2)  # Line per station
                ax.set_ylabel("Lost", fontsize=8); ax.set_title("Lost by Day of Week", fontsize=9)
                ax.tick_params(labelsize=7); ax.legend(fontsize=7)
                plt.xticks(rotation=0); plt.tight_layout()
                st.pyplot(fig)  # Display

        with t6:  # Export
            for n in names:  # Each station download
                exc = ["Prev Event DT", "previous_event_datetime", "bucket", "sub_bucket",
                       "previous_reason", "previous_reason_3", "event_datetime", "PM Event DT"]
                ec = [c for c in stations[n].columns if c not in exc]  # Export cols
                st.download_button(f"⬇️ {n}", stations[n][ec].to_csv(index=False),
                                   f"Lost_{n}.csv", "text/csv", key=f"dl_{n}")  # Button

    elif len(uploaded) == 1: st.warning("Need ≥2 stations.")  # Only 1 pair
    else: st.info("👆 Upload file pairs above.")  # Nothing uploaded
