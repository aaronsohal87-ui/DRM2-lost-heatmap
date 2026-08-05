import streamlit as st  # Streamlit UI framework
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Charting

# ─────────────────────────────────────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DRM2 Lost Heatmap", page_icon="📦", layout="wide")  # Wide layout for charts
st.title("📦 DRM2 Lost Parcel Heatmap")  # Main title
st.markdown("---")  # Divider

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]  # Multi-station chart colours
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]  # Parcel size tiers
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  # Week order
SHIFT_ORDER = ["NS", "AM", "PM", "OTR"]  # Display order for shift leaderboard
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen", "OTR": "firebrick"}  # Bar colours per shift
SHIFT_DEFINITIONS = {  # Human-readable time windows per shift
    "NS": "00:00 – 09:59 (Night Sort — stow)",
    "AM": "10:00 – 13:59 (Pick, stage, dispatch)",
    "PM": "14:00 – 23:59 (Dispatch, RELO)",
    "OTR": "On The Road (DSP responsibility)"
}
# Map each hour of the day to the responsible shift (NS extended to 10AM)
SHIFT_HOUR_MAP = {
    0: "NS", 1: "NS", 2: "NS", 3: "NS", 4: "NS",
    5: "NS", 6: "NS", 7: "NS", 8: "NS", 9: "NS",
    10: "AM", 11: "AM", 12: "AM", 13: "AM",
    14: "PM", 15: "PM", 16: "PM", 17: "PM", 18: "PM",
    19: "PM", 20: "PM", 21: "PM", 22: "PM", 23: "PM"
}
# Sub-buckets with deterministic shift assignment (no time needed)
SUB_BUCKET_SHIFT_MAP = {
    "Lost At Station - Inducted Not Stowed": "NS",       # Inducted during NS but never stowed
    "Lost At Station - Stowed Not Picked Up": "AM",      # Stowed but never picked (AM responsibility)
    "Lost At Station - Debrief Receive(RTS)": "PM",      # Driver returned at debrief (PM)
    "Lost On Road - Attempted": "OTR",                   # Lost with driver
    "Lost On Road - Damage": "OTR",                      # Lost with driver (damaged)
    "Lost On Road - No Further Status": "OTR",           # Lost with driver (no update)
}
# Columns containing PII that must be auto-removed from SCC exports
SENSITIVE_COLS = [
    "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
    "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
    "Payment Method", "District", "Scheduled Delivery End Time"
]
# Required columns for validation
REQUIRED_SCC_COLS = [
    "Tracking ID", "Sort Zone", "Aisle", "Cluster",
    "Package Length", "Package Width", "Package Height",
    "DSP Name", "Assigned Cycle", "Last Updated Time"
]
REQUIRED_PM_COLS = ["tracking_id", "sub_bucket"]  # Minimum PM columns
# Chart sizing constants
CHART = (7, 2.5)       # Standard chart dimensions
CHART_SM = (6, 2)      # Small chart dimensions
DSP_MAX = 20           # Max chars for DSP label truncation
LABEL_MAX = 25         # Max chars for general label truncation
# Default detail columns for parcel tables
DETAIL_COLS = ["Tracking ID", "Cluster", "Aisle", "Sort Zone",
               "DSP Name", "Size Category", "Shift", "Sub Bucket"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def get_size(val):
    """Classify parcel by longest side into size tier."""
    if pd.isna(val):
        return "Unknown"
    if val <= 35:
        return "Small"          # ≤35cm
    if val <= 45:
        return "Medium"         # 36–45cm
    if val <= 61:
        return "Small Oversize"  # 46–61cm
    return "Large Oversize"      # >61cm


def hour_to_shift(hour):
    """Convert hour (0–23) to shift using SHIFT_HOUR_MAP."""
    if pd.isna(hour):
        return "Unknown"  # No time data available
    return SHIFT_HOUR_MAP.get(int(hour), "Unknown")  # Lookup shift for this hour


def assign_shift(row):
    """Assign shift responsibility to a parcel row.
    
    Uses sub-bucket as primary determinant, confirmed/supplemented by time data.
    Priority:
    1. Fixed sub-bucket map (deterministic — sub-bucket alone determines shift)
    2. previous_event_datetime hour (last scan before EoD scrub from Perfect Mile)
    3. Dispatch Time hour (from SCC — when parcel left station)
    4. Assigned Cycle text (fallback when no timestamps available)
    5. 'Unknown' if nothing works
    """
    sb = row.get("Sub Bucket", "")  # Get sub-bucket classification from Perfect Mile
    # Step 1: Check if this sub-bucket has a fixed assignment
    if sb in SUB_BUCKET_SHIFT_MAP:
        return SUB_BUCKET_SHIFT_MAP[sb]  # Deterministic — no time needed
    # Step 2: Use previous_event_datetime from Perfect Mile (time of last scan before EoD)
    prev_dt = row.get("Prev Event DT")
    if pd.notna(prev_dt):
        return hour_to_shift(prev_dt.hour)  # Convert hour to shift
    # Step 3: Fall back to Dispatch Time from SCC
    disp_dt = row.get("Dispatch Time")
    if pd.notna(disp_dt):
        return hour_to_shift(disp_dt.hour)  # Convert hour to shift
    # Step 4: Fall back to Assigned Cycle text
    cyc = row.get("Assigned Cycle", "")
    if pd.notna(cyc):
        u = str(cyc).upper()  # Normalise for matching
        if "NS" in u or "NIGHT" in u:
            return "NS"
        if "PM" in u or "RELO" in u or "C2" in u:
            return "PM"
        if "AM" in u or "C1" in u:
            return "AM"
    # Step 5: No data available
    return "Unknown"


def clean_scc(df):
    """Clean SCC export: drop PII, parse dimensions, compute size category."""
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])  # Remove PII columns
    # Parse package dimensions (remove 'cm' suffix, convert to numeric)
    for col in ["Package Length", "Package Width", "Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm", "").str.replace("cm", "")  # Strip units
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float
    # Compute longest side for size classification
    dims = ["Package Length", "Package Width", "Package Height"]
    if all(c in df.columns for c in dims):
        df["Longest Side"] = df[dims].max(axis=1)  # Max of 3 dimensions
    else:
        df["Longest Side"] = float("nan")  # No dimension data
    df["Size Category"] = df["Longest Side"].apply(get_size)  # Classify into size tier
    # Parse datetime columns (UK format: dd/mm/yyyy)
    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], dayfirst=True, errors="coerce")
        df["Day of Week"] = df["Last Updated Time"].dt.day_name()  # Monday, Tuesday, etc.
    if "Dispatch Time" in df.columns:
        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], dayfirst=True, errors="coerce")
    return df


def merge_data(pm_df, scc_df):
    """Merge Perfect Mile (master) and SCC (detail) on Tracking ID.
    
    Perfect Mile is the source of truth for total count.
    SCC provides location, DSP, and dispatch details.
    Left join ensures all PM parcels are retained even if not in SCC.
    """
    scc_clean = clean_scc(scc_df.copy())  # Clean the SCC data
    # Extract relevant PM columns
    pm_cols = pm_df[["tracking_id", "bucket", "sub_bucket", "previous_event_datetime"]].copy()
    pm_cols = pm_cols.rename(columns={"tracking_id": "Tracking ID"})  # Standardise column name
    # Parse previous_event_datetime (format: dd/mm/yyyy HH:MM — many are corrupt MM:SS.s)
    pm_cols["Prev Event DT"] = pd.to_datetime(
        pm_cols["previous_event_datetime"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    # Left join: PM is master, SCC provides detail
    merged = pm_cols.merge(scc_clean, on="Tracking ID", how="left")
    # Create display columns
    merged["Sub Bucket"] = merged["sub_bucket"]  # Friendly name
    merged["Bucket"] = merged["bucket"]  # Top-level category
    # Assign responsible shift to each parcel
    merged["Shift"] = merged.apply(assign_shift, axis=1)
    # Ensure key columns exist even if SCC didn't have them
    for col in ["Cluster", "Aisle", "Sort Zone", "DSP Name", "Size Category"]:
        if col not in merged.columns:
            merged[col] = None
    return merged


def get_date_range(df):
    """Get human-readable date range string from Last Updated Time."""
    if "Last Updated Time" not in df.columns:
        return ""  # No date column
    valid = df["Last Updated Time"].dropna()  # Drop rows without dates
    if len(valid) == 0:
        return ""  # No valid dates
    s = valid.min().strftime("%d %b %Y")  # Start date
    e = valid.max().strftime("%d %b %Y")  # End date
    return s if s == e else f"{s} – {e}"  # Single day or range


def safe_top(series):
    """Get the most common value from a series (safe for empty series)."""
    c = series.dropna().value_counts()  # Count occurrences
    return c.index[0] if len(c) > 0 else "N/A"  # Return top or N/A


def trunc(labels, max_len=LABEL_MAX):
    """Truncate labels for chart display."""
    return [str(l)[:max_len] + "..." if len(str(l)) > max_len else str(l) for l in labels]


def get_detail_cols(df, extra=None):
    """Get available detail columns for display tables."""
    base = list(DETAIL_COLS)  # Start with default columns
    if extra:
        base = extra + [c for c in base if c not in extra]  # Prepend extras
    return [c for c in base if c in df.columns]  # Only return columns that exist


# ─────────────────────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    """Create horizontal bar chart with auto-scaled height."""
    h = max(2, len(data) * 0.3)  # Scale height to number of bars (min 2in)
    fig, ax = plt.subplots(figsize=(figsize_width, h))
    labs = trunc(data.index, max_label)  # Truncate long labels
    ax.barh(labs, data.values, color=color)  # Draw horizontal bars
    ax.invert_yaxis()  # Highest at top
    for i, v in enumerate(data.values):
        ax.text(v + 0.2, i, str(int(v)), va="center", fontsize=7)  # Value labels
    ax.set_xlabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def make_bar_vert(data, xl, yl, title, color="steelblue", figsize=CHART):
    """Create vertical bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    labs = trunc(data.index, LABEL_MAX)
    ax.bar(labs, data.values, color=color)  # Draw vertical bars
    for i, v in enumerate(data.values):
        ax.text(i, v + 0.2, str(int(v)), ha="center", fontsize=7)  # Value labels above bars
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def make_bar_shift(data, title):
    """Create bar chart specifically for shift data with fixed order and colours."""
    order = SHIFT_ORDER  # Always show all shifts in order
    data = data.reindex(order, fill_value=0)  # Fill missing shifts with 0
    fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(order, [data[s] for s in order],
                  color=[SHIFT_COLORS[s] for s in order])  # Colour-coded bars
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.2, str(int(h)),
                ha="center", fontsize=7)  # Value labels
    ax.set_xlabel("Shift", fontsize=8)
    ax.set_ylabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def make_table(series, c1, c2):
    """Convert a value_counts Series to a display DataFrame with 1-based index."""
    t = series.reset_index()  # Convert index to column
    t.columns = [c1, c2]  # Rename columns
    t.index = range(1, len(t) + 1)  # 1-based index for display
    return t


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────


def verify_totals(df, total, label=""):
    """Verify row count matches expected total — shows error if mismatch."""
    if len(df) != total:
        st.error(f"⚠️ COUNT MISMATCH {label}: Expected {total}, got {len(df)}. Check your data.")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SUGGESTED OPPORTUNITIES TAB (formerly Shift Rankings)
# ─────────────────────────────────────────────────────────────────────────────


def render_opportunities_tab(df, total, dr, key_prefix=""):
    """Render the Suggested Opportunities tab.
    
    Shows shift responsibility assignment based on sub-bucket + time,
    a leaderboard, a selectbox to drill into each shift's sub-bucket breakdown,
    and full parcel detail per shift.
    """
    # Explanation banner
    st.info(
        "💡 **What's here:** Each lost parcel is assigned to the shift most likely "
        "responsible based on its **sub-bucket** (from Perfect Mile) and the **time of "
        "last scan** before EoD scrub. Use the dropdown to explore each shift."
    )
    # Expandable explanation of assignment rules
    with st.expander("📖 How shifts are assigned"):
        st.markdown("""
| Sub Bucket | Assigned Shift | Reasoning |
|---|---|---|
| Inducted Not Stowed | **NS** | Parcel inducted during Night Sort but never stowed |
| Stowed Not Picked Up | **AM** | Stowed but never picked for dispatch |
| Debrief Receive(RTS) | **PM** | Driver returned parcel at debrief (PM) |
| Lost On Road - * | **OTR** | Lost while with DSP driver |
| PNOV / UTR Reprocess / Other | **Time-based** | Uses last scan time before EoD to assign shift |
""")
        st.caption("Time-based: hour 0–9 → NS, 10–13 → AM, 14–23 → PM. "
                   "Falls back to Dispatch Time or Assigned Cycle if previous_event_datetime is corrupt.")

    # Show shift time windows
    cols = st.columns(4)  # 4 columns for 4 shifts
    for i, (s, d) in enumerate(SHIFT_DEFINITIONS.items()):
        cols[i].markdown(f"**{s}:** {d}")  # Display each shift definition
    st.markdown("---")

    # ── Shift Responsibility Leaderboard ──
    st.subheader("🏆 Shift Responsibility Leaderboard")
    shift_counts = df[df["Shift"] != "Unknown"]["Shift"].value_counts()  # Count per shift (excl Unknown)
    # Build leaderboard table showing all shifts
    rows = []
    for s in SHIFT_ORDER:
        n = int(shift_counts.get(s, 0))  # Count for this shift (0 if none)
        pct = round(n / total * 100, 1) if total > 0 else 0  # Percentage of total
        rows.append({"Shift": s, "Lost Parcels": n, "% of Total": f"{pct}%",
                     "Time Window": SHIFT_DEFINITIONS[s]})
    rows.sort(key=lambda r: r["Lost Parcels"], reverse=True)  # Sort worst first
    lb = pd.DataFrame(rows)
    lb.index = range(1, len(lb) + 1)  # 1-based ranking
    st.dataframe(lb, use_container_width=True, height=200)  # Display table

    # Shift bar chart
    if len(shift_counts) > 0:
        st.pyplot(make_bar_shift(shift_counts, f"Lost by Responsible Shift ({dr})"))
    st.markdown("---")

    # ── Shift Drill-Down (selectbox) ──
    st.subheader("🔍 Shift Drill-Down")
    st.caption("Select a shift to see its sub-bucket breakdown and all parcels.")
    # Build options: shift name + count for clarity
    shift_options = []  # List of shift names to show in selectbox
    for s in SHIFT_ORDER:
        n = int(shift_counts.get(s, 0))
        shift_options.append(f"{s} — {n} parcels")  # e.g. "AM — 87 parcels"
    # Add Unknown if any exist
    unk_count = len(df[df["Shift"] == "Unknown"])
    if unk_count > 0:
        shift_options.append(f"Unknown — {unk_count} parcels")
    # Selectbox for shift selection
    selected = st.selectbox("Select Shift:", shift_options, key=f"{key_prefix}opp_shift_sel")
    selected_shift = selected.split(" — ")[0]  # Extract shift name from "NS — 41 parcels"
    # Filter data for selected shift
    s_df = df[df["Shift"] == selected_shift].copy()
    count = len(s_df)  # Number of parcels in this shift
    if count > 0:
        pct = round(count / total * 100, 1)  # Percentage of total
        st.markdown(f"**{selected_shift}** — **{count} parcels** ({pct}% of total) — "
                    f"*{SHIFT_DEFINITIONS.get(selected_shift, 'N/A')}*")
        # Sub-bucket breakdown table for this shift
        sb_counts = s_df["Sub Bucket"].value_counts()  # Count per sub-bucket
        sb_tbl = sb_counts.reset_index()
        sb_tbl.columns = ["Sub Bucket", "Count"]  # Rename columns
        sb_tbl["% of Shift"] = (sb_tbl["Count"] / count * 100).round(1).astype(str) + "%"  # Pct within shift
        sb_tbl["% of Total"] = (sb_tbl["Count"] / total * 100).round(1).astype(str) + "%"  # Pct of all lost
        sb_tbl.index = range(1, len(sb_tbl) + 1)  # 1-based index
        st.dataframe(sb_tbl, use_container_width=True)  # Display sub-bucket table
        # Full parcel detail for this shift
        with st.expander(f"📦 All {count} parcels for {selected_shift} shift"):
            show_cols = [c for c in ["Tracking ID", "Sub Bucket", "Cluster", "Aisle",
                                     "Sort Zone", "DSP Name", "Size Category"] if c in df.columns]
            detail = s_df[show_cols].sort_values("Sub Bucket").reset_index(drop=True)  # Sort by sub-bucket
            detail.index = range(1, len(detail) + 1)  # 1-based index
            st.dataframe(detail, use_container_width=True)
    else:
        st.success(f"✅ No parcels assigned to {selected_shift} shift.")  # Clean shift

    # ── Verification footer ──
    st.markdown("---")
    assigned = len(df[df["Shift"] != "Unknown"])  # Total assigned to a shift
    st.caption(f"✅ Verification: {assigned} assigned + {unk_count} unknown = "
               f"{assigned + unk_count} (Total: {total})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

# Mode selection: single station analysis or multi-station comparison
mode = st.radio("Mode:", ["Single Station", "Multi-Station Compare"], horizontal=True, key="mode")

# Instructions expander (always visible)
with st.expander("📖 How to get your data (click if you need help)"):
    st.markdown("""
**Step 1 — Export from PerfectMile:**
1. Open PerfectMile → Concessions Control Tower
2. Go to the **L&U** tab → **Lost** bucket → drill into sub-buckets
3. Export the table as CSV (this gives you `tracking_id`, `sub_bucket`, `previous_event_datetime`)

**Step 2 — Export from SCC:**
1. Open [SCC](https://logistics.amazon.co.uk/station/dashboard/outboundAMZL)
2. Paste all Tracking IDs from the PerfectMile export into SCC search
3. Click View Options → Select All
4. Click **Export → CSV**

**Step 3 — Upload both files below:**
- Perfect Mile CSV → provides total count, sub-bucket classification, and last-scan time
- SCC CSV → provides location (cluster/aisle), DSP, dispatch time, and parcel dimensions

**Important:** The Perfect Mile file is the source of truth for total lost count.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE STATION MODE
# ─────────────────────────────────────────────────────────────────────────────

if mode == "Single Station":
    st.subheader("Upload Data")
    col_pm, col_scc = st.columns(2)  # Two upload columns side by side
    with col_pm:
        pm_file = st.file_uploader("📊 Perfect Mile export (.csv)", type="csv", key="pm_upload")
    with col_scc:
        scc_file = st.file_uploader("📋 SCC export (.csv)", type="csv", key="scc_upload")

    if pm_file is not None and scc_file is not None:
        pm_df = pd.read_csv(pm_file)  # Load Perfect Mile data
        scc_df = pd.read_csv(scc_file)  # Load SCC data

        # ── Validate Perfect Mile ──
        pm_missing = [c for c in REQUIRED_PM_COLS if c not in pm_df.columns]
        if pm_missing:
            st.error(f"❌ Perfect Mile missing columns: {', '.join(pm_missing)}")
            st.stop()  # Halt execution

        # ── Validate SCC ──
        scc_missing = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
        if scc_missing:
            st.error(f"❌ SCC missing columns: {', '.join(scc_missing)}")
            st.stop()  # Halt execution

        # ── Sensitive columns warning ──
        found = [c for c in SENSITIVE_COLS if c in scc_df.columns]
        if found:
            st.warning(f"🔒 Sensitive columns auto-removed from SCC: {', '.join(found)}")

        # ── Merge data (PM = master) ──
        df = merge_data(pm_df, scc_df)
        total = len(df)  # This should equal len(pm_df)

        if total == 0:
            st.warning("⚠️ No data rows after merge.")
            st.stop()

        # ── Count verification banner ──
        pm_total = len(pm_df)  # Source of truth count
        scc_total = len(scc_df)  # SCC count (may differ)
        matched = df["Cluster"].notna().sum()  # Parcels that found a match in SCC
        unmatched = total - matched  # Parcels in PM but not in SCC

        st.success(f"✅ Loaded — **{total} lost parcels** "
                   f"(Perfect Mile: {pm_total}, SCC: {scc_total}, Matched: {matched})")
        if unmatched > 0:
            st.info(f"ℹ️ {unmatched} parcel(s) in Perfect Mile not found in SCC — "
                    "included with limited detail.")

        # ── Integrity check ──
        if total != pm_total:
            st.error(f"🚨 TOTAL MISMATCH: Merged has {total} but Perfect Mile has {pm_total}.")

        dr = get_date_range(df)  # Date range string for chart titles

        # ── Quick Summary metrics ──
        st.subheader(f"Quick Summary ({dr})")
        c1, c2, c3, c4, c5 = st.columns(5)  # 5 metric cards
        c1.metric("Total Lost", total)
        c2.metric("Worst Cluster", safe_top(df["Cluster"]))
        c3.metric("Worst Aisle", safe_top(df["Aisle"]))
        c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])  # Truncate long names
        sk = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"]  # Only valid shifts
        c5.metric("Worst Shift", safe_top(sk) if len(sk) > 0 else "N/A")

        # Cluster priority banner
        cl_r = df["Cluster"].dropna().value_counts()
        if len(cl_r) > 0:
            parts = []
            for i, (cl, n) in enumerate(cl_r.head(3).items()):
                pct = round(n / total * 100, 1)
                parts.append(f"#{i+1} Cluster {cl} — {n} parcels ({pct}%)")
            st.info("🎯 **Cluster Priority (highest losts first):** " + " → ".join(parts))

        # ── Tab layout ──
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📊 Summary", "📍 Lost Locations", "🚚 DSP",
            "💡 Suggested Opportunities", "📅 Day of Week", "💾 Export", "📋 Bridge"
        ])

        # ════════════════════════════════════════════════════════════════════
        # TAB 1: SUMMARY
        # ════════════════════════════════════════════════════════════════════
        with t1:
            st.info("💡 **What's here:** High-level breakdown by **size**, **cluster**, "
                    "**sub-bucket**, and reason.")
            verify_totals(df, total, "Summary Tab")  # Count check
            view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="sum_v")
            if view == "Chart":
                # Size breakdown chart
                sc = df["Size Category"].value_counts()
                if len(sc) > 0:
                    colors = ["green", "orange", "red", "darkred", "grey"][:len(sc)]
                    st.pyplot(make_bar_vert(sc, "Size", "Lost Parcels",
                                           f"Lost by Size ({dr})", color=colors))
                # Cluster breakdown chart
                cc = df["Cluster"].dropna().value_counts()
                if len(cc) > 0:
                    st.pyplot(make_bar_horiz(cc, f"Lost by Cluster ({dr})"))
                # Sub-bucket breakdown chart
                sb_c = df["Sub Bucket"].value_counts()
                if len(sb_c) > 0:
                    st.pyplot(make_bar_horiz(sb_c, f"Lost by Sub Bucket ({dr})", color="teal"))
            else:
                # Table views
                st.subheader("Size Breakdown")
                st.dataframe(make_table(df["Size Category"].value_counts(), "Size", "Lost Parcels"))
                st.subheader("Cluster × Size")
                pivot = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0)
                pivot["Total"] = pivot.sum(axis=1)  # Add total column
                st.dataframe(pivot)
                st.subheader("Sub Bucket Breakdown")
                st.dataframe(make_table(df["Sub Bucket"].value_counts(), "Sub Bucket", "Lost Parcels"),
                             use_container_width=True)

        # ════════════════════════════════════════════════════════════════════
        # TAB 2: LOST LOCATIONS
        # ════════════════════════════════════════════════════════════════════
        with t2:
            st.info("💡 **What's here:** Find exactly **where** parcels are being lost.")
            verify_totals(df, total, "Locations Tab")
            st.subheader("🏆 Top 10 Worst Locations")
            rank_by = st.selectbox("Rank by:", ["Cluster", "Aisle", "Sort Zone"], key="rb")
            rank_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="rv")
            rank_data = df[rank_by].dropna().value_counts().head(10)  # Top 10
            if len(rank_data) > 0:
                if rank_view == "Chart":
                    st.pyplot(make_bar_horiz(rank_data, f"Top 10 {rank_by}s ({dr})", color="darkred"))
                else:
                    st.dataframe(make_table(rank_data, rank_by, "Lost Parcels"))
            st.markdown("---")
            # Cluster drill-down
            st.subheader("🔍 Cluster Drill-Down")
            clusters = sorted(df["Cluster"].dropna().unique())  # All unique clusters
            if clusters:
                sel_cluster = st.selectbox("Select Cluster:", clusters, key="cl_sel")
                filtered = df[df["Cluster"] == sel_cluster]  # Filter to selected cluster
                st.write(f"**{len(filtered)} parcels** in Cluster {sel_cluster}")
                drill_by = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="drill")
                drill_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dv")
                drill_data = filtered[drill_by].dropna().value_counts()
                if len(drill_data) > 0:
                    if drill_view == "Chart":
                        st.pyplot(make_bar_horiz(drill_data,
                                                 f"Cluster {sel_cluster} — {drill_by}s", color="steelblue"))
                    else:
                        st.dataframe(make_table(drill_data, drill_by, "Lost Parcels"))
                # Expandable full parcel list for the cluster
                with st.expander(f"📦 All parcels in Cluster {sel_cluster}"):
                    show_cols = get_detail_cols(filtered, extra=["Tracking ID", "Aisle", "Sort Zone"])
                    detail = filtered[show_cols].sort_values("DSP Name").reset_index(drop=True)
                    detail.index = range(1, len(detail) + 1)
                    st.dataframe(detail, use_container_width=True)

        # ════════════════════════════════════════════════════════════════════
        # TAB 3: DSP
        # ════════════════════════════════════════════════════════════════════
        with t3:
            st.info("💡 **What's here:** See which DSPs are losing the most parcels.")
            verify_totals(df, total, "DSP Tab")
            dsp_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dsp_v")
            dsp_data = df["DSP Name"].dropna().value_counts()  # Worst-first
            # Assigned Cycle counts if available
            cycle_data = (df["Assigned Cycle"].dropna().value_counts()
                          if "Assigned Cycle" in df.columns else pd.Series(dtype="int64"))
            if dsp_view == "Chart":
                if len(dsp_data) > 0:
                    st.pyplot(make_bar_horiz(dsp_data, f"Lost by DSP ({dr})",
                                            color="orange", max_label=DSP_MAX))
                if len(cycle_data) > 0:
                    st.pyplot(make_bar_horiz(cycle_data, f"Lost by Cycle ({dr})", color="purple"))
            else:
                if len(dsp_data) > 0:
                    st.subheader("DSP (A–Z)")
                    dsp_alpha = dsp_data.sort_index()  # Alphabetical order
                    st.dataframe(make_table(dsp_alpha, "DSP", "Lost Parcels"), use_container_width=True)
                if len(cycle_data) > 0:
                    st.subheader("Cycle")
                    st.dataframe(make_table(cycle_data, "Cycle", "Lost Parcels"))
            # Expandable full list
            with st.expander("📦 All parcels grouped by DSP (alphabetical)"):
                show_cols = get_detail_cols(df, extra=["Tracking ID", "DSP Name"])
                all_by_dsp = df[show_cols].sort_values("DSP Name").reset_index(drop=True)
                all_by_dsp.index = range(1, len(all_by_dsp) + 1)
                st.dataframe(all_by_dsp, use_container_width=True)

        # ════════════════════════════════════════════════════════════════════
        # TAB 4: SUGGESTED OPPORTUNITIES
        # ════════════════════════════════════════════════════════════════════
        with t4:
            render_opportunities_tab(df, total, dr, key_prefix="single_")

        # ════════════════════════════════════════════════════════════════════
        # TAB 5: DAY OF WEEK
        # ════════════════════════════════════════════════════════════════════
        with t5:
            st.info("💡 **What's here:** See which **days of the week** have the most lost parcels.")
            verify_totals(df, total, "Day Tab")
            if "Day of Week" in df.columns:
                day_data = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
            else:
                day_data = pd.Series(dtype="int64")
            day_view = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="day_v")
            if len(day_data) > 0:
                if day_view == "Chart":
                    fig, ax = plt.subplots(figsize=CHART)
                    ax.plot(day_data.index, day_data.values, marker="o", color="green",
                            linewidth=2, markersize=6)  # Line chart
                    for i, (day, val) in enumerate(day_data.items()):
                        ax.annotate(str(int(val)), xy=(i, val), xytext=(0, 8),
                                    textcoords="offset points", ha="center", fontsize=8,
                                    fontweight="bold")  # Annotate each point
                    ax.set_xlabel("Day", fontsize=8)
                    ax.set_ylabel("Lost Parcels", fontsize=8)
                    ax.set_title(f"Lost by Day of Week ({dr})", fontsize=9)
                    ax.tick_params(labelsize=7)
                    plt.xticks(rotation=0, ha="center")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.dataframe(make_table(day_data, "Day", "Lost Parcels"))
                # Drill-down by day
                with st.expander("🔍 Parcel Details by Day"):
                    avail_days = [d for d in DAY_ORDER if d in df["Day of Week"].values]
                    if avail_days:
                        sel_day = st.selectbox("Select day:", avail_days, key="day_sel")
                        day_df = df[df["Day of Week"] == sel_day]
                        st.write(f"**{len(day_df)} parcels** on {sel_day}")
                        show_cols = get_detail_cols(day_df)
                        out = day_df[show_cols].sort_values("DSP Name").reset_index(drop=True)
                        out.index = range(1, len(out) + 1)
                        st.dataframe(out, use_container_width=True)
            else:
                st.warning("No day-of-week data available (Last Updated Time missing from SCC).")

        # ════════════════════════════════════════════════════════════════════
        # TAB 6: EXPORT
        # ════════════════════════════════════════════════════════════════════
        with t6:
            st.info("💡 **What's here:** Download the merged dataset as a CSV.")
            verify_totals(df, total, "Export Tab")
            # Remove internal/working columns from export
            export_cols = [c for c in df.columns
                          if c not in ["Prev Event DT", "previous_event_datetime", "bucket", "sub_bucket"]]
            st.download_button("⬇️ Download Merged CSV", df[export_cols].to_csv(index=False),
                               "Lost_Parcels_Merged.csv", "text/csv")

        # ════════════════════════════════════════════════════════════════════
        # TAB 7: BRIDGE
        # ════════════════════════════════════════════════════════════════════
        with t7:
            st.info("💡 **What's here:** Auto-generated bridge with sub-bucket root causes "
                    "and shift-based actions.")
            verify_totals(df, total, "Bridge Tab")
            # Compute key aggregates for bridge
            cl_c = df["Cluster"].dropna().value_counts()  # Cluster counts
            ai_c = df["Aisle"].dropna().value_counts()  # Aisle counts
            dsp_c = df["DSP Name"].dropna().value_counts()  # DSP counts
            sz_c = df["Size Category"].value_counts()  # Size counts
            sb_c = df["Sub Bucket"].value_counts()  # Sub-bucket counts
            sh_c = df[df["Shift"] != "Unknown"]["Shift"].value_counts()  # Shift counts

            # Worst offenders
            wc = cl_c.index[0] if len(cl_c) > 0 else "N/A"  # Worst cluster
            wc_n = int(cl_c.values[0]) if len(cl_c) > 0 else 0
            wc_p = round(wc_n / total * 100, 1) if total > 0 else 0
            wa = ai_c.index[0] if len(ai_c) > 0 else "N/A"  # Worst aisle
            wa_n = int(ai_c.values[0]) if len(ai_c) > 0 else 0
            wd = dsp_c.index[0] if len(dsp_c) > 0 else "N/A"  # Worst DSP
            wd_n = int(dsp_c.values[0]) if len(dsp_c) > 0 else 0
            avg_d = dsp_c.mean() if len(dsp_c) > 0 else 1  # Average DSP losts
            dm = round(wd_n / avg_d, 1) if avg_d > 0 else 1.0  # Multiplier vs average
            ws = sz_c.index[0] if len(sz_c) > 0 else "N/A"  # Worst size
            ws_n = int(sz_c.values[0]) if len(sz_c) > 0 else 0
            wsb = sb_c.index[0] if len(sb_c) > 0 else "N/A"  # Worst sub-bucket
            wsb_n = int(sb_c.values[0]) if len(sb_c) > 0 else 0
            wsh = sh_c.index[0] if len(sh_c) > 0 else "N/A"  # Worst shift
            wsh_n = int(sh_c.values[0]) if len(sh_c) > 0 else 0

            # Daily breakdown text
            if "Last Updated Time" in df.columns:
                df_temp = df.copy()
                df_temp["Date"] = df_temp["Last Updated Time"].dt.strftime("%d/%m")  # dd/mm format
                daily = df_temp.groupby("Date").size()  # Count per date
                dl = "\n".join([f"  {d}: {n} lost" for d, n in daily.items()])
            else:
                dl = "  (No date data)"

            # Shift breakdown text
            sl = "\n".join([f"  {s}: {int(sh_c.get(s, 0))} "
                            f"({round(int(sh_c.get(s, 0)) / total * 100, 1)}%)" for s in SHIFT_ORDER])
            # Sub-bucket breakdown text (top 6)
            sb_lines = "\n".join([f"  {sb}: {n} ({round(int(n) / total * 100, 1)}%)"
                                  for sb, n in sb_c.head(6).items()])
            # Cluster detail with top aisles
            cdet = ""
            for cn, cv in cl_c.head(3).items():
                ta = df[df["Cluster"] == cn]["Aisle"].dropna().value_counts().head(3)
                al = ", ".join([f"{a} ({n})" for a, n in ta.items()])
                cdet += f"  Cluster {cn}: {cv} ({round(int(cv) / total * 100, 1)}%) — {al}\n"
            # DSP detail (top 3)
            dlines = "\n".join([f"  {d}: {n} ({round(int(n) / total * 100, 1)}%)"
                                for d, n in dsp_c.head(3).items()])

            # ── Generate actions ──
            acts = []

            def ac(t):
                acts.append(f"AC{len(acts) + 1}: {t}")  # Auto-numbered action

            # Cluster concentration action
            if wc_p > 40:
                ac(f"Dedicated PS to Cluster {wc} — {wc_p}% of losts concentrated here.")
            else:
                ac(f"PS rotation between top clusters "
                   f"({', '.join([str(c) for c in cl_c.head(3).index])}).")
            # DSP outlier action
            if dm >= 2:
                ac(f"Stand-down meeting with DSP {wd} leadership — {dm}x the station average.")
            elif dm >= 1.5:
                ac(f"Process briefing for DSP {wd} — {dm}x the station average.")
            else:
                ac("Station-wide process refresher — losts spread across DSPs evenly.")
            # Sub-bucket specific actions
            ins = int(sb_c.get("Lost At Station - Inducted Not Stowed", 0))
            if ins > total * 0.15:
                ac(f"Night Sort stow audit — {ins} parcels inducted but never stowed "
                   f"({round(ins / total * 100, 1)}%).")
            pnov = int(sb_c.get("Lost At Station - PNOV", 0))
            if pnov > total * 0.3:
                ac(f"PNOV deep-dive — {pnov} parcels picked but not on vehicle "
                   f"({round(pnov / total * 100, 1)}%).")
            utr = int(sb_c.get("Lost At Station - UTR Reprocess", 0))
            if utr > total * 0.1:
                ac(f"UTR process review — {utr} parcels stuck in reprocess "
                   f"({round(utr / total * 100, 1)}%).")
            otr_total = sum(int(sb_c.get(k, 0)) for k in sb_c.index if "Lost On Road" in k)
            if otr_total > total * 0.05:
                ac(f"DSP driver briefing — {otr_total} parcels lost on road "
                   f"({round(otr_total / total * 100, 1)}%).")
            if len(sh_c) > 1 and wsh_n > total * 0.4:
                ac(f"5-whys session for {wsh} shift — {wsh_n} losts "
                   f"({round(wsh_n / total * 100, 1)}%).")

            # Build bridge text
            bridge = f"""Lost Parcels Bridge — DRM2
{dr}
TOTAL LOST: {total} (Source: Perfect Mile)
DAILY BREAKDOWN:
{dl}
SHIFT RESPONSIBILITY:
{sl}
RC1) SUB BUCKET:
{sb_lines}
RC2) LOCATION:
{cdet}
RC3) DSP:
{dlines}
RC4) SIZE: {ws} ({ws_n}, {round(ws_n / total * 100, 1)}%)
ACTIONS:
{chr(10).join(acts)}
"""
            st.text_area("✏️ Edit bridge below:", value=bridge, height=400, key="bridge_edit")
            # AI enhancement prompt
            st.subheader("🤖 Enhance with Quick")
            prompt = (
                f"Write a professional Lost Parcels bridge for DRM2 station. "
                f"Include shift responsibility analysis and sub-bucket root causes.\n\n"
                f"Data ({dr}): Total={total}, Worst Cluster={wc} ({wc_n}, {wc_p}%), "
                f"Worst Aisle={wa} ({wa_n}), Worst DSP={wd} ({wd_n}, {dm}x avg), "
                f"Worst Sub-Bucket={wsb} ({wsb_n}), Worst Shift={wsh} ({wsh_n})\n\n"
                f"Sub-Buckets:\n{sb_lines}\nShifts:\n{sl}\nClusters:\n{cdet}DSPs:\n{dlines}\n\n"
                f"Generate specific, actionable recommendations referencing exact shifts, "
                f"sub-buckets, clusters, aisles, and DSPs."
            )
            st.code(prompt, language="text")
            st.caption("📋 Click the copy icon → open Quick → paste → get AI-enhanced bridge")

    # ── Partial upload guidance ──
    elif pm_file is not None and scc_file is None:
        st.info("👆 Now upload your SCC export to complete the analysis.")
    elif pm_file is None and scc_file is not None:
        st.info("👆 Now upload your Perfect Mile export to complete the analysis.")
    else:
        st.info("👆 Upload both files above to get started.")

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-STATION COMPARE MODE
# ─────────────────────────────────────────────────────────────────────────────

else:
    st.subheader("Upload Station Data")
    st.caption("Upload Perfect Mile + SCC pairs per station (2–5 stations).")
    num = st.slider("How many stations?", 2, 5, 2, key="num_stations")  # Number of stations
    uploaded = {}  # Store uploaded file pairs
    for i in range(num):
        with st.expander(f"Station {i + 1}", expanded=(i < 2)):  # Expand first two by default
            col_a, col_b = st.columns(2)
            with col_a:
                pm_f = st.file_uploader(f"Perfect Mile (Station {i + 1})", type="csv",
                                        key=f"mc_pm_{i}")
            with col_b:
                scc_f = st.file_uploader(f"SCC (Station {i + 1})", type="csv",
                                         key=f"mc_scc_{i}")
            if pm_f and scc_f:
                uploaded[i] = (pm_f, scc_f)  # Store pair

    if len(uploaded) >= 2:
        stations = {}  # Dict of station_name → merged DataFrame
        names = []  # Ordered list of station names
        for i, (pm_f, scc_f) in uploaded.items():
            pm_tmp = pd.read_csv(pm_f)  # Load PM
            scc_tmp = pd.read_csv(scc_f)  # Load SCC
            merged_tmp = merge_data(pm_tmp, scc_tmp)  # Merge
            # Determine station name (SCC Station col > PM location col > generic)
            if "Station" in scc_tmp.columns and len(scc_tmp["Station"].dropna()) > 0:
                name = scc_tmp["Station"].dropna().iloc[0]
            elif "location" in pm_tmp.columns and len(pm_tmp["location"].dropna()) > 0:
                name = pm_tmp["location"].dropna().iloc[0]
            else:
                name = f"Station {i + 1}"
            stations[name] = merged_tmp
            names.append(name)

        st.success(f"✅ Loaded: **{', '.join(names)}**")  # Confirm load
        # ── Station overview metrics ──
        st.subheader("Station Overview")
        mc = st.columns(len(names))  # One column per station
        for i, n in enumerate(names):
            sk = stations[n][stations[n]["Shift"].isin(SHIFT_ORDER)]["Shift"]  # Valid shifts only
            mc[i].metric(n, f"{len(stations[n])} lost")  # Total metric
            mc[i].caption(f"Worst shift: {safe_top(sk) if len(sk) > 0 else 'N/A'}")

        # ── Tabs ──
        t1, t2, t3, t4, t5, t6 = st.tabs([
            "📊 Summary", "📍 Locations", "🚚 DSP",
            "💡 Opportunities", "📅 Day of Week", "💾 Export"
        ])

        # ── Tab 1: Summary comparison ──
        with t1:
            st.info("💡 Compare totals and sub-bucket breakdowns across stations.")
            # Total lost bar chart
            fig, ax = plt.subplots(figsize=CHART)
            bars = ax.bar(names, [len(stations[n]) for n in names],
                          color=STATION_COLORS[:len(names)])
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.3, str(int(h)),
                        ha="center", fontsize=8)
            ax.set_ylabel("Lost Parcels", fontsize=8)
            ax.set_title("Total Lost by Station", fontsize=9)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            # Sub-bucket comparison per station
            st.markdown("---")
            st.subheader("Sub-Bucket Comparison")
            for i, n in enumerate(names):
                sb = stations[n]["Sub Bucket"].value_counts()
                if len(sb) > 0:
                    st.pyplot(make_bar_horiz(sb, f"{n} — Sub Buckets",
                                            color=STATION_COLORS[i], figsize_width=6))

        # ── Tab 2: Locations comparison ──
        with t2:
            rank_by = st.selectbox("Rank by:", ["Cluster", "Aisle", "Sort Zone"], key="mc_rb")
            for i, n in enumerate(names):
                d = stations[n][rank_by].dropna().value_counts().head(10)
                if len(d) > 0:
                    st.pyplot(make_bar_horiz(d, f"{n} — Top 10 {rank_by}s",
                                            color=STATION_COLORS[i], figsize_width=6))

        # ── Tab 3: DSP comparison ──
        with t3:
            for i, n in enumerate(names):
                dsp = stations[n]["DSP Name"].dropna().value_counts().head(10)
                if len(dsp) > 0:
                    st.pyplot(make_bar_horiz(dsp, f"{n} — Worst DSPs",
                                            color=STATION_COLORS[i], max_label=DSP_MAX))

        # ── Tab 4: Suggested Opportunities (per station with selectbox) ──
        with t4:
            st.info("💡 Select a station, then drill into each shift's sub-bucket breakdown.")
            # Station selector
            sel_station = st.selectbox("Select Station:", names, key="mc_opp_station")
            stn_df = stations[sel_station]  # Get data for selected station
            stn_total = len(stn_df)  # Total for this station
            stn_dr = get_date_range(stn_df)  # Date range for this station
            # Render the opportunities tab for the selected station
            render_opportunities_tab(stn_df, stn_total, stn_dr,
                                     key_prefix=f"mc_{sel_station}_")

        # ── Tab 5: Day of Week comparison ──
        with t5:
            fig, ax = plt.subplots(figsize=CHART)
            for i, n in enumerate(names):
                if "Day of Week" in stations[n].columns:
                    dd = stations[n]["Day of Week"].dropna().value_counts().reindex(
                        DAY_ORDER, fill_value=0)
                    ax.plot(dd.index, dd.values, marker="o", label=n,
                            color=STATION_COLORS[i], linewidth=2)
            ax.set_ylabel("Lost Parcels", fontsize=8)
            ax.set_title("Lost by Day of Week", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        # ── Tab 6: Export ──
        with t6:
            for n in names:
                export_cols = [c for c in stations[n].columns
                              if c not in ["Prev Event DT", "previous_event_datetime",
                                           "bucket", "sub_bucket"]]
                st.download_button(f"⬇️ {n}", stations[n][export_cols].to_csv(index=False),
                                   f"Lost_{n}.csv", "text/csv", key=f"dl_{n}")
            # Combined export
            combined = pd.concat([stations[n].assign(Station_Name=n) for n in names],
                                 ignore_index=True)
            export_cols = [c for c in combined.columns
                          if c not in ["Prev Event DT", "previous_event_datetime",
                                       "bucket", "sub_bucket"]]
            st.download_button("⬇️ All Combined", combined[export_cols].to_csv(index=False),
                               "Lost_Combined.csv", "text/csv", key="dl_all")
    elif len(uploaded) == 1:
        st.warning("⚠️ Upload at least 2 station pairs to compare.")
    else:
        st.info("👆 Upload your file pairs above to get started.")
