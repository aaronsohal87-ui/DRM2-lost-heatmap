import streamlit as st  # Streamlit — builds the web app interface
import pandas as pd  # Pandas — reads/manipulates the CSV data as tables (dataframes)
import matplotlib.pyplot as plt  # Matplotlib — generates all the charts/graphs

# --- PAGE SETUP ---
st.set_page_config(page_title="SCC Lost Heatmap", page_icon="📦", layout="wide")
st.title("SCC Lost Parcel Heatmap")
st.markdown("---")

# --- CONSTANTS ---
STATION_COLORS = ["steelblue", "orange", "green", "red", "purple"]
SIZE_ORDER = ["Small", "Medium", "Small Oversize", "Large Oversize", "Unknown"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SHIFT_ORDER = ["NS", "AM", "PM"]
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen"}

# Shift definitions (displayed for transparency)
SHIFT_DEFINITIONS = {
    "NS": "00:00 – 08:59 (Night Sort — stow)",
    "AM": "09:00 – 13:59 (AM — pick, stage, dispatch)",
    "PM": "14:00 – 23:59 (PM — dispatch, RELO)"
}

# Hour → Shift mapping
SHIFT_HOUR_MAP = {
    0: "NS", 1: "NS", 2: "NS", 3: "NS", 4: "NS",
    5: "NS", 6: "NS", 7: "NS", 8: "NS",
    9: "AM", 10: "AM", 11: "AM", 12: "AM", 13: "AM",
    14: "PM", 15: "PM", 16: "PM", 17: "PM", 18: "PM",
    19: "PM", 20: "PM", 21: "PM", 22: "PM", 23: "PM"
}

SENSITIVE_COLS = [
    "Last Scan By", "Driver Id", "Holder Name", "City", "Postal",
    "Province", "Ordering Order ID", "Order Amount", "Receivable Amount",
    "Payment Method", "District", "Scheduled Delivery End Time"
]

REQUIRED_COLS = [
    "Tracking ID", "Sort Zone", "Aisle", "Cluster",
    "Package Length", "Package Width", "Package Height",
    "DSP Name", "Assigned Cycle", "Last Updated Time"
]

# Chart sizes (compact)
CHART = (7, 2.5)
CHART_SM = (6, 2)
DSP_MAX = 20


# --- HELPERS ---

def get_size(val):
    """Classify parcel by longest side (cm) into Amazon UK size tiers."""
    if pd.isna(val): return "Unknown"
    if val <= 35: return "Small"
    if val <= 45: return "Medium"
    if val <= 61: return "Small Oversize"
    return "Large Oversize"


def classify_shift(hour):
    """Map hour (0-23) to shift. Returns 'Unknown' if NaN."""
    if pd.isna(hour): return "Unknown"
    return SHIFT_HOUR_MAP.get(int(hour), "Unknown")


def clean_data(df):
    """Full cleaning: remove sensitive, fix dims, add size/day/shift."""
    df = df.drop(columns=[c for c in SENSITIVE_COLS if c in df.columns])

    for col in ["Package Length", "Package Width", "Package Height"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(" cm", "").str.replace("cm", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dims = ["Package Length", "Package Width", "Package Height"]
    df["Longest Side"] = df[dims].max(axis=1) if all(c in df.columns for c in dims) else float("nan")
    df["Size Category"] = df["Longest Side"].apply(get_size)

    if "Last Updated Time" in df.columns:
        df["Last Updated Time"] = pd.to_datetime(df["Last Updated Time"], errors="coerce")
        df["Day of Week"] = df["Last Updated Time"].dt.day_name()

    if "Dispatch Time" in df.columns:
        df["Dispatch Time"] = pd.to_datetime(df["Dispatch Time"], errors="coerce")
        df["Shift"] = df["Dispatch Time"].dt.hour.apply(classify_shift)
    elif "Assigned Cycle" in df.columns:
        def cyc(c):
            if pd.isna(c): return "Unknown"
            u = str(c).upper()
            if "NS" in u or "NIGHT" in u: return "NS"
            if "PM" in u or "RELO" in u or "C2" in u: return "PM"
            if "AM" in u or "C1" in u: return "AM"
            return "Unknown"
        df["Shift"] = df["Assigned Cycle"].apply(cyc)
    else:
        df["Shift"] = "Unknown"

    return df


def get_station_name(df, filename):
    """Station name from column or filename."""
    if "Station" in df.columns and len(df["Station"].dropna()) > 0:
        return df["Station"].dropna().iloc[0]
    return filename.replace(".csv", "").replace("_", " ").strip()[:20]


def get_date_range(df):
    """Formatted date range string."""
    s = df["Last Updated Time"].min().strftime("%d %b %Y")
    e = df["Last Updated Time"].max().strftime("%d %b %Y")
    return s if s == e else f"{s} - {e}"


def safe_top(series):
    """Most common value or 'N/A'."""
    c = series.dropna().value_counts()
    return c.index[0] if len(c) > 0 else "N/A"


def trunc(labels):
    """Truncate long labels for charts."""
    return [str(l)[:DSP_MAX] + "..." if len(str(l)) > DSP_MAX else str(l) for l in labels]


def bar(data, xl, yl, title, color="steelblue", horiz=False, figsize=CHART):
    """Compact bar chart. Horizontal for rankings."""
    fig, ax = plt.subplots(figsize=figsize)
    labs = trunc(data.index)
    if horiz:
        ax.barh(labs, data.values, color=color)
        ax.invert_yaxis()
    else:
        ax.bar(labs, data.values, color=color)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def bar_dsp(data, title, color="orange"):
    """Horizontal DSP chart. Auto-height."""
    h = max(2, len(data) * 0.3)
    fig, ax = plt.subplots(figsize=(CHART[0], h))
    ax.barh(trunc(data.index), data.values, color=color)
    ax.invert_yaxis()
    ax.set_xlabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def bar_shift(data, title):
    """Shift chart — always shows all 3 shifts."""
    data = data.reindex(SHIFT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=CHART)
    ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER],
           color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    ax.set_xlabel("Shift", fontsize=8)
    ax.set_ylabel("Lost Parcels", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    return fig


def tbl(series, c1, c2):
    """Value counts → numbered table."""
    t = series.reset_index()
    t.columns = [c1, c2]
    t.index = range(1, len(t) + 1)
    return t


def shift_leaderboard(df, total):
    """Leaderboard showing all 3 shifts always. Sorted worst first."""
    counts = df[df["Shift"] != "Unknown"]["Shift"].value_counts()
    rows = []
    for s in SHIFT_ORDER:
        n = int(counts.get(s, 0))
        pct = round(n / total * 100, 1) if total > 0 else 0
        rows.append({"Shift": s, "Lost Parcels": n, "% of Total": f"{pct}%",
                     "Time Window": SHIFT_DEFINITIONS[s]})
    rows.sort(key=lambda r: r["Lost Parcels"], reverse=True)
    t = pd.DataFrame(rows)
    t.index = range(1, len(t) + 1)
    return t


def render_shift(df, total, dr):
    """Full shift tab: definitions, leaderboard, chart, per-shift parcel tables."""
    # Definitions
    st.caption("Parcels assigned to shifts based on **Dispatch Time** (last scan before lost).")
    cols = st.columns(3)
    for i, (s, d) in enumerate(SHIFT_DEFINITIONS.items()):
        cols[i].markdown(f"**{s}:** {d}")
    st.markdown("---")

    # Leaderboard (large — only 3 rows so give it space)
    st.subheader("🏆 Shift Leaderboard")
    lb = shift_leaderboard(df, total)
    st.dataframe(lb, use_container_width=True, height=150)  # fixed height, fills width

    # Chart
    shift_data = df[df["Shift"] != "Unknown"]["Shift"].value_counts()
    st.pyplot(bar_shift(shift_data if len(shift_data) > 0 else pd.Series(dtype="int64"),
                         f"Lost by Shift ({dr})"))
    st.markdown("---")

    # Per-shift parcel tables (expandable — worst first)
    st.subheader("📦 Parcels Per Shift")
    st.caption("Expand a shift to see every parcel + dispatch time for verification.")
    for row in lb.itertuples():
        s_name, count, pct_str = row.Shift, row._2, row._3
        shift_df = df[df["Shift"] == s_name].copy()
        with st.expander(f"**{s_name}** — {count} parcels ({pct_str}) | {SHIFT_DEFINITIONS[s_name]}"):
            if count == 0:
                st.success(f"✅ No parcels lost on {s_name} shift.")
            else:
                cols_show = [c for c in ["Tracking ID", "Dispatch Time", "Cluster", "Aisle",
                             "Sort Zone", "DSP Name", "Size Category"] if c in df.columns]
                if "Dispatch Time" in cols_show:
                    shift_df = shift_df.sort_values("Dispatch Time")
                    shift_df["Dispatch Time"] = shift_df["Dispatch Time"].dt.strftime("%d/%m/%Y %H:%M")
                out = shift_df[cols_show].reset_index(drop=True)
                out.index = range(1, len(out) + 1)
                st.dataframe(out, use_container_width=True)

    unk = len(df[df["Shift"] == "Unknown"])
    if unk > 0:
        st.warning(f"⚠️ {unk} parcels couldn't be assigned (no Dispatch Time). Excluded from rankings.")


# --- MODE TOGGLE ---
mode = st.radio("Mode:", ["Single Station", "Multi-Station Compare"], horizontal=True, key="mode")

# =====================================================================
# SINGLE STATION
# =====================================================================
if mode == "Single Station":

    uploaded_file = st.file_uploader("Upload SCC export (.csv)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        found = [c for c in SENSITIVE_COLS if c in df.columns]
        if found:
            st.warning(f"Sensitive columns removed: {', '.join(found)}")
        df = clean_data(df)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        elif len(df) == 0:
            st.warning("File has no data rows.")
        else:
            st.success(f"Data loaded — {len(df)} packages.")
            dr = get_date_range(df)

            # Summary
            st.subheader(f"Quick Summary ({dr})")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Lost", len(df))
            c2.metric("Worst Cluster", safe_top(df["Cluster"]))
            c3.metric("Worst Aisle", safe_top(df["Aisle"]))
            c4.metric("Worst DSP", str(safe_top(df["DSP Name"]))[:15])
            sk = df[df["Shift"] != "Unknown"]["Shift"]
            c5.metric("Worst Shift", safe_top(sk) if len(sk) > 0 else "N/A")

            # Sweep priority
            cl_r = df["Cluster"].dropna().value_counts()
            if len(cl_r) > 0:
                parts = [f"#{i+1} {cl} ({n})" for i, (cl, n) in enumerate(cl_r.head(3).items())]
                st.info(f"🧹 **Sweep Priority:** {' → '.join(parts)}")

            # Tabs
            t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
                ["Overview", "Location", "Rankings", "DSP & Cycle", "Shift", "Time", "Export", "Bridge"])

            # TAB 1: OVERVIEW
            with t1:
                st.caption("Size breakdown and cluster summary.")
                v = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="ov_v")
                if v == "Chart":
                    sc = df["Size Category"].value_counts()
                    if len(sc) > 0:
                        st.pyplot(bar(sc, "Size", "Lost", f"Lost by Size ({dr})",
                                       color=["green","orange","red","darkred","grey"][:len(sc)]))
                    cc = df["Cluster"].dropna().value_counts()
                    if len(cc) > 0:
                        st.pyplot(bar(cc, "Cluster", "Lost", f"Lost by Cluster ({dr})"))
                else:
                    st.write(df["Size Category"].value_counts())
                    t = df.groupby(["Cluster", "Size Category"]).size().unstack(fill_value=0)
                    t["Total"] = t.sum(axis=1)
                    st.dataframe(t)

            # TAB 2: LOCATION
            with t2:
                st.caption("Drill into a cluster.")
                clusters = sorted(df["Cluster"].dropna().unique())
                if clusters:
                    sel = st.selectbox("Cluster:", clusters, key="cl")
                    f2 = df[df["Cluster"] == sel]
                    st.write(f"{len(f2)} parcels in Cluster {sel}")
                    vb = st.selectbox("View by:", ["Aisle", "Sort Zone"], key="vb")
                    lv = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="lv")
                    ld = f2[vb].dropna().value_counts()
                    if lv == "Chart":
                        if len(ld) > 0:
                            st.pyplot(bar(ld, vb, "Lost", f"Cluster {sel} by {vb}"))
                    else:
                        if len(ld) > 0:
                            st.dataframe(tbl(ld, "Location", "Lost Parcels"))

            # TAB 3: RANKINGS (now includes Cluster)
            with t3:
                st.caption("Top 10 worst locations.")
                rb = st.selectbox("Rank by:", ["Sort Zone", "Aisle", "Cluster"], key="rb")
                rv = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="rv")
                rd = df[rb].dropna().value_counts().head(10)
                if len(rd) > 0:
                    if rv == "Chart":
                        st.pyplot(bar(rd, "Lost", rb, f"Top 10 {rb}s ({dr})",
                                       color="darkred", horiz=True))
                    else:
                        st.dataframe(tbl(rd, rb, "Lost Parcels"))

            # TAB 4: DSP & CYCLE
            with t4:
                st.caption("DSP performance and cycle distribution.")
                dv = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="dv")
                dd = df["DSP Name"].dropna().value_counts()
                cd = df["Assigned Cycle"].dropna().value_counts()
                if dv == "Chart":
                    if len(dd) > 0:
                        st.pyplot(bar_dsp(dd, f"Lost by DSP ({dr})"))
                    if len(cd) > 0:
                        st.pyplot(bar(cd, "Cycle", "Lost", f"Lost by Cycle ({dr})", color="purple"))
                else:
                    if len(dd) > 0:
                        st.subheader("DSP (alphabetical)")
                        st.dataframe(tbl(dd.sort_index(), "DSP", "Lost Parcels"))
                    if len(cd) > 0:
                        st.subheader("Cycle")
                        st.dataframe(tbl(cd, "Cycle", "Lost Parcels"))

            # TAB 5: SHIFT
            with t5:
                render_shift(df, len(df), dr)

            # TAB 6: TIME
            with t6:
                st.caption("Day-of-week patterns.")
                dd2 = df["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                tv = st.radio("Display:", ["Chart", "Table"], horizontal=True, key="tv")
                if tv == "Chart":
                    st.pyplot(bar(dd2, "Day", "Lost", f"Lost by Day ({dr})", color="green"))
                else:
                    st.dataframe(tbl(dd2, "Day", "Lost Parcels"))

                with st.expander("🔍 Parcel Details by Day"):
                    avail = [d for d in DAY_ORDER if d in df["Day of Week"].values]
                    if avail:
                        sd = st.selectbox("Day:", avail, key="ds")
                        ddf = df[df["Day of Week"] == sd]
                        st.write(f"{len(ddf)} parcels on {sd}")
                        show = [c for c in ["Tracking ID","Cluster","Aisle","Sort Zone","DSP Name","Size Category","Shift"] if c in df.columns]
                        st.dataframe(ddf[show])

            # TAB 7: EXPORT
            with t7:
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "Lost_Parcels_Cleaned.csv", "text/csv")

            # TAB 8: BRIDGE
            with t8:
                st.caption("Auto-generated bridge.")
                total = len(df)
                cl_c = df["Cluster"].dropna().value_counts()
                ai_c = df["Aisle"].dropna().value_counts()
                dsp_c = df["DSP Name"].dropna().value_counts()
                sz_c = df["Size Category"].value_counts()
                cy_c = df["Assigned Cycle"].dropna().value_counts()
                day_c = df["Day of Week"].dropna().value_counts()
                sh_c = df[df["Shift"] != "Unknown"]["Shift"].value_counts()

                wc = cl_c.index[0] if len(cl_c) > 0 else "N/A"
                wc_n = cl_c.values[0] if len(cl_c) > 0 else 0
                wc_p = round(wc_n/total*100,1) if total > 0 else 0
                wa = ai_c.index[0] if len(ai_c) > 0 else "N/A"
                wa_n = ai_c.values[0] if len(ai_c) > 0 else 0
                avg_a = ai_c.mean() if len(ai_c) > 0 else 1
                wd = dsp_c.index[0] if len(dsp_c) > 0 else "N/A"
                wd_n = dsp_c.values[0] if len(dsp_c) > 0 else 0
                avg_d = dsp_c.mean() if len(dsp_c) > 0 else 1
                dm = round(wd_n/avg_d,1) if avg_d > 0 else 1.0
                ws = sz_c.index[0] if len(sz_c) > 0 else "N/A"
                ws_n = sz_c.values[0] if len(sz_c) > 0 else 0
                tc = cy_c.index[0] if len(cy_c) > 0 else "N/A"
                wday = day_c.index[0] if len(day_c) > 0 else "N/A"
                wday_n = day_c.values[0] if len(day_c) > 0 else 0
                wsh = sh_c.index[0] if len(sh_c) > 0 else "N/A"
                wsh_n = sh_c.values[0] if len(sh_c) > 0 else 0

                df["Date"] = df["Last Updated Time"].dt.strftime("%d/%m")
                daily = df.groupby("Date").size()
                dl = "\n".join([f"{d} - {n} lost" for d, n in daily.items()])
                sl = "\n".join([f"  {s}: {int(sh_c.get(s,0))} ({round(int(sh_c.get(s,0))/total*100,1)}%)" for s in SHIFT_ORDER])

                cdet = ""
                for cn, cv in cl_c.head(3).items():
                    ta = df[df["Cluster"]==cn]["Aisle"].dropna().value_counts().head(3)
                    al = ", ".join([f"{a} ({n})" for a,n in ta.items()])
                    cdet += f"  Cluster {cn}: {cv} ({round(cv/total*100,1)}%) — {al}\n"

                dlines = "\n".join([f"  {d}: {n} ({round(n/total*100,1)}%)" for d,n in dsp_c.head(3).items()])
                slines = "\n".join([f"  {s}: {n}" for s,n in sz_c.items()])
                stlines = "\n".join([f"  {s}: {n}" for s,n in df["State"].dropna().value_counts().items()]) if "State" in df.columns else "  (Not in export)"

                acts = []
                ac = lambda t: acts.append(f"AC{len(acts)+1}: {t}")
                if wc_p > 40: ac(f"Dedicated PS to Cluster {wc} — {wc_p}% of losts.")
                else: ac(f"PS rotation: top clusters ({', '.join([str(c) for c in cl_c.head(3).index])}).")
                if dm >= 2: ac(f"DSP {wd} stand-down — {dm}x average.")
                elif dm >= 1.5: ac(f"DSP {wd} briefing — {dm}x average.")
                else: ac("Station-wide refresher — losts spread across DSPs.")
                if ws in ["Large Oversize","Small Oversize"]: ac(f"Oversize stow audit Aisle {wa}.")
                elif ws == "Small": ac(f"Small parcel stow briefing — {ws_n} lost.")
                else: ac(f"Stow walk Cluster {wc} for {ws} parcels.")
                if avg_a > 0 and wa_n >= avg_a*3: ac(f"Inspect Aisle {wa} — {round(wa_n/avg_a,1)}x avg.")
                elif avg_a > 0 and wa_n >= avg_a*2: ac(f"PS increase Aisle {wa}.")
                else: ac("Daily PS huddle by aisle.")
                if len(sh_c) > 1 and wsh_n > total*0.5: ac(f"{wsh} shift 5-whys — {wsh_n} losts.")
                relo = sum(cy_c.get(c,0) for c in cy_c.index if "RELO" in str(c))
                if relo > total*0.15: ac(f"RELO review — {relo} losts.")
                if len(daily) > 1 and wday_n > total*0.3: ac(f"Review {wday} staffing — {wday_n} losts.")

                bridge = f"""Lost Parcels Bridge - DRM2
{dr}

Lost (Total): {total}

Daily Breakdown:
{dl}

Shift Breakdown:
{sl}

RC1) Location:
{cdet}
RC2) DSP:
{dlines}

RC3) Size:
{slines}

RC4) Status:
{stlines}

{chr(10).join(acts)}
"""
                st.text_area("Edit:", value=bridge, height=400, key="br")
                st.subheader("Enhance with Quick")
                prompt = f"""Write a Lost Parcels bridge for DRM2. RC1-4 root causes, AC1-4 actions. Be specific.
Data ({dr}): Total={total}, Cluster={wc} ({wc_n},{wc_p}%), Aisle={wa} ({wa_n}), DSP={wd} ({wd_n},{dm}x), Size={ws} ({ws_n}), Day={wday} ({wday_n}), Shift={wsh} ({wsh_n})
Daily: {dl}
Shifts: {sl}
Clusters: {cdet}DSPs: {dlines}
Generate specific actions referencing clusters, aisles, DSPs, sizes, days, shifts."""
                st.code(prompt, language="text")
                st.info("Copy icon → Quick → Ctrl+V")

# =====================================================================
# MULTI-STATION COMPARE
# =====================================================================
else:
    st.subheader("Upload Station Data (2–5)")
    st.caption("One SCC export per station.")
    num = st.slider("Stations:", 2, 5, 2, key="ns")

    uploaded = {}
    cols = st.columns(num)
    for i in range(num):
        with cols[i]:
            f = st.file_uploader(f"Station {i+1}", type="csv", key=f"up_{i}")
            if f: uploaded[i] = f

    if len(uploaded) >= 2:
        stations, names = {}, []
        for i, file in uploaded.items():
            tmp = clean_data(pd.read_csv(file))
            name = get_station_name(tmp, file.name)
            stations[name] = tmp
            names.append(name)

        st.success(f"Loaded: {', '.join(names)}")

        # Summary
        st.subheader("Station Summary")
        mc = st.columns(len(names))
        for i, n in enumerate(names):
            sk = stations[n][stations[n]["Shift"]!="Unknown"]["Shift"]
            mc[i].metric(n, f"{len(stations[n])} lost")
            mc[i].caption(f"Worst shift: {safe_top(sk) if len(sk)>0 else 'N/A'}")

        for n in names:
            cr = stations[n]["Cluster"].dropna().value_counts()
            if len(cr) > 0:
                p = [f"#{i+1} {c} ({v})" for i,(c,v) in enumerate(cr.head(3).items())]
                st.info(f"🧹 **{n}:** {' → '.join(p)}")

        t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs(
            ["Overview","Location","Rankings","DSP & Cycle","Shift","Time","Export","Bridge"])

        # TAB 1: OVERVIEW
        with t1:
            v = st.radio("Display:", ["Chart","Table"], horizontal=True, key="mc_ov")
            if v == "Chart":
                fig, ax = plt.subplots(figsize=CHART)
                ax.bar(names, [len(stations[n]) for n in names], color=STATION_COLORS[:len(names)])
                ax.set_ylabel("Lost", fontsize=8); ax.set_title("Total Lost by Station", fontsize=9)
                ax.tick_params(labelsize=7); plt.tight_layout()
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(8,2.5))
                x = range(len(SIZE_ORDER)); w = 0.8/len(names)
                for i,n in enumerate(names):
                    cts = [len(stations[n][stations[n]["Size Category"]==s]) for s in SIZE_ORDER]
                    ax2.bar([xi+(i-len(names)/2+0.5)*w for xi in x], cts, w, label=n, color=STATION_COLORS[i])
                ax2.set_xticks(x); ax2.set_xticklabels(SIZE_ORDER, fontsize=7)
                ax2.set_ylabel("Lost", fontsize=8); ax2.legend(fontsize=7); plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.dataframe(pd.DataFrame({"Station":names,"Total":[len(stations[n]) for n in names]}, index=range(1,len(names)+1)))
                sz = {n:[len(stations[n][stations[n]["Size Category"]==s]) for s in SIZE_ORDER] for n in names}
                st.dataframe(pd.DataFrame(sz, index=SIZE_ORDER))

        # TAB 2: LOCATION
        with t2:
            for i, n in enumerate(names):
                with st.expander(f"📍 {n}", expanded=False):
                    sdf = stations[n]
                    cls = sorted(sdf["Cluster"].dropna().unique())
                    if cls:
                        sel = st.selectbox(f"Cluster ({n}):", cls, key=f"mc_cl_{i}")
                        flt = sdf[sdf["Cluster"]==sel]
                        st.write(f"{len(flt)} parcels")
                        ai = flt["Aisle"].dropna().value_counts().head(10)
                        if len(ai) > 0:
                            st.pyplot(bar(ai,"Aisle","Lost",f"{n} — {sel} Aisles", color=STATION_COLORS[i], figsize=CHART_SM))

        # TAB 3: RANKINGS (includes Cluster)
        with t3:
            rb = st.selectbox("Rank by:", ["Sort Zone","Aisle","Cluster"], key="mc_rb")
            rv = st.radio("Display:", ["Chart","Table"], horizontal=True, key="mc_rv")
            for i, n in enumerate(names):
                d = stations[n][rb].dropna().value_counts().head(10)
                if len(d) > 0:
                    if rv == "Chart":
                        st.pyplot(bar(d,"Lost",rb,f"{n} — Top 10 {rb}s", color=STATION_COLORS[i], horiz=True, figsize=CHART_SM))
                    else:
                        st.subheader(n)
                        st.dataframe(tbl(d, rb, "Lost Parcels"))

        # TAB 4: DSP & CYCLE
        with t4:
            dv = st.radio("Display:", ["Chart","Table"], horizontal=True, key="mc_dv")
            if dv == "Chart":
                for i, n in enumerate(names):
                    dsp = stations[n]["DSP Name"].dropna().value_counts().head(10)
                    if len(dsp) > 0:
                        st.pyplot(bar_dsp(dsp, f"{n} — Worst DSPs", color=STATION_COLORS[i]))
            else:
                for i, n in enumerate(names):
                    with st.expander(f"{n}", expanded=False):
                        dsp = stations[n]["DSP Name"].dropna().value_counts()
                        if len(dsp) > 0:
                            st.dataframe(tbl(dsp.sort_index(), "DSP", "Lost Parcels"))
                        cy = stations[n]["Assigned Cycle"].dropna().value_counts()
                        if len(cy) > 0:
                            st.dataframe(tbl(cy, "Cycle", "Lost Parcels"))

        # TAB 5: SHIFT
        with t5:
            st.subheader("⏱️ Shift Comparison")
            st.caption(" | ".join([f"**{s}:** {d}" for s,d in SHIFT_DEFINITIONS.items()]))

            fig_sh, ax_sh = plt.subplots(figsize=CHART)
            x = range(len(SHIFT_ORDER)); w = 0.8/len(names)
            for i, n in enumerate(names):
                sd = stations[n][stations[n]["Shift"]!="Unknown"]["Shift"].value_counts()
                cts = [sd.get(s,0) for s in SHIFT_ORDER]
                ax_sh.bar([xi+(i-len(names)/2+0.5)*w for xi in x], cts, w, label=n, color=STATION_COLORS[i])
            ax_sh.set_xticks(x); ax_sh.set_xticklabels(SHIFT_ORDER, fontsize=8)
            ax_sh.set_ylabel("Lost", fontsize=8); ax_sh.set_title("Shift Comparison", fontsize=9)
            ax_sh.tick_params(labelsize=7); ax_sh.legend(fontsize=7); plt.tight_layout()
            st.pyplot(fig_sh)

            st.markdown("---")
            for i, n in enumerate(names):
                with st.expander(f"📍 {n} — Shift Detail", expanded=False):
                    sdf = stations[n]
                    render_shift(sdf, len(sdf), get_date_range(sdf) if "Last Updated Time" in sdf.columns else "")

        # TAB 6: TIME
        with t6:
            fig, ax = plt.subplots(figsize=CHART)
            for i, n in enumerate(names):
                if "Day of Week" in stations[n].columns:
                    dd = stations[n]["Day of Week"].dropna().value_counts().reindex(DAY_ORDER, fill_value=0)
                    ax.plot(dd.index, dd.values, marker="o", label=n, color=STATION_COLORS[i], linewidth=2, markersize=4)
            ax.set_ylabel("Lost", fontsize=8); ax.set_title("Lost by Day", fontsize=9)
            ax.tick_params(labelsize=7); ax.legend(fontsize=7); plt.xticks(rotation=0); plt.tight_layout()
            st.pyplot(fig)

        # TAB 7: EXPORT
        with t7:
            for n in names:
                st.download_button(f"Download {n}", stations[n].to_csv(index=False),
                                   f"Lost_{n}.csv", "text/csv", key=f"dl_{n}")
            combined = pd.concat([stations[n].assign(Station=n) for n in names], ignore_index=True)
            st.download_button("Download Combined", combined.to_csv(index=False),
                               "Lost_Combined.csv", "text/csv", key="dl_all")

        # TAB 8: BRIDGE
        with t8:
            comp = []
            for n in names:
                sdf = stations[n]; sk = sdf[sdf["Shift"]!="Unknown"]["Shift"]
                comp.append({"Station":n, "Total":len(sdf), "Worst Cluster":safe_top(sdf["Cluster"]),
                             "Worst Aisle":safe_top(sdf["Aisle"]), "Worst DSP":safe_top(sdf["DSP Name"]),
                             "Worst Shift":safe_top(sk) if len(sk)>0 else "N/A"})
            st.dataframe(pd.DataFrame(comp, index=range(1,len(comp)+1)))

            best = min(names, key=lambda n: len(stations[n]))
            worst = max(names, key=lambda n: len(stations[n]))
            gap = len(stations[worst]) - len(stations[best])
            st.write(f"**Best:** {best} ({len(stations[best])}) | **Worst:** {worst} ({len(stations[worst])}) | **Gap:** {gap}")

            summ = "\n".join([f"- {n}: {len(stations[n])} losts, cluster={safe_top(stations[n]['Cluster'])}, DSP={safe_top(stations[n]['DSP Name'])}" for n in names])
            st.code(f"Compare stations:\n{summ}\nBest={best}, Worst={worst}, Gap={gap}\nRecommend what {worst} can learn from {best}.", language="text")
            st.info("Copy → Quick → Ctrl+V")

    elif len(uploaded) == 1:
        st.warning("Upload at least 2 files.")
    else:
        st.info("Upload CSV files above.")
