import streamlit as st
import pandas as pd
import os
import sqlite3
import requests
import pydeck as pdk
import platform

# Auto-detect if running locally or in production (Streamlit Cloud uses Linux)
IS_LOCAL = platform.system() == "Windows"
API_BASE = "http://127.0.0.1:8000" if IS_LOCAL else "https://parkiq-glrk.onrender.com"

st.set_page_config(page_title="ParkIQ Dashboard", layout="wide", page_icon="🚦")

# ── CSS overrides for a cleaner look ──────────────────────────────────────────
st.markdown("""
<style>
    .risk-card-high   { background:#ff4b4b22; border-left:4px solid #ff4b4b; padding:10px 14px; border-radius:6px; margin-bottom:8px; }
    .risk-card-medium { background:#ffa50022; border-left:4px solid #ffa500; padding:10px 14px; border-radius:6px; margin-bottom:8px; }
    .risk-card-low    { background:#00c80022; border-left:4px solid #00c800; padding:10px 14px; border-radius:6px; margin-bottom:8px; }
    .suggest-box      { background:#1e3a5f22; border-left:4px solid #4a9eff; padding:10px 14px; border-radius:6px; margin-bottom:6px; }
    .stat-label       { font-size:0.78rem; color:#888; text-transform:uppercase; letter-spacing:0.05em; }
    @keyframes blink-red {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px #ff2222, 0 0 20px #ff2222; }
        50%      { opacity: 0.2; box-shadow: none; }
    }
    .led-blink { animation: blink-red 0.6s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

PAGES = [
    "🧠 AI Predictive Impact Map",
    "⚠️ High-Risk Area Analysis",
    "📡 IoT Sensor Monitor",
    "🎥 City-Wide CCTV Network",
]

default_index = 0
if "page" in st.query_params:
    query_page = st.query_params["page"]
    if query_page in PAGES:
        default_index = PAGES.index(query_page)

page = st.sidebar.radio("Navigate", PAGES, index=default_index)

if st.query_params.get("page") != page:
    loader_placeholder = st.empty()
    with loader_placeholder.container():
        with st.spinner(f"Loading {page}..."):
            import time
            time.sleep(0.6)  # Forced delay to make loader visible
    loader_placeholder.empty()
    st.query_params["page"] = page

try:
    if page == "🎥 City-Wide CCTV Network":
        requests.post(f"{API_BASE}/api/set-mode", json={"mode": "CCTV"}, timeout=0.5)
    else:
        requests.post(f"{API_BASE}/api/set-mode", json={"mode": "IOT"}, timeout=0.5)
except Exception:
    pass

DATA_PATH = "data/violations_scored.csv"
DB_PATH   = "data/parkiq.db"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "created_datetime" in df.columns:
            # Parse dates and extract month/day for filtering
            df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
            df["date"] = df["created_datetime"].dt.date
        return df
    return pd.DataFrame()

# ── Global AI Threat Monitor (Simulating 200,000 Cameras) ─────────────
@st.fragment(run_every=8)
def global_threat_monitor():
    df_mon = load_data()
    if not df_mon.empty and "location" in df_mon.columns:
        import random
        crit_pool = df_mon[df_mon["CIS"] > 80]
        if not crit_pool.empty:
            event = crit_pool.sample(1).iloc[0]
            loc = event['location']
            cis = event['CIS']
            cam_id = f"CAM-{random.randint(100000, 999999)}"
            # st.toast(f"🚨 **{cam_id} LIVE ALERT:** Critical gridlock detected at {loc}. CIS: {cis:.1f}. Tow dispatch requested.", icon='🚨')

global_threat_monitor()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE VIOLATIONS MAP
# ══════════════════════════════════════════════════════════════════════════════
if page == "🧠 AI Predictive Impact Map":
    st.title("🧠 AI Predictive Impact Map")
    st.caption("AI-driven real-time congestion forecasting powered by 5 months of historical Bangalore Traffic Police data (Jan–May).")
    
    with st.spinner("Loading AI Impact Engine..."):
        df_full = load_data()

    if not df_full.empty and "latitude" in df_full.columns and "longitude" in df_full.columns:
        df_full = df_full.dropna(subset=["latitude", "longitude"])
        
        # Simulate Live AI Environment by filtering by hour
        st.markdown("### 🎛️ Live AI Simulation Control")
        
        df = df_full.copy()
        
        # Center the time slider
        _, c_hour, _ = st.columns([1, 2, 1])
        with c_hour:
            if "hour" in df.columns:
                selected_hour = st.slider("Simulate Time Range (Hours):", min_value=0, max_value=23, value=(8, 18), format="%02d:00", key="h1")
                df = df[(df["hour"] >= selected_hour[0]) & (df["hour"] <= selected_hour[1])]
                
        st.info(f"Displaying AI Congestion Impact Scores for {len(df):,} violations matching the selected filters.")

        if df.empty:
            st.warning("No data available for the selected parameters.")
            st.stop()

        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()

        def get_color(cis):
            if cis > 70: return [255, 50,  50,  180]
            if cis > 40: return [255, 165, 0,   180]
            return           [0,   200, 80,  150]

        df["color"] = df["CIS"].apply(get_color)

        high_risk  = df[df["CIS"] > 70]
        med_risk   = df[(df["CIS"] > 40) & (df["CIS"] <= 70)]
        low_risk   = df[df["CIS"] <= 40]
        avg_cis    = df["CIS"].mean()
        crit_risk  = df[df["CIS"] > 80]

        # ── KPI Row ───────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("📊 Total Violations",    f"{len(df):,}")
        k2.metric("🔴 High Risk (CIS>70)",  f"{len(high_risk):,}",
                  delta=f"{len(high_risk)/len(df)*100:.1f}% of total", delta_color="inverse")
        k3.metric("🟠 Medium Risk",         f"{len(med_risk):,}")
        k4.metric("🟢 Low Risk",            f"{len(low_risk):,}")
        k5.metric("💥 Critical (CIS>80)",   f"{len(crit_risk):,}",
                  delta=f"Avg CIS: {avg_cis:.1f}", delta_color="off")

        if len(crit_risk) > 0 and "location" in df.columns:
            crit_locs = crit_risk["location"].unique()
            loc_str = ", ".join(crit_locs[:3])
            if len(crit_locs) > 3:
                loc_str += f" and {len(crit_locs)-3} more areas"
            st.error(f"🚨 **CRITICAL ALERT:** {len(crit_risk)} critical violations detected at **{loc_str}**. These areas are completely gridlocked and require immediate tow units.")

        st.markdown("---")

        # ── Map ───────────────────────────────────────────────────────────────
        st.subheader("Interactive Violation Heatmap")
        with st.spinner('Loading map...'):
            df_render = df[["latitude", "longitude", "color", "vehicle_type", "CIS"]].copy()
            if "location" in df.columns:
                df_render["location"] = df["location"]
            else:
                df_render["location"] = "Unknown"
            df_render["CIS_str"] = df_render["CIS"].apply(lambda x: f"{x:.1f}")
    
            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=45)
            scatter = pdk.Layer(
                "ScatterplotLayer", data=df_render,
                get_position="[longitude, latitude]",
                get_fill_color="color", get_radius=15,
                radius_min_pixels=2, radius_max_pixels=15, pickable=True,
            )
            heatmap_layer = pdk.Layer(
                "HeatmapLayer", data=df_render,
                get_position="[longitude, latitude]",
                get_weight="CIS", aggregation="MEAN",
                opacity=0.8,
            )
            st.pydeck_chart(pdk.Deck(
                map_style="road",
                layers=[heatmap_layer, scatter],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>📍 Location:</b> {location} <br/>"
                            "<b>🚗 Vehicle:</b> {vehicle_type} <br/>"
                            "<b>⚠️ CIS Score:</b> {CIS_str}"
                },
            ))

        st.markdown("---")
        
        # ── Targeted Enforcement Priority Engine ─────────────────────────────────
        st.subheader("🎯 Targeted Enforcement Priority Engine")
        st.caption(f"Real-time dispatch recommendations for the selected time range ({selected_hour[0]:02d}:00 - {selected_hour[1]:02d}:00)")

        if "location" in df.columns:
            # Group by location to find the worst hotspots at this specific hour
            loc_stats = (
                df[df["CIS"] > 40]
                .groupby("location")
                .agg(
                    violation_count=("CIS", "count"),
                    avg_cis=("CIS", "mean"),
                    max_cis=("CIS", "max"),
                )
                .sort_values("avg_cis", ascending=False)
                .reset_index()
            )

            if not loc_stats.empty:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 📍 Top Active Hotspots")
                    
                    display_df = loc_stats.head(5).rename(columns={
                        "location": "Location",
                        "violation_count": "Total Violations",
                        "avg_cis": "Average Impact",
                        "max_cis": "Peak Danger"
                    })
                    
                    st.dataframe(
                        display_df.style.format({"Average Impact": "{:.1f}", "Peak Danger": "{:.1f}"}),
                        width='stretch',
                        hide_index=True
                    )
                
                with col2:
                    st.markdown("### 🚓 Immediate Dispatch Directives")
                    # Generate AI suggestions for the top 2 areas
                    top_areas = loc_stats.head(2)
                    
                    for i, row in top_areas.iterrows():
                        loc_name = row["location"]
                        max_score = row["max_cis"]
                        v_count = int(row["violation_count"])
                        
                        if max_score > 80:
                            badge = "🔴 CRITICAL"
                            action = f"**Go to {loc_name} immediately.** Severe blockage with {v_count} active high-impact violations. Deploy tow trucks to clear the carriageway."
                            css = "risk-card-high"
                        elif max_score > 70:
                            badge = "🟠 HIGH PRIORITY"
                            action = f"**Dispatch patrol to {loc_name}.** {v_count} violations are choking the intersection. Issue e-challans and clear heavy vehicles."
                            css = "risk-card-high"
                        else:
                            badge = "🟡 MEDIUM PRIORITY"
                            action = f"**Monitor {loc_name}.** {v_count} moderate-impact violations detected. Send a beat officer to issue warnings."
                            css = "risk-card-medium"
                            
                        st.markdown(f"""
                        <div class="{css}">
                          <strong>{badge} | Dispatch Unit</strong><br>
                          <span style="font-size:0.92rem">{action}</span>
                        </div>""", unsafe_allow_html=True)
            
            else:
                st.success(f"No high-impact hotspots detected for the selected time range.")
        else:
            st.info("Location data missing from dataset.")

        st.markdown("---")
        st.subheader("📊 Live Zone Analytics")
        
        ac1, ac2 = st.columns(2)
        with ac1:
            if "vehicle_type" in df.columns:
                st.markdown("**Violations by Vehicle Type**")
                v_counts = df["vehicle_type"].value_counts().head(7)
                st.bar_chart(v_counts, color="#ff4b4b")
        with ac2:
            if "violation_type" in df.columns:
                st.markdown("**Top Violation Categories**")
                viol_counts = df["violation_type"].value_counts().head(7)
                st.bar_chart(viol_counts, color="#ffa500")
        
        if "hour" in df.columns and len(df) > 0:
            st.markdown("**Congestion Trend (Selected Time Range)**")
            trend = df["hour"].value_counts().sort_index()
            st.area_chart(trend, color="#ff4b4b")

    else:
        st.warning("No scored data found! Run: `python src/data_pipeline.py`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HIGH-RISK AREA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ High-Risk Area Analysis":
    st.title("⚠️ High-Risk Area Analysis")
    st.caption("AI-powered hotspot identification trained on Bangalore Traffic Police violation records (Jan–May).")

    with st.spinner("Loading AI Impact Engine..."):
        df_full = load_data()
    if df_full.empty:
        st.warning("No data found. Run `python src/data_pipeline.py` first.")
        st.stop()

    df_full = df_full.dropna(subset=["latitude", "longitude"])

    st.markdown("### 🎛️ Live AI Simulation Control")
    df = df_full.copy()
    
    _, c_hour, _ = st.columns([1, 2, 1])
    with c_hour:
        if "hour" in df.columns:
            selected_hour = st.slider("Simulate Time Range (Hours):", min_value=0, max_value=23, value=(8, 18), format="%02d:00", key="h2")
            df = df[(df["hour"] >= selected_hour[0]) & (df["hour"] <= selected_hour[1])]
            
    st.info(f"Generating High-Risk Policy Analysis for {len(df):,} violations matching the selected filters.")

    if df.empty:
        st.warning("No high-risk data available for the selected parameters.")
        st.stop()

    # ── Compute hotspot summary ───────────────────────────────────────────────
    high_risk_df = df[df["CIS"] > 70].copy()
    total        = len(df)
    n_high       = len(high_risk_df)
    n_crit       = len(df[df["CIS"] > 80])
    pct_high     = n_high / total * 100 if total > 0 else 0
    avg_cis_high = high_risk_df["CIS"].mean() if n_high > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 High-Risk Violations",  f"{n_high:,}",  f"{pct_high:.1f}% of all",    delta_color="inverse")
    k2.metric("💥 Critical Violations",   f"{n_crit:,}",  "CIS > 80 — Tow required",    delta_color="inverse")
    k3.metric("📍 Avg CIS (High-Risk)",   f"{avg_cis_high:.1f}", "Score out of 100",     delta_color="off")
    k4.metric("📊 Active Violations",     f"{total:,}",   f"{selected_hour[0]:02d}:00 - {selected_hour[1]:02d}:00", delta_color="off")

    st.markdown("---")

    # ── Top hotspot locations ─────────────────────────────────────────────────
    st.subheader("🔥 Top 10 High-Risk Locations")

    if "location" in df.columns:
        loc_stats = (
            df[df["CIS"] > 70]
            .groupby("location")
            .agg(
                violation_count=("CIS", "count"),
                avg_cis=("CIS", "mean"),
                max_cis=("CIS", "max"),
            )
            .sort_values("avg_cis", ascending=False)
            .head(10)
            .reset_index()
        )

        # Rename columns for display
        display_loc = loc_stats.rename(columns={
            "location": "Location",
            "violation_count": "High-Risk Violations",
            "avg_cis": "Average Impact",
            "max_cis": "Peak Danger"
        })
        
        st.dataframe(
            display_loc.style.format({"Average Impact": "{:.1f}", "Peak Danger": "{:.1f}"}),
            width='stretch',
            hide_index=True
        )

    st.markdown("---")

    # ── Hotspot map ───────────────────────────────────────────────────────────
    st.subheader("🗺️ High-Risk Zone Map (CIS > 70)")
    with st.spinner('Loading map...'):
        hr_render = high_risk_df[["latitude", "longitude", "CIS", "vehicle_type"]].copy()
        if "location" in high_risk_df.columns:
            hr_render["location"] = high_risk_df["location"]
        else:
            hr_render["location"] = "Unknown"
        hr_render["color"] = [[255, 50, 50, 200]] * len(hr_render)
        hr_render["CIS_str"] = hr_render["CIS"].apply(lambda x: f"{x:.1f}")
    
        view_state = pdk.ViewState(
            latitude=df["latitude"].mean(), longitude=df["longitude"].mean(), zoom=12, pitch=40
        )
        layer = pdk.Layer(
            "HeatmapLayer", data=hr_render,
            get_position="[longitude, latitude]",
            get_weight="CIS", aggregation="MEAN",
            opacity=0.8,
        )
        scatter = pdk.Layer(
            "ScatterplotLayer", data=hr_render,
            get_position="[longitude, latitude]",
            get_fill_color="color", get_radius=20,
            radius_min_pixels=3, radius_max_pixels=20, pickable=True,
        )
        st.pydeck_chart(pdk.Deck(
            map_style="road", layers=[layer, scatter], initial_view_state=view_state,
            tooltip={
                "html": "<b>📍 Location:</b> {location} <br/>"
                        "<b>🚗 Vehicle:</b> {vehicle_type} <br/>"
                        "<b>⚠️ CIS Score:</b> {CIS_str}"
            },
        ))

    st.markdown("---")

    # ── Vehicle type breakdown for high-risk ─────────────────────────────────
    st.subheader("🚗 High-Risk Violations by Vehicle Type")
    c1, c2 = st.columns(2)
    with c1:
        if "vehicle_type" in high_risk_df.columns:
            vt_counts = high_risk_df["vehicle_type"].value_counts()
            st.bar_chart(vt_counts)
    with c2:
        if "hour" in high_risk_df.columns:
            st.markdown("**Peak Hours for High-Risk Violations**")
            hr_hour = high_risk_df.groupby("hour").size()
            st.bar_chart(hr_hour)

    st.markdown("---")

    # ── AI Suggestions ────────────────────────────────────────────────────────
    st.subheader("💡 AI-Generated Resolution Suggestions")
    st.info("Suggestions are generated from pattern analysis of the CIS scores, vehicle types, hotspot clustering, and peak-hour distributions.")

    # Dynamic suggestions based on data
    suggestions = []

    if n_crit > 0:
        suggestions.append(("🚨 Immediate Tow Operations", "HIGH",
            f"{n_crit:,} violations scored above 80 (CRITICAL). Deploy tow trucks to the top 3 hotspot locations immediately. "
            "These are junction-adjacent violations by heavy vehicles during peak hours — maximum congestion impact."))

    if "vehicle_type" in high_risk_df.columns:
        top_vt = high_risk_df["vehicle_type"].value_counts().idxmax() if len(high_risk_df) > 0 else "N/A"
        suggestions.append(("🚛 Vehicle-Specific Enforcement", "HIGH",
            f"'{top_vt}' is the dominant vehicle type in high-risk zones. "
            "Issue standing orders for officers to prioritise this category. Consider dedicated no-parking signage at hotspot entries."))

    if "hour" in high_risk_df.columns and len(high_risk_df) > 0:
        peak_hour = int(high_risk_df.groupby("hour").size().idxmax())
        suggestions.append(("⏰ Peak-Hour Patrol Scheduling", "MEDIUM",
            f"High-risk violations peak at {peak_hour:02d}:00 hrs. Shift patrol rosters to ensure maximum officer presence "
            f"from {max(0,peak_hour-1):02d}:00–{min(23,peak_hour+2):02d}:00 at top hotspot locations."))

    if "junction_name" in df.columns:
        n_junction = len(high_risk_df[high_risk_df.get("near_junction", 0) == 1]) if "near_junction" in high_risk_df.columns else 0
        suggestions.append(("🚦 Junction Clearance Priority", "HIGH",
            f"Junction-adjacent violations carry 20% higher CIS weight. Coordinate with traffic signal control to "
            "extend green-phase duration at the top 5 high-CIS junctions during peak hours to ease downstream queuing."))

    suggestions.append(("📱 Digital Notice Dispatch", "MEDIUM",
        "For MEDIUM-priority violations (CIS 40–70), integrate with BBMP vehicle registration database to auto-send "
        "e-challans via SMS/WhatsApp, reducing officer field time by an estimated 60%."))

    suggestions.append(("📡 IoT Sensor Expansion", "LOW",
        "Deploy additional HC-SR04 ultrasonic nodes (ESP32-based) at the top 5 hotspot coordinates. "
        "Real-time sensor data will allow sub-minute violation detection vs. current camera-latency pipeline."))

    suggestions.append(("🧠 Predictive Pre-positioning", "LOW",
        "Use the historical peak-hour + geo_cell frequency table to pre-position enforcement officers "
        "10–15 minutes before predicted violation spikes. This converts reactive enforcement to proactive deterrence."))

    priority_class = {"HIGH": "risk-card-high", "MEDIUM": "risk-card-medium", "LOW": "risk-card-low"}
    for title, priority, body in suggestions:
        css = priority_class.get(priority, "risk-card-low")
        badge_icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[priority]
        st.markdown(f"""
<div class="{css}">
  <strong>{badge_icon} [{priority}] {title}</strong><br>
  <span style="font-size:0.92rem">{body}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── CIS Distribution ──────────────────────────────────────────────────────
    st.subheader("📈 CIS Score Distribution")
    bins = [0, 30, 55, 70, 80, 100]
    labels = ["LOW (0-30)", "MEDIUM (31-55)", "HIGH (56-70)", "HIGH+ (71-80)", "CRITICAL (81-100)"]
    df["cis_band"] = pd.cut(df["CIS"], bins=bins, labels=labels, include_lowest=True)
    band_counts = df["cis_band"].value_counts().sort_index()
    st.bar_chart(band_counts)

    # ── Raw high-risk table ───────────────────────────────────────────────────
    st.subheader("📋 Raw High-Risk Records (Top 50)")
    show_cols = [c for c in ["location", "vehicle_type", "CIS", "priority", "action",
                              "junction_name", "hour"] if c in df.columns]
    st.dataframe(
        df[df["CIS"] > 70].sort_values("CIS", ascending=False)[show_cols].head(50),
        width='stretch',
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — IoT SENSOR MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 IoT Sensor Monitor":
    st.title("📡 IoT Sensor Monitor")
    st.caption("Live HC-SR04 ultrasonic events from the ESP32 parking node")

    @st.fragment(run_every=2)
    def auto_update_table():
        if not os.path.exists(DB_PATH):
            st.warning("Database not found. Make sure the backend API has been run at least once.")
            return

        conn = sqlite3.connect(DB_PATH)
        events_df = pd.read_sql_query(
            "SELECT * FROM sensor_events ORDER BY id DESC LIMIT 50", conn
        )
        # Summary stats
        total_events = pd.read_sql_query("SELECT COUNT(*) as n FROM sensor_events", conn).iloc[0]["n"]
        violations   = pd.read_sql_query(
            "SELECT COUNT(*) as n FROM sensor_events WHERE event='VIOLATION_CONFIRMED'", conn
        ).iloc[0]["n"]
        
        # Calculate live nodes (events in the last 60 seconds)
        live_nodes_df = pd.read_sql_query(
            "SELECT DISTINCT device_id FROM sensor_events WHERE timestamp > strftime('%s','now') - 60", conn
        )
        live_nodes_count = len(live_nodes_df)
        if live_nodes_count > 0:
            active_nodes = ", ".join(live_nodes_df["device_id"].tolist())
            delta_val = f"↑ {active_nodes}"
        else:
            delta_val = "↓ Offline"
            
        conn.close()

        k1, k2, k3 = st.columns(3)
        k1.metric("Total IoT Events",   f"{int(total_events):,}")
        k2.metric("Violations Detected",f"{int(violations):,}")
        k3.metric("Live Nodes",         f"{live_nodes_count}", delta=delta_val)

        st.markdown("---")
        if not events_df.empty:
            st.dataframe(events_df, width='stretch')
        else:
            st.info("No sensor events yet. Start the ESP32 node!")

    auto_update_table()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CITY-WIDE CCTV NETWORK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎥 City-Wide CCTV Network":
    import requests as req

    st.title("🎥 City-Wide CCTV Network — Live Node Dashboard")
    st.caption("Monitoring real-time parking detections across 200,000 simulated city-wide computer vision nodes")

    # ── Auto-switch ESP32 to CCTV mode silently on page load ──────────────────
    try:
        req.post(f"{API_BASE}/api/set-mode", json={"mode": "CCTV"}, timeout=1)
    except Exception:
        pass  # Backend offline — no crash

    @st.fragment(run_every=2)
    def render_hardware_badge():
        try:
            state = req.get(f"{API_BASE}/api/device-state", timeout=1).json()
            alert_level = state.get("alert_level", "VACANT")
        except Exception:
            alert_level = "VACANT"
        level_cfg = {
            "VACANT":   {"color": "#555",    "bg": "#22222233", "label": "⚫ NO VIOLATION",  "dot": "#555",    "blink": False},
            "LOW":      {"color": "#00c800", "bg": "#00c80022", "label": "🟢 LOW",           "dot": "#00c800", "blink": False},
            "MEDIUM":   {"color": "#f5c200", "bg": "#f5c20022", "label": "🟡 MEDIUM",        "dot": "#f5c200", "blink": False},
            "HIGH":     {"color": "#ff4b4b", "bg": "#ff4b4b22", "label": "🔴 HIGH",          "dot": "#ff4b4b", "blink": False},
            "CRITICAL": {"color": "#ff1111", "bg": "#ff111122", "label": "🚨 CRITICAL",      "dot": "#ff1111", "blink": True },
        }
        cfg = level_cfg.get(alert_level, level_cfg["VACANT"])
        dot_class = "led-blink" if cfg["blink"] else ""
        dot_style = (
            f"width:14px;height:14px;border-radius:50%;background:{cfg['dot']};"
            f"display:inline-block;margin-right:6px;"
            + (f"box-shadow:0 0 8px {cfg['dot']};" if not cfg["blink"] and alert_level != "VACANT" else "")
        )
        # st.markdown(f"""
        # <div style="display:flex;align-items:center;gap:16px;padding:10px 18px;
        #             border-radius:10px;background:#1a1a2e;margin-bottom:12px;border:1px solid #333;">
        #     <span style="font-size:1rem;font-weight:700;color:#fff;">🖥️ ESP32 Hardware Node</span>
        #     <span style="background:#00c8ff22;color:#00c8ff;border:1px solid #00c8ff;
        #                 padding:3px 12px;border-radius:20px;font-size:0.82rem;font-weight:700;">
        #         📷 MODE: CCTV — Cameras Active
        #     </span>
        #     <span style="background:{cfg['bg']};color:{cfg['color']};border:1px solid {cfg['color']};
        #                 padding:3px 12px;border-radius:20px;font-size:0.82rem;font-weight:700;
        #                 display:inline-flex;align-items:center;">
        #         <span class="{dot_class}" style="{dot_style}"></span>
        #         ALERT: {cfg['label']}
        #     </span>
        #     <span style="color:#888;font-size:0.78rem;">HC-SR04 sensor disabled</span>
        # </div>
        # """, unsafe_allow_html=True)

    render_hardware_badge()

    # st.markdown("---")

    @st.fragment
    def calibration_tool():
        import json, base64, io
        from PIL import Image
        import streamlit.components.v1 as components

        with st.expander("🛠️ Zone Calibration Tool — Click to Define No-Parking Zones", expanded=False):
            st.caption("Upload a frame from your camera. Click directly on the image — lines draw **instantly**. Hit **Send to Streamlit** when done.")

            cam_id_calib = st.selectbox("Camera to Calibrate", ["CCTV-CAM-01", "CCTV-CAM-02"], key="calib_cam")
            frame_src = st.radio("Frame Source", ["Upload an image/screenshot", "Use first frame from uploaded video"], horizontal=True, key="frame_src")

            calib_img = None
            if frame_src == "Upload an image/screenshot":
                img_file = st.file_uploader("Upload camera frame (JPG/PNG)", type=["jpg", "jpeg", "png"], key="calib_img_upload")
                if img_file:
                    calib_img = Image.open(img_file).convert("RGB")
            else:
                video_path_calib = "data/uploaded_video.mp4"
                if os.path.exists(video_path_calib):
                    import cv2 as _cv2
                    _cap = _cv2.VideoCapture(video_path_calib)
                    _ret, _frame = _cap.read()
                    _cap.release()
                    if _ret:
                        calib_img = Image.fromarray(_cv2.cvtColor(_frame, _cv2.COLOR_BGR2RGB))
                        st.success("✅ First frame extracted from uploaded video.")
                    else:
                        st.warning("Could not read video. Upload video in the section below first.")
                else:
                    st.info("No video uploaded yet. Upload a video below, or switch to image upload.")

            if calib_img:
                # Convert image to base64 for embedding in HTML
                buf = io.BytesIO()
                calib_img.save(buf, format="JPEG", quality=80)
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                orig_w, orig_h = calib_img.size
                display_w = min(760, orig_w)
                display_h = int(orig_h * display_w / orig_w)

                canvas_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; background:#111; font-family:sans-serif; color:#eee; }}
  #toolbar {{ padding:8px; background:#1a1a2e; display:flex; gap:8px; align-items:center; flex-wrap:wrap; border-bottom:1px solid #333; }}
  #status {{ flex:1; font-size:0.85rem; color:#aef; font-weight:600; }}
  button {{ padding:6px 14px; border-radius:6px; border:none; cursor:pointer; font-size:0.82rem; font-weight:700; }}
  #btnUndo  {{ background:#333; color:#eee; }}
  #btnReset {{ background:#553; color:#ffd; }}
  #btnSend  {{ background:#1a7a3a; color:#fff; }}
  #btnSend:disabled {{ background:#333; color:#666; cursor:not-allowed; }}
  #container {{ position:relative; display:inline-block; line-height:0; }}
  #canvas {{ position:absolute; top:0; left:0; cursor:crosshair; }}
  #toast {{ display:none; margin:8px; padding:8px 14px; border-radius:6px; font-weight:700; font-size:0.85rem; }}
  .toast-ok  {{ background:#1a7a3a; color:#fff; }}
  .toast-err {{ background:#7a1a1a; color:#fff; }}
</style>
</head>
<body>
<div id="toolbar">
  <div id="status">Step 1: Click the LEFT edge of the road/lane</div>
  <button id="btnUndo" onclick="undo()">↩️ Undo</button>
  <button id="btnReset" onclick="reset()">🔄 Reset</button>
  <button id="btnSend" onclick="sendPoints()" disabled>💾 Save Calibration</button>
</div>
<div id="toast"></div>
<div id="container">
  <img id="bg" src="data:image/jpeg;base64,{img_b64}"
       width="{display_w}" height="{display_h}"
       onload="initCanvas()" style="display:block;">
  <canvas id="canvas" width="{display_w}" height="{display_h}"></canvas>
</div>
<script>
const ORIG_W = {orig_w}, ORIG_H = {orig_h};
const DISP_W = {display_w}, DISP_H = {display_h};
const scaleX = ORIG_W / DISP_W, scaleY = ORIG_H / DISP_H;
const CAM_ID = "{cam_id_calib}";
const API_BASE = "{API_BASE}";

let phase = 'road_left';
let roadPts = [], zonePts = [], savedZones = [];
let canvas, ctx;

function initCanvas() {{
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');
  canvas.addEventListener('click', handleClick);
  redraw();
}}

function handleClick(e) {{
  const rect = canvas.getBoundingClientRect();
  const dx = Math.round((e.clientX - rect.left) * (DISP_W / rect.width));
  const dy = Math.round((e.clientY - rect.top)  * (DISP_H / rect.height));
  const ox = Math.round(dx * scaleX), oy = Math.round(dy * scaleY);

  if (phase === 'road_left') {{
    roadPts = [[dx,dy,ox,oy]];
    phase = 'road_right';
    setStatus('Step 1: Now click the RIGHT edge of the road/lane');
  }} else if (phase === 'road_right') {{
    roadPts.push([dx,dy,ox,oy]);
    phase = 'zone';
    setStatus('Step 2: Click corner 1 of 4 for the no-parking zone polygon');
  }} else {{
    zonePts.push([dx,dy,ox,oy]);
    if (zonePts.length < 4) {{
      setStatus(`Step 2: Click corner ${{zonePts.length+1}} of 4`);
    }} else {{
      const rw = Math.abs(roadPts[1][2] - roadPts[0][2]);
      savedZones.push({{ road_width_px: rw, polygon: zonePts.map(p=>[p[2],p[3]]) }});
      zonePts = [];
      document.getElementById('btnSend').disabled = false;
      setStatus(`✅ Zone ${{savedZones.length}} done! Click 4 more corners for another zone, or Save.`);
    }}
  }}
  redraw();
}}

function redraw() {{
  if (!ctx) return;
  ctx.clearRect(0, 0, DISP_W, DISP_H);
  savedZones.forEach(z => {{
    const pts = z.polygon.map(p => [Math.round(p[0]/scaleX), Math.round(p[1]/scaleY)]);
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,220,80,0.25)'; ctx.fill();
    ctx.strokeStyle = '#00dc50'; ctx.lineWidth = 2; ctx.stroke();
    pts.forEach(p => dot(p[0], p[1], '#00dc50'));
  }});
  if (roadPts.length >= 1) dot(roadPts[0][0], roadPts[0][1], '#00dcff');
  if (roadPts.length === 2) {{
    ctx.beginPath(); ctx.moveTo(roadPts[0][0], roadPts[0][1]);
    ctx.lineTo(roadPts[1][0], roadPts[1][1]);
    ctx.strokeStyle = '#00dcff'; ctx.lineWidth = 3; ctx.stroke();
    dot(roadPts[1][0], roadPts[1][1], '#00dcff');
  }}
  zonePts.forEach((p, i) => {{
    dot(p[0], p[1], '#ff5050');
    if (i > 0) {{
      ctx.beginPath(); ctx.moveTo(zonePts[i-1][0], zonePts[i-1][1]);
      ctx.lineTo(p[0], p[1]);
      ctx.strokeStyle = '#ff5050'; ctx.lineWidth = 2; ctx.stroke();
    }}
  }});
}}

function dot(x, y, color) {{
  ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI*2);
  ctx.fillStyle = color; ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
}}

function setStatus(msg) {{ document.getElementById('status').textContent = msg; }}

function showToast(msg, ok) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = ok ? 'toast-ok' : 'toast-err';
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 4000);
}}

function undo() {{
  if (zonePts.length > 0) {{ zonePts.pop(); }}
  else if (savedZones.length > 0) {{
    const z = savedZones.pop();
    zonePts = z.polygon.map(p => [Math.round(p[0]/scaleX), Math.round(p[1]/scaleY), p[0], p[1]]);
    phase = 'zone';
    if (savedZones.length === 0) document.getElementById('btnSend').disabled = true;
  }} else if (roadPts.length > 0) {{
    roadPts.pop(); phase = roadPts.length === 0 ? 'road_left' : 'road_right';
  }}
  redraw();
}}

function reset() {{
  roadPts=[]; zonePts=[]; savedZones=[]; phase='road_left';
  document.getElementById('btnSend').disabled = true;
  setStatus('Step 1: Click the LEFT edge of the road/lane'); redraw();
}}

async function sendPoints() {{
  const btn = document.getElementById('btnSend');
  btn.disabled = true; btn.textContent = 'Saving...';
  const payload = {{
    cam_id: CAM_ID,
    road_pts: roadPts.map(p=>[p[2],p[3]]),
    saved_zones: savedZones
  }};
  try {{
    const resp = await fetch(API_BASE + '/api/save-calibration', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(payload)
    }});
    const data = await resp.json();
    if (data.status === 'saved') {{
      showToast(`✅ Saved ${{data.zones}} zone(s) for ${{data.cam}}! Reload the page to see the update.`, true);
      btn.textContent = '✅ Saved!';
    }} else {{
      showToast('❌ Error: ' + (data.msg || 'Unknown'), false);
      btn.disabled = false; btn.textContent = '💾 Save Calibration';
    }}
  }} catch(err) {{
    showToast('❌ Could not reach API: ' + err, false);
    btn.disabled = false; btn.textContent = '💾 Save Calibration';
  }}
}}
</script>
</body>
</html>"""


                components.html(canvas_html, height=display_h + 60, scrolling=False)


        # ── Always show saved calibration (OUTSIDE expander so it always renders) ──
        import json as _json
        if os.path.exists("data/calibration.json"):
            try:
                with open("data/calibration.json") as f:
                    existing = _json.load(f)
                st.markdown("#### 📋 Current Saved Calibration")
                st.json(existing)
            except Exception:
                pass


    calibration_tool()

    st.markdown("---")



    st.markdown("### 🎛️ Upload & Run AI Detection")
    uploaded_file = st.file_uploader("Upload a traffic video (.mp4) to run the AI detector", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        import os
        os.makedirs("data", exist_ok=True)
        video_path = "data/uploaded_video.mp4"
        
        # Only overwrite the file if it's a new upload to prevent Windows lock OSErrors
        file_id = getattr(uploaded_file, "file_id", f"{uploaded_file.name}_{uploaded_file.size}")
        if st.session_state.get("last_uploaded_vid") != file_id:
            try:
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state["last_uploaded_vid"] = file_id
            except OSError:
                pass # File is likely locked by the running detector, ignore


        st.success("Video uploaded! Calibrate your zones in the tool above, then start detection below.")

        detector_running = (
            "detector_proc" in st.session_state
            and st.session_state["detector_proc"] is not None
            and st.session_state["detector_proc"].poll() is None
        )

        bc2, bc3 = st.columns(2)
        with bc2:
            if not detector_running:
                if st.button("🚀 Start Live Detection", type="primary", use_container_width=True):
                    import subprocess
                    proc = subprocess.Popen(["python", "src/cctv_detector.py", "--source", video_path, "--no-show"])
                    st.session_state["detector_proc"] = proc
                    st.success("Detector started! Scroll down to see the Live Feed and Telemetry.")
                    st.rerun()
            else:
                st.success("🟢 Detector is running...")
        with bc3:
            if detector_running:
                if st.button("⏹️ Stop Stream", type="secondary", use_container_width=True):
                    import psutil
                    proc = st.session_state["detector_proc"]
                    try:
                        parent = psutil.Process(proc.pid)
                        for child in parent.children(recursive=True):
                            child.kill()
                        parent.kill()
                    except Exception:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    st.session_state["detector_proc"] = None
                    try:
                        import requests
                        requests.post(f"{API_BASE}/api/set-buzzer",
                                      json={"active": False, "zone_id": "", "level": "VACANT"},
                                      timeout=2.0)
                    except Exception:
                        pass
                    st.warning("Detector stopped. Hardware LEDs reset.")
                    st.rerun()


    st.markdown("---")

    st.markdown("### 📷 Live CCTV Feed")
    # MJPEG img tag is OUTSIDE any fragment — the browser holds ONE persistent
    # HTTP connection and the image never drops or flickers on re-render.
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Node 1: CCTV-CAM-01**")
        st.markdown(
            f"""
            <div style="border:2px solid #444;border-radius:10px;overflow:hidden;
                        background:#111;line-height:0;position:relative;min-height:250px;">
              <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:#888;">
                 <h3 style="margin:0;">🚫</h3>
                 <p style="margin:5px 0 0 0;">Camera Offline / Not Available</p>
              </div>
              <img id="feed-cam1"
                   src="{API_BASE}/api/video_feed?cam_id=CCTV-CAM-01"
                   style="width:100%;display:block;position:relative;z-index:10;"
                   onload="this.style.display='block'"
                   onerror="this.style.display='none'; let img=this; setTimeout(()=>{{ img.src='{API_BASE}/api/video_feed?cam_id=CCTV-CAM-01&t='+Date.now() }},2000)">
              <button onclick="let img = document.getElementById('feed-cam1'); if(img.style.display==='none'){{ alert('Wait for camera feed to connect.'); return; }} if(img.requestFullscreen){{ img.requestFullscreen().catch(err=>alert('Fullscreen error: '+err.message)); }} else if(img.webkitRequestFullscreen){{ img.webkitRequestFullscreen(); }}" 
                      style="position:absolute;top:10px;right:10px;z-index:20;background:rgba(0,0,0,0.5);border:none;border-radius:6px;color:#fff;cursor:pointer;padding:6px 10px;font-size:16px;box-shadow:0 0 5px rgba(0,0,0,0.5);"
                      title="Full Screen">⛶</button>
            </div>
            """, unsafe_allow_html=True
        )
    with v2:
        st.markdown("**Node 2: CCTV-CAM-02**")
        st.markdown(
            f"""
            <div style="border:2px solid #444;border-radius:10px;overflow:hidden;
                        background:#111;line-height:0;position:relative;min-height:250px;">
              <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:#888;">
                 <h3 style="margin:0;">🚫</h3>
                 <p style="margin:5px 0 0 0;">Camera Offline / Not Available</p>
              </div>
              <img id="feed-cam2"
                   src="{API_BASE}/api/video_feed?cam_id=CCTV-CAM-02"
                   style="width:100%;display:block;position:relative;z-index:10;"
                   onload="this.style.display='block'"
                   onerror="this.style.display='none'; let img=this; setTimeout(()=>{{ img.src='{API_BASE}/api/video_feed?cam_id=CCTV-CAM-02&t='+Date.now() }},2000)">
              <button onclick="let img = document.getElementById('feed-cam2'); if(img.style.display==='none'){{ alert('Wait for camera feed to connect.'); return; }} if(img.requestFullscreen){{ img.requestFullscreen().catch(err=>alert('Fullscreen error: '+err.message)); }} else if(img.webkitRequestFullscreen){{ img.webkitRequestFullscreen(); }}" 
                      style="position:absolute;top:10px;right:10px;z-index:20;background:rgba(0,0,0,0.5);border:none;border-radius:6px;color:#fff;cursor:pointer;padding:6px 10px;font-size:16px;box-shadow:0 0 5px rgba(0,0,0,0.5);"
                      title="Full Screen">⛶</button>
            </div>
            """, unsafe_allow_html=True
        )
    
    st.markdown("""
        <p style="font-size:0.78rem;color:#888;margin-top:4px;">
          🟢 Live &nbsp;·&nbsp; Distributed YOLOv8 AI Feed &nbsp;·&nbsp; MJPEG stream
        </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    @st.fragment(run_every=3)
    def cctv_live():
        # Auto-refresh UI if the detector finished naturally (e.g. video ended)
        if "detector_proc" in st.session_state and st.session_state["detector_proc"] is not None:
            if st.session_state["detector_proc"].poll() is not None:
                st.session_state["detector_proc"] = None
                st.rerun()

        try:
            stats = req.get(f"{API_BASE}/api/cctv-events/stats", timeout=2).json()
        except Exception:
            st.warning("Backend not reachable. Start `python src/backend_api.py`")
            return


        # ── KPIs — LIVE (90-second rolling window) ────────────────────────────
        all_time = stats.get("all_time_violations", 0)
        live_v   = stats.get("total_violations", 0)
        live_h   = stats.get("high_priority", 0)
        live_m   = stats.get("medium_priority", 0)
        live_l   = stats.get("low_priority", 0)
        live_a   = stats.get("active_vehicles", 0)

        st.markdown(
            "<p style='font-size:0.75rem;color:#888;margin-bottom:4px;'>"
            "🔴 <b>LIVE</b> — Rolling 90-second window. "
            f"All-time total violations: <b>{all_time}</b></p>",
            unsafe_allow_html=True
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🚨 Live Violations",  live_v,  delta="active now" if live_v > 0 else "clear",
                  delta_color="inverse" if live_v > 0 else "off")
        c2.metric("🔴 High Priority",    live_h,  delta="tow needed" if live_h > 0 else "none",
                  delta_color="inverse" if live_h > 0 else "off")
        c3.metric("🟠 Medium Priority",  live_m,  delta="dispatch" if live_m > 0 else "none",
                  delta_color="inverse" if live_m > 0 else "off")
        c4.metric("🟢 Low Priority",     live_l,  delta="warning" if live_l > 0 else "none",
                  delta_color="inverse" if live_l > 0 else "off")
        c5.metric("🚗 Active Vehicles",  live_a,  delta="in zones" if live_a > 0 else "clear",
                  delta_color="inverse" if live_a > 0 else "off")

        st.markdown("---")
        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("### 📋 Priority Guide")
            st.markdown("""
* 🔴 **HIGH**: Immediate tow requested
* 🟠 **MEDIUM**: Officer dispatched
* 🟢 **LOW**: Warning logged
            """)

            by_zone = stats.get("by_zone", [])
            if by_zone:
                import pandas as pd
                zone_df = pd.DataFrame(by_zone).set_index("zone_id")
                st.markdown("### Violations by Zone")
                st.bar_chart(zone_df["count"])

                # High-risk zone suggestions
                if len(zone_df) > 0:
                    top_zone = zone_df["count"].idxmax()
                    top_count = int(zone_df["count"].max())
                    st.markdown("---")
                    st.markdown("### 💡 Zone Suggestions")
                    st.markdown(f"""
<div class="risk-card-high">
  🔴 <b>{top_zone}</b> has {top_count} violations — highest risk zone.<br>
  <i>Recommend: Deploy marshal + activate ESP32 buzzer alert.</i>
</div>""", unsafe_allow_html=True)

        with col1:
            st.markdown("### 🔍 Live Detection Events")
            try:
                # Fetch more events to allow deduplication across multiple tracks
                events = req.get(f"{API_BASE}/api/cctv-events/recent?limit=200", timeout=2).json()
                if not events:
                    st.info("No CCTV events yet. Run: `python src/cctv_detector.py --no-show`")
                    return

                import pandas as pd
                df = pd.DataFrame(events)
                
                # Keep only the latest event for each track_id so the duration updates in a single row
                df = df.sort_values("timestamp", ascending=False).drop_duplicates(subset=["track_id"], keep="first")
                df = df.head(30)
                
                df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")

                def row_style(row):
                    colours = {
                        "HIGH":   "background-color:#ff4b4b33",
                        "MEDIUM": "background-color:#ffa50033",
                        "LOW":    "background-color:#00c80033",
                    }
                    return [colours.get(row.get("priority", ""), "")] * len(row)

                display_cols = ["time", "track_id", "vehicle_type", "zone_id",
                                "duration_sec", "cis", "priority"]
                available = [c for c in display_cols if c in df.columns]
                styled = df[available].style.apply(row_style, axis=1)
                st.dataframe(styled, width='stretch', height=400)
            except Exception as e:
                st.error(f"Error fetching events: {e}")

    cctv_live()
