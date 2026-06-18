import streamlit as st
import pandas as pd
import os
import sqlite3
import requests
import pydeck as pdk

st.set_page_config(page_title="ParkIQ Dashboard", layout="wide", page_icon="🚦")

# ── CSS overrides for a cleaner look ──────────────────────────────────────────
st.markdown("""
<style>
    .risk-card-high   { background:#ff4b4b22; border-left:4px solid #ff4b4b; padding:10px 14px; border-radius:6px; margin-bottom:8px; }
    .risk-card-medium { background:#ffa50022; border-left:4px solid #ffa500; padding:10px 14px; border-radius:6px; margin-bottom:8px; }
    .risk-card-low    { background:#00c80022; border-left:4px solid #00c800; padding:10px 14px; border-radius:6px; margin-bottom:8px; }
    .suggest-box      { background:#1e3a5f22; border-left:4px solid #4a9eff; padding:10px 14px; border-radius:6px; margin-bottom:6px; }
    .stat-label       { font-size:0.78rem; color:#888; text-transform:uppercase; letter-spacing:0.05em; }
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "🧠 AI Predictive Impact Map",
    "⚠️ High-Risk Area Analysis",
    "📡 IoT Sensor Monitor",
    "🎥 City-Wide CCTV Network",
])

try:
    if page == "🎥 City-Wide CCTV Network":
        requests.post("http://127.0.0.1:8000/api/set-mode", json={"mode": "CCTV"}, timeout=0.5)
    else:
        requests.post("http://127.0.0.1:8000/api/set-mode", json={"mode": "IOT"}, timeout=0.5)
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
            st.toast(f"🚨 **{cam_id} LIVE ALERT:** Critical gridlock detected at {loc}. CIS: {cis:.1f}. Tow dispatch requested.", icon='🚨')

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
        conn.close()

        k1, k2, k3 = st.columns(3)
        k1.metric("Total IoT Events",   f"{int(total_events):,}")
        k2.metric("Violations Detected",f"{int(violations):,}")
        k3.metric("Live Nodes",         "1", delta="ESP32-NODE-01")

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
    API_BASE = "http://127.0.0.1:8000"

    st.title("🎥 City-Wide CCTV Network — Live Node Dashboard")
    st.caption("Monitoring real-time parking detections across 200,000 simulated city-wide computer vision nodes")

    # ── Auto-switch ESP32 to CCTV mode silently on page load ──────────────────
    try:
        req.post(f"{API_BASE}/api/set-mode", json={"mode": "CCTV"}, timeout=1)
    except Exception:
        pass  # Backend offline — no crash

    # Fetch current state for the status badge
    try:
        state = req.get(f"{API_BASE}/api/device-state", timeout=1).json()
        current_mode = state.get("mode", "CCTV")
        alert_level  = state.get("alert_level", "VACANT")
    except Exception:
        current_mode = "CCTV"
        alert_level  = "VACANT"

    level_colors = {"CRITICAL": "#ff4b4b", "HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "VACANT": "#00c800", "LOW": "#00c800"}
    level_color  = level_colors.get(alert_level, "#888888")

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;padding:10px 18px;
                border-radius:10px;background:#1a1a2e;margin-bottom:12px;border:1px solid #333;">
        <span style="font-size:1rem;font-weight:700;color:#fff;">🖥️ ESP32 Hardware Node</span>
        <span style="background:#00c8ff22;color:#00c8ff;border:1px solid #00c8ff;
                    padding:3px 12px;border-radius:20px;font-size:0.82rem;font-weight:700;">
            📷 MODE: CCTV — Cameras Active
        </span>
        <span style="background:{level_color}22;color:{level_color};border:1px solid {level_color};
                    padding:3px 12px;border-radius:20px;font-size:0.82rem;font-weight:700;">
            ALERT: {alert_level}
        </span>
        <span style="color:#888;font-size:0.78rem;">HC-SR04 sensor disabled</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📷 Live CCTV Feed")
    # MJPEG img tag is OUTSIDE any fragment — the browser holds ONE persistent
    # HTTP connection and the image never drops or flickers on re-render.
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Node 1: CCTV-CAM-01**")
        st.markdown(
            """
            <div style="border:2px solid #444;border-radius:10px;overflow:hidden;
                        background:#000;line-height:0;">
              <img id="feed-cam1"
                   src="http://127.0.0.1:8000/api/video_feed?cam_id=CCTV-CAM-01"
                   style="width:100%;display:block;"
                   onerror="setTimeout(()=>{ this.src='http://127.0.0.1:8000/api/video_feed?cam_id=CCTV-CAM-01&t='+Date.now() },2000)">
            </div>
            """, unsafe_allow_html=True
        )
    with v2:
        st.markdown("**Node 2: CCTV-CAM-02**")
        st.markdown(
            """
            <div style="border:2px solid #444;border-radius:10px;overflow:hidden;
                        background:#000;line-height:0;">
              <img id="feed-cam2"
                   src="http://127.0.0.1:8000/api/video_feed?cam_id=CCTV-CAM-02"
                   style="width:100%;display:block;"
                   onerror="setTimeout(()=>{ this.src='http://127.0.0.1:8000/api/video_feed?cam_id=CCTV-CAM-02&t='+Date.now() },2000)">
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
        try:
            stats = req.get(f"{API_BASE}/api/cctv-events/stats", timeout=2).json()
        except Exception:
            st.warning("Backend not reachable. Start `python src/backend_api.py`")
            return

        # ── KPIs ──────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚨 Total Violations",  stats.get("total_violations", 0))
        c2.metric("🔴 High Priority",     stats.get("high_priority", 0))
        c3.metric("🚗 Active Vehicles",   stats.get("active_vehicles", 0))
        medium = stats.get("total_violations", 0) - stats.get("high_priority", 0)
        c4.metric("🟠 Medium Priority",   max(medium, 0))

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
                events = req.get(f"{API_BASE}/api/cctv-events/recent?limit=30", timeout=2).json()
                if not events:
                    st.info("No CCTV events yet. Run: `python src/cctv_detector.py --no-show`")
                    return

                import pandas as pd
                df = pd.DataFrame(events)
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

    st.markdown("---")
    st.markdown("### 🚀 How to Start the Headless Detector")
    st.code("""# Run entirely in the background (no separate popup window!)
python src/cctv_detector.py --no-show

# RTSP Camera stream (production)
python src/cctv_detector.py --source rtsp://192.168.1.100:554/stream""", language="bash")
