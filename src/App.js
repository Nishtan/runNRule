import { useState, useEffect, useRef, useCallback } from "react";

function loadScript(src) {
  return new Promise((res) => {
    if (document.querySelector(`script[src="${src}"]`)) return res();
    const s = document.createElement("script");
    s.src = src;
    s.onload = res;
    document.head.appendChild(s);
  });
}

function loadLink(href) {
  if (document.querySelector(`link[href="${href}"]`)) return;
  const l = document.createElement("link");
  l.rel = "stylesheet";
  l.href = href;
  document.head.appendChild(l);
}

const MAPBOX_TOKEN =
  "pk.eyJ1IjoibmlzaHRhbiIsImEiOiJja3ZkcGNmZWg0d25wMm5xd2RkcDBzeHVsIn0.irJll1qHLs4XBFONtsVYFA";

function scoreToFill(score) {
  const t = Math.min(score / 6, 1);
  const r = Math.round(110 + (5 - 110) * t);
  const g = Math.round(231 + (150 - 231) * t);
  const b = Math.round(183 + (105 - 183) * t);
  return `rgba(${r},${g},${b},${0.3 + 0.55 * t})`;
}

function scoreToStroke(score) {
  const t = Math.min(score / 6, 1);
  const g = Math.round(185 - 120 * t);
  return `rgba(16,${g},82,${0.7 + 0.3 * t})`;
}

function cellBoundaryToGeoJSON(h3index) {
  const boundary = window.h3.cellToBoundary(h3index);
  const coords = boundary.map(([lat, lng]) => [lng, lat]);
  coords.push(coords[0]);
  return { type: "Polygon", coordinates: [coords] };
}

const MAX_GAP_CELLS = 40;

function bridgeTo(fromCell, toCell, fromLng, fromLat, toLng, toLat, res) {
  if (fromCell === toCell) return [];
  const h3 = window.h3;
  let gridDist = MAX_GAP_CELLS;
  try { gridDist = h3.gridDistance(fromCell, toCell); } catch { /* cross-face */ }
  if (gridDist > MAX_GAP_CELLS) return [];
  const steps = Math.max(gridDist * 3, 6);
  const cells = [];
  let prevCell = fromCell;
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const lat = fromLat + (toLat - fromLat) * t;
    const lng = fromLng + (toLng - fromLng) * t;
    let cell;
    try { cell = h3.latLngToCell(lat, lng, res); } catch { continue; }
    if (cell !== prevCell) { cells.push(cell); prevCell = cell; }
  }
  return cells;
}

function floodFillInterior(ringSet) {
  if (ringSet.size === 0) return [];
  const h3 = window.h3;
  const ringArr = [...ringSet];
  const pivot = ringArr[0];
  let maxHops = 0;
  for (const cell of ringArr) {
    try { const d = h3.gridDistance(pivot, cell); if (d > maxHops) maxHops = d; } catch { /**/ }
  }
  const safeRadius = maxHops + 3;
  const boundingDisk = new Set(h3.gridDisk(pivot, safeRadius));
  for (const cell of ringArr) {
    if (!boundingDisk.has(cell)) h3.gridDisk(cell, safeRadius).forEach((c) => boundingDisk.add(c));
  }
  let seeds = [];
  try { seeds = h3.gridRingUnsafe(pivot, safeRadius); } catch { /**/ }
  if (!seeds || seeds.length === 0) {
    const inner = new Set(h3.gridDisk(pivot, safeRadius - 1));
    seeds = [...boundingDisk].filter((c) => !inner.has(c));
  }
  const exterior = new Set();
  const queue = [];
  for (const seed of seeds) {
    if (!ringSet.has(seed) && boundingDisk.has(seed)) { exterior.add(seed); queue.push(seed); }
  }
  while (queue.length > 0) {
    const cell = queue.shift();
    for (const nb of h3.gridDisk(cell, 1)) {
      if (nb === cell || exterior.has(nb) || ringSet.has(nb) || !boundingDisk.has(nb)) continue;
      exterior.add(nb);
      queue.push(nb);
    }
  }
  return [...boundingDisk].filter((c) => !ringSet.has(c) && !exterior.has(c));
}

function buildScoredFeatures(scoredCells) {
  return Object.entries(scoredCells).map(([h3id, score]) => ({
    type: "Feature",
    properties: { fillColor: scoreToFill(score), strokeColor: scoreToStroke(score) },
    geometry: cellBoundaryToGeoJSON(h3id),
  }));
}

function buildPathFeatures(path) {
  return path.map((h3id, i) => {
    const isStart = i === 0, isLast = i === path.length - 1;
    return {
      type: "Feature",
      properties: {
        fillColor: isStart ? "rgba(239,68,68,0.55)" : isLast ? "rgba(251,191,36,0.7)" : "rgba(59,130,246,0.32)",
        strokeColor: isStart ? "#ef4444" : isLast ? "#f59e0b" : "#3b82f6",
        strokeWidth: isStart || isLast ? 2.5 : 1.2,
      },
      geometry: cellBoundaryToGeoJSON(h3id),
    };
  });
}

const SCORE_COLORS = [1, 2, 3, 4, 5, 6];

export default function App() {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markerRef = useRef(null);          // Mapbox Marker for GPS dot
  const watchIdRef = useRef(null);         // geolocation watchPosition id
  const stateRef = useRef({
    path: [],
    pathIndex: {},
    scoredCells: {},
    circuitCount: 0,
    res: 12,
    lastLng: null,
    lastLat: null,
  });
  const mapReady = useRef(false);
  const isRunningRef = useRef(false);      // mirrors isRunning state, avoids stale closure in GPS cb

  const [res, setRes] = useState(12);
  const [stats, setStats] = useState({ path: 0, circuits: 0, captured: 0, maxScore: 0 });
  const [info, setInfo] = useState("Allow location access — your position will appear on the map");
  const [libsLoaded, setLibsLoaded] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [gpsReady, setGpsReady] = useState(false);   // true once we have ≥1 GPS fix
  const [gpsAccuracy, setGpsAccuracy] = useState(null);

  useEffect(() => {
    loadLink("https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css");
    Promise.all([
      loadScript("https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"),
      loadScript("https://unpkg.com/h3-js@4.1.0/dist/h3-js.umd.js"),
    ]).then(() => setLibsLoaded(true));
  }, []);

  const updateStats = useCallback(() => {
    const { path, scoredCells, circuitCount } = stateRef.current;
    const scores = Object.values(scoredCells);
    setStats({
      path: path.length,
      circuits: circuitCount,
      captured: Object.keys(scoredCells).length,
      maxScore: scores.length ? Math.max(...scores) : 0,
    });
  }, []);

  const redraw = useCallback(() => {
    if (!mapReady.current || !mapRef.current) return;
    mapRef.current.getSource("scored").setData({ type: "FeatureCollection", features: buildScoredFeatures(stateRef.current.scoredCells) });
    mapRef.current.getSource("path").setData({ type: "FeatureCollection", features: buildPathFeatures(stateRef.current.path) });
    updateStats();
  }, [updateStats]);

  const flashInterior = useCallback((cells) => {
    if (!mapReady.current || !mapRef.current) return;
    mapRef.current.getSource("flash").setData({
      type: "FeatureCollection",
      features: cells.map((h3id) => ({ type: "Feature", properties: {}, geometry: cellBoundaryToGeoJSON(h3id) })),
    });
    setTimeout(() => {
      if (mapReady.current && mapRef.current)
        mapRef.current.getSource("flash").setData({ type: "FeatureCollection", features: [] });
    }, 700);
  }, []);

  const closeCircuit = useCallback((loopStartIdx, closingCell) => {
    const st = stateRef.current;
    const loop = st.path.slice(loopStartIdx);
    const ringSet = new Set(loop);
    const interior = floodFillInterior(ringSet);
    interior.forEach((id) => { st.scoredCells[id] = (st.scoredCells[id] || 0) + 1; });
    loop.forEach((id)     => { st.scoredCells[id] = (st.scoredCells[id] || 0) + 1; });
    st.circuitCount++;
    st.path = [closingCell];
    st.pathIndex = { [closingCell]: 0 };
    flashInterior(interior);
    setInfo(`Circuit #${st.circuitCount} closed! ${interior.length} interior cells captured · reseeded ✓`);
    redraw();
  }, [flashInterior, redraw]);

  const addCell = useCallback((rawCell, lng, lat) => {
    if (!rawCell) return;
    const st = stateRef.current;
    if (st.path.length === 0) {
      st.path.push(rawCell);
      st.pathIndex[rawCell] = 0;
      st.lastLng = lng;
      st.lastLat = lat;
      redraw();
      return;
    }
    const tail = st.path[st.path.length - 1];
    if (tail === rawCell) { st.lastLng = lng; st.lastLat = lat; return; }
    const fromLng = st.lastLng ?? lng;
    const fromLat = st.lastLat ?? lat;
    const bridge = bridgeTo(tail, rawCell, fromLng, fromLat, lng, lat, st.res);
    st.lastLng = lng;
    st.lastLat = lat;
    for (const cell of bridge) {
      if (st.pathIndex[cell] !== undefined) {
        st.path.push(cell);
        closeCircuit(st.pathIndex[cell], cell);
        return;
      }
      st.pathIndex[cell] = st.path.length;
      st.path.push(cell);
    }
    redraw();
  }, [closeCircuit, redraw]);

  const resetAll = useCallback(() => {
    stateRef.current = {
      path: [],
      pathIndex: {},
      scoredCells: {},
      circuitCount: 0,
      res: stateRef.current.res,
      lastLng: null,
      lastLat: null,
    };
    if (mapReady.current && mapRef.current) {
      ["scored", "path", "flash"].forEach((src) =>
        mapRef.current.getSource(src).setData({ type: "FeatureCollection", features: [] })
      );
    }
    updateStats();
    setInfo("Press Start Run to begin capturing territory with your GPS");
  }, [updateStats]);

  // ── GPS callback — always update marker; only capture area when running ───
  const handleGPSPosition = useCallback((pos) => {
    const { longitude: lng, latitude: lat, accuracy } = pos.coords;
    setGpsAccuracy(Math.round(accuracy));
    setGpsReady(true);

    // Move the marker
    if (markerRef.current) {
      markerRef.current.setLngLat([lng, lat]);
    }

    // Center map on first fix only (or if very first load)
    if (mapRef.current && !mapRef.current._hasCenteredOnUser) {
      mapRef.current.flyTo({ center: [lng, lat], zoom: 17, duration: 1200 });
      mapRef.current._hasCenteredOnUser = true;
    }

    // Area capture only while running
    if (!isRunningRef.current) return;

    const cell = (() => {
      try { return window.h3.latLngToCell(lat, lng, stateRef.current.res); }
      catch { return null; }
    })();
    if (cell) addCell(cell, lng, lat);
  }, [addCell]);

  const handleGPSError = useCallback((err) => {
    console.warn("GPS error:", err);
    setInfo("GPS error: " + err.message);
  }, []);

  // Start / stop run
  const toggleRun = useCallback(() => {
    setIsRunning((prev) => {
      const next = !prev;
      isRunningRef.current = next;
      if (next) {
        // Reset path state but keep scored cells so territory persists across runs
        const st = stateRef.current;
        st.path = [];
        st.pathIndex = {};
        st.lastLng = null;
        st.lastLat = null;
        if (mapReady.current && mapRef.current) {
          mapRef.current.getSource("path").setData({ type: "FeatureCollection", features: [] });
        }
        setInfo("Run started — walk around to capture territory · close a loop to score!");
      } else {
        setInfo(`Run stopped · ${stateRef.current.circuitCount} circuit(s) completed · ${Object.keys(stateRef.current.scoredCells).length} cells captured`);
      }
      return next;
    });
  }, []);

  // Map init
  useEffect(() => {
    if (!libsLoaded || !mapContainerRef.current || mapRef.current) return;

    const initializeMap = (center) => {
      window.mapboxgl.accessToken = MAPBOX_TOKEN;
      const map = new window.mapboxgl.Map({
        container: mapContainerRef.current,
        style: "mapbox://styles/mapbox/dark-v11",
        center,
        zoom: 17,
        minZoom: 13,
        maxZoom: 21,
        antialias: true,
      });
      map.addControl(new window.mapboxgl.NavigationControl({ showCompass: false }), "top-right");
      mapRef.current = map;

      map.on("load", () => {
        ["scored", "path", "flash"].forEach((id) =>
          map.addSource(id, { type: "geojson", data: { type: "FeatureCollection", features: [] } })
        );
        map.addLayer({ id: "scored-fill",   type: "fill", source: "scored", paint: { "fill-color": ["get", "fillColor"], "fill-opacity": 1 } });
        map.addLayer({ id: "scored-stroke", type: "line", source: "scored", paint: { "line-color": ["get", "strokeColor"], "line-width": 1.5 } });
        map.addLayer({ id: "path-fill",     type: "fill", source: "path",   paint: { "fill-color": ["get", "fillColor"], "fill-opacity": 1 } });
        map.addLayer({ id: "path-stroke",   type: "line", source: "path",   paint: { "line-color": ["get", "strokeColor"], "line-width": ["get", "strokeWidth"] } });
        map.addLayer({ id: "flash-fill",    type: "fill", source: "flash",  paint: { "fill-color": "rgba(255,255,120,0.6)", "fill-opacity": 1 } });
        mapReady.current = true;
        updateStats();

        // ── GPS Marker (pulsing dot) ──────────────────────────────────────
        const el = document.createElement("div");
        el.style.cssText = `
          width: 20px; height: 20px; border-radius: 50%;
          background: rgba(56,189,248,0.9);
          border: 2.5px solid #fff;
          box-shadow: 0 0 0 0 rgba(56,189,248,0.6);
          animation: gpsPulse 2s infinite;
          position: relative; cursor: default;
        `;
        // inject keyframes once
        if (!document.getElementById("gpsPulseStyle")) {
          const style = document.createElement("style");
          style.id = "gpsPulseStyle";
          style.textContent = `
            @keyframes gpsPulse {
              0%   { box-shadow: 0 0 0 0   rgba(56,189,248,0.55); }
              70%  { box-shadow: 0 0 0 14px rgba(56,189,248,0);   }
              100% { box-shadow: 0 0 0 0   rgba(56,189,248,0);    }
            }
            @keyframes gpsPulseRun {
              0%   { box-shadow: 0 0 0 0   rgba(74,222,128,0.7);  }
              70%  { box-shadow: 0 0 0 18px rgba(74,222,128,0);   }
              100% { box-shadow: 0 0 0 0   rgba(74,222,128,0);    }
            }
          `;
          document.head.appendChild(style);
        }

        const marker = new window.mapboxgl.Marker({ element: el, anchor: "center" })
          .setLngLat(center)
          .addTo(map);
        markerRef.current = marker;
        markerRef.current._el = el; // keep ref to change style on run toggle
      });
    };

    // Start continuous GPS watch from the start
    if ("geolocation" in navigator) {
      const id = navigator.geolocation.watchPosition(
        handleGPSPosition,
        handleGPSError,
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
      );
      watchIdRef.current = id;

      // Initial center — grab a quick one-shot first then watch takes over
      navigator.geolocation.getCurrentPosition(
        (pos) => initializeMap([pos.coords.longitude, pos.coords.latitude]),
        ()    => initializeMap([72.8777, 19.076]),
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
      );
    } else {
      initializeMap([72.8777, 19.076]);
    }

    return () => {
      if (watchIdRef.current != null) navigator.geolocation.clearWatch(watchIdRef.current);
      if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; mapReady.current = false; }
    };
  }, [libsLoaded, handleGPSPosition, handleGPSError, updateStats]);

  // Update GPS marker color when run state changes
  useEffect(() => {
    if (!markerRef.current?._el) return;
    const el = markerRef.current._el;
    if (isRunning) {
      el.style.background = "rgba(74,222,128,0.95)";
      el.style.animation = "gpsPulseRun 1.4s infinite";
    } else {
      el.style.background = "rgba(56,189,248,0.9)";
      el.style.animation = "gpsPulse 2s infinite";
    }
  }, [isRunning]);

  const handleResChange = useCallback((newRes) => {
    setRes(newRes);
    stateRef.current.res = newRes;
    if (mapRef.current) mapRef.current.flyTo({ zoom: newRes === 11 ? 15.5 : 17, duration: 700 });
    resetAll();
    setInfo(`Switched to Res ${newRes} (~${newRes === 11 ? "25m" : "9m"} cells) · Press Start Run to capture`);
  }, [resetAll]);

  // Re-center map on user
  const reCenterOnUser = useCallback(() => {
    if (!mapRef.current || !markerRef.current) return;
    const lngLat = markerRef.current.getLngLat();
    mapRef.current.flyTo({ center: [lngLat.lng, lngLat.lat], zoom: 17, duration: 800 });
  }, []);

  return (
    <div style={{ fontFamily: "'DM Mono','Courier New',monospace", display: "flex", flexDirection: "column", height: "100vh", background: "#0a0e14", color: "#e2eaf5" }}>
      {/* ── Header ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "9px 16px", background: "#0f1520", borderBottom: "1px solid rgba(99,140,200,0.18)", flexShrink: 0, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, letterSpacing: 2, textTransform: "uppercase", color: "#22d3ee", fontWeight: 700, whiteSpace: "nowrap" }}>
          H3 · Territory
        </span>
        <div style={{ width: 1, height: 22, background: "rgba(99,140,200,0.18)", flexShrink: 0 }} />

        {/* Res selector */}
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ fontSize: 10, color: "#6b82a0", letterSpacing: 1, textTransform: "uppercase" }}>Res</span>
          {[11, 12].map((r) => (
            <button key={r} onClick={() => handleResChange(r)} disabled={isRunning} style={{ fontSize: 11, padding: "4px 12px", borderRadius: 5, border: res === r ? "1.5px solid #3b82f6" : "1px solid rgba(99,140,200,0.25)", background: res === r ? "#3b82f6" : "transparent", color: res === r ? "#fff" : "#6b82a0", cursor: isRunning ? "not-allowed" : "pointer", fontFamily: "inherit", fontWeight: res === r ? 700 : 400, opacity: isRunning ? 0.5 : 1, transition: "all 0.15s" }}>
              {r} <span style={{ opacity: 0.6, fontWeight: 400 }}>~{r === 11 ? "25m" : "9m"}</span>
            </button>
          ))}
        </div>

        <div style={{ width: 1, height: 22, background: "rgba(99,140,200,0.18)", flexShrink: 0 }} />

        {/* Stats */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {[["Path", stats.path], ["Circuits", stats.circuits], ["Captured", stats.captured], ["Max score", stats.maxScore]].map(([label, val]) => (
            <div key={label} style={{ background: "#141c2b", border: "1px solid rgba(99,140,200,0.18)", borderRadius: 6, padding: "4px 10px", fontSize: 11, color: "#6b82a0", whiteSpace: "nowrap" }}>
              {label}: <strong style={{ color: "#e2eaf5" }}>{val}</strong>
            </div>
          ))}
          {/* GPS accuracy pill */}
          <div style={{ background: "#141c2b", border: `1px solid ${gpsReady ? "rgba(74,222,128,0.35)" : "rgba(251,191,36,0.35)"}`, borderRadius: 6, padding: "4px 10px", fontSize: 11, color: gpsReady ? "#4ade80" : "#fbbf24", whiteSpace: "nowrap", display: "flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: gpsReady ? "#4ade80" : "#fbbf24", display: "inline-block", flexShrink: 0 }} />
            {gpsReady ? `GPS ±${gpsAccuracy}m` : "GPS…"}
          </div>
        </div>

        {/* Start / Stop Run button */}
        <button
          onClick={toggleRun}
          disabled={!gpsReady}
          style={{
            fontSize: 12, padding: "6px 18px", borderRadius: 7, fontFamily: "inherit", fontWeight: 700, cursor: gpsReady ? "pointer" : "not-allowed", transition: "all 0.2s", whiteSpace: "nowrap", letterSpacing: 0.5,
            border: isRunning ? "1.5px solid rgba(248,113,113,0.6)" : "1.5px solid rgba(74,222,128,0.55)",
            background: isRunning ? "rgba(239,68,68,0.15)" : "rgba(74,222,128,0.12)",
            color: isRunning ? "#f87171" : "#4ade80",
            boxShadow: isRunning ? "0 0 12px rgba(239,68,68,0.2)" : "0 0 12px rgba(74,222,128,0.15)",
            opacity: gpsReady ? 1 : 0.45,
          }}
        >
          {isRunning ? "⏹ Stop Run" : "▶ Start Run"}
        </button>

        {/* Re-center button */}
        <button onClick={reCenterOnUser} disabled={!gpsReady} title="Re-center on my location" style={{ fontSize: 13, padding: "5px 10px", borderRadius: 6, border: "1px solid rgba(99,140,200,0.25)", background: "transparent", color: "#6b82a0", cursor: gpsReady ? "pointer" : "not-allowed", fontFamily: "inherit", opacity: gpsReady ? 1 : 0.4 }}>
          ◎
        </button>

        {/* Reset */}
        <button onClick={resetAll} disabled={isRunning} style={{ fontSize: 11, padding: "5px 14px", borderRadius: 6, border: "1px solid rgba(239,68,68,0.4)", background: "rgba(239,68,68,0.08)", color: isRunning ? "#6b82a0" : "#f87171", cursor: isRunning ? "not-allowed" : "pointer", fontFamily: "inherit", whiteSpace: "nowrap", opacity: isRunning ? 0.45 : 1 }}>
          ↺ Reset
        </button>
      </div>

      {/* ── Map ── */}
      <div style={{ position: "relative", flex: 1 }}>
        {!libsLoaded && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#6b82a0", fontSize: 13, zIndex: 10 }}>
            Loading map…
          </div>
        )}
        <div ref={mapContainerRef} style={{ width: "100%", height: "100%" }} />

        {/* Running indicator banner */}
        {isRunning && (
          <div style={{ position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)", background: "rgba(10,14,20,0.92)", border: "1px solid rgba(74,222,128,0.5)", backdropFilter: "blur(8px)", borderRadius: 8, padding: "5px 16px", fontSize: 11, color: "#4ade80", pointerEvents: "none", zIndex: 25, whiteSpace: "nowrap", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#4ade80", display: "inline-block", animation: "gpsPulseRun 1.4s infinite" }} />
            Tracking · walk to capture territory
          </div>
        )}

        {/* Info bar */}
        <div style={{ position: "absolute", bottom: 28, left: "50%", transform: "translateX(-50%)", background: "rgba(10,14,20,0.88)", border: "1px solid rgba(99,140,200,0.2)", backdropFilter: "blur(10px)", borderRadius: 10, padding: "8px 18px", fontSize: 11, color: "#22d3ee", pointerEvents: "none", zIndex: 20, maxWidth: "90vw", textAlign: "center" }}>
          {info}
        </div>

        {/* Legend */}
        <div style={{ position: "absolute", bottom: 74, right: 14, background: "rgba(10,14,20,0.88)", border: "1px solid rgba(99,140,200,0.2)", backdropFilter: "blur(10px)", borderRadius: 10, padding: "10px 14px", zIndex: 20, fontSize: 10, color: "#6b82a0" }}>
          {[["rgba(239,68,68,0.55)", "#ef4444", "Path start"], ["rgba(59,130,246,0.32)", "#3b82f6", "Active path"], ["rgba(251,191,36,0.7)", "#f59e0b", "Current cell"]].map(([bg, border, label]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
              <div style={{ width: 13, height: 13, borderRadius: 3, background: bg, border: `1.5px solid ${border}`, flexShrink: 0 }} />
              {label}
            </div>
          ))}
          {/* GPS marker legend row */}
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
            <div style={{ width: 13, height: 13, borderRadius: "50%", background: isRunning ? "rgba(74,222,128,0.9)" : "rgba(56,189,248,0.9)", border: "2px solid #fff", flexShrink: 0 }} />
            {isRunning ? "You (running)" : "You (idle)"}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 4 }}>
            {SCORE_COLORS.map((s) => (
              <div key={s} style={{ width: 13, height: 13, borderRadius: 3, background: scoreToFill(s), border: `1px solid ${scoreToStroke(s)}` }} title={`Score ${s}`} />
            ))}
            <span style={{ marginLeft: 4 }}>score ×1→6</span>
          </div>
        </div>
      </div>
    </div>
  );
}