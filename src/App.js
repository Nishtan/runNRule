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

// ── Geographic interpolation between two GPS pings ───────────────────────────
// Instead of using H3's gridPathCells (which finds the shortest H3-grid path,
// ignoring real-world geometry), we walk the actual line segment between the
// two lat/lng coordinates and sample every H3 cell the segment passes through.
//
// This is what Strava-style apps do: the GPS trace is the ground truth.
// If you walked a curve, the cells you actually passed through are captured —
// not the cells along the shortest H3 grid route.
//
// How it works:
//   1. Estimate how many H3 cells apart the two points are (gridDistance).
//   2. Sample (steps * 3) evenly-spaced points along the line segment.
//      Oversampling ensures we don't skip thin cells at oblique angles.
//   3. Convert each sample to an H3 cell and deduplicate consecutive repeats.
//   4. Return only the NEW cells (drop index 0 = fromCell, already in path).
//
// MAX_GAP_CELLS: if the two pings are more than this many H3 cells apart we
// treat it as a GPS dropout and do NOT connect them — same as Strava's
// "stationary gap" handling. Tweak as needed.
const MAX_GAP_CELLS = 40; // ~360 m at res-12, ~1 km at res-11

function bridgeTo(fromCell, toCell, fromLng, fromLat, toLng, toLat, res) {
  if (fromCell === toCell) return [];

  const h3 = window.h3;

  // Estimate grid distance to decide oversampling count and gap check
  let gridDist = MAX_GAP_CELLS; // default: assume large if cross-face throws
  try { gridDist = h3.gridDistance(fromCell, toCell); } catch { /* cross-face */ }

  // GPS dropout guard — don't bridge huge jumps (lifted finger, bad fix, etc.)
  if (gridDist > MAX_GAP_CELLS) return [];

  // Number of sample points: oversample by 3× to avoid skipping thin cells
  const steps = Math.max(gridDist * 3, 6);

  const cells = [];
  let prevCell = fromCell;

  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const lat = fromLat + (toLat - fromLat) * t;
    const lng = fromLng + (toLng - fromLng) * t;

    let cell;
    try { cell = h3.latLngToCell(lat, lng, res); }
    catch { continue; }

    if (cell !== prevCell) {
      cells.push(cell);
      prevCell = cell;
    }
  }

  return cells; // fromCell already in path; toCell is naturally last if reached
}

// ── Robust flood fill via BFS ─────────────────────────────────────────────────
function floodFillInterior(ringSet) {
  if (ringSet.size === 0) return [];
  const h3 = window.h3;
  const ringArr = [...ringSet];
  const pivot = ringArr[0];

  let maxHops = 0;
  for (const cell of ringArr) {
    try {
      const d = h3.gridDistance(pivot, cell);
      if (d > maxHops) maxHops = d;
    } catch { /* cross-face, skip */ }
  }
  const safeRadius = maxHops + 3;

  const boundingDisk = new Set(h3.gridDisk(pivot, safeRadius));
  for (const cell of ringArr) {
    if (!boundingDisk.has(cell)) {
      h3.gridDisk(cell, safeRadius).forEach((c) => boundingDisk.add(c));
    }
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
    if (!ringSet.has(seed) && boundingDisk.has(seed)) {
      exterior.add(seed);
      queue.push(seed);
    }
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
  const stateRef = useRef({
    path: [],        // flat, fully-bridged path — every consecutive pair is adjacent
    pathIndex: {},   // h3id → first index in path (O(1) loop-closure lookup)
    scoredCells: {},
    circuitCount: 0,
    res: 12,
    // ── Last known lat/lng (the "previous GPS ping") ──────────────────────
    // We store the real map coordinates of the last processed ping so that
    // bridgeTo can interpolate along the actual geographic segment instead of
    // asking H3 for its grid shortest-path.
    lastLng: null,
    lastLat: null,
  });
  const isDrawing = useRef(false);
  const mapReady = useRef(false);

  const [res, setRes] = useState(12);
  const [stats, setStats] = useState({ path: 0, circuits: 0, captured: 0, maxScore: 0 });
  const [info, setInfo] = useState(
    "Click and drag on the map to draw your path — close a loop to capture territory"
  );
  const [libsLoaded, setLibsLoaded] = useState(false);
  const [hoveredCell, setHoveredCell] = useState(null);

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
    mapRef.current.getSource("scored").setData({
      type: "FeatureCollection",
      features: buildScoredFeatures(stateRef.current.scoredCells),
    });
    mapRef.current.getSource("path").setData({
      type: "FeatureCollection",
      features: buildPathFeatures(stateRef.current.path),
    });
    updateStats();
  }, [updateStats]);

  const flashInterior = useCallback((cells) => {
    if (!mapReady.current || !mapRef.current) return;
    mapRef.current.getSource("flash").setData({
      type: "FeatureCollection",
      features: cells.map((h3id) => ({
        type: "Feature", properties: {},
        geometry: cellBoundaryToGeoJSON(h3id),
      })),
    });
    setTimeout(() => {
      if (mapReady.current && mapRef.current)
        mapRef.current.getSource("flash").setData({ type: "FeatureCollection", features: [] });
    }, 700);
  }, []);

  const closeCircuit = useCallback(
    (loopStartIdx, closingCell) => {
      const st = stateRef.current;

      // Ring = everything from loopStartIdx to current end (closingCell already appended)
      const loop = st.path.slice(loopStartIdx);
      const ringSet = new Set(loop);
      const interior = floodFillInterior(ringSet);

      interior.forEach((id) => { st.scoredCells[id] = (st.scoredCells[id] || 0) + 1; });
      loop.forEach((id)     => { st.scoredCells[id] = (st.scoredCells[id] || 0) + 1; });
      st.circuitCount++;

      // Reseed fresh from the closing cell only
      st.path = [closingCell];
      st.pathIndex = { [closingCell]: 0 };
      // lastLng/lastLat intentionally kept so next segment starts from the right coord

      flashInterior(interior);
      setInfo(`Circuit #${st.circuitCount} closed! ${interior.length} interior cells captured · reseeded ✓`);
      redraw();
    },
    [flashInterior, redraw]
  );

  // ── addCell: geographic interpolation ────────────────────────────────────
  // lng/lat are the REAL map coordinates of this ping.
  // We call bridgeTo with both the H3 cells AND the raw coordinates so it can
  // walk the actual geographic segment — not the H3 grid shortcut.
  const addCell = useCallback(
    (rawCell, lng, lat) => {
      if (!rawCell) return;
      const st = stateRef.current;

      // Bootstrap: very first cell
      if (st.path.length === 0) {
        st.path.push(rawCell);
        st.pathIndex[rawCell] = 0;
        st.lastLng = lng;
        st.lastLat = lat;
        redraw();
        return;
      }

      const tail = st.path[st.path.length - 1];
      if (tail === rawCell) {
        // Same H3 cell — still update coords so next bridge starts from here
        st.lastLng = lng;
        st.lastLat = lat;
        return;
      }

      // Use stored previous coords for geographic interpolation
      const fromLng = st.lastLng ?? lng;
      const fromLat = st.lastLat ?? lat;

      const bridge = bridgeTo(tail, rawCell, fromLng, fromLat, lng, lat, st.res);

      // Update last coords BEFORE processing bridge so reseed after closeCircuit
      // still has correct coordinates for the next segment
      st.lastLng = lng;
      st.lastLat = lat;

      for (const cell of bridge) {
        if (st.pathIndex[cell] !== undefined) {
          // Close the loop: append the closing cell, then trigger closure
          st.path.push(cell);
          closeCircuit(st.pathIndex[cell], cell);
          return; // path reseeded inside closeCircuit; stop here
        }
        st.pathIndex[cell] = st.path.length;
        st.path.push(cell);
      }

      redraw();
    },
    [closeCircuit, redraw]
  );

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
    setInfo("Click and drag on the map to draw your path — close a loop to capture territory");
  }, [updateStats]);

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
      });

      const canvas = map.getCanvas();

      // ── getH3WithCoords: returns { cell, lng, lat } for a pointer/touch event ──
      // We extract BOTH the H3 cell AND the raw map coordinates from every event
      // so addCell can do geographic interpolation between consecutive pings.
      const getH3WithCoords = (e) => {
        const rect = canvas.getBoundingClientRect();
        const pt = e.touches ? e.touches[0] : e;
        const { lng, lat } = map.unproject([pt.clientX - rect.left, pt.clientY - rect.top]);
        try {
          const cell = window.h3.latLngToCell(lat, lng, stateRef.current.res);
          return { cell, lng, lat };
        } catch {
          return { cell: null, lng, lat };
        }
      };

      canvas.addEventListener("mousedown",  (e) => {
        isDrawing.current = true;
        map.dragPan.disable();
        const { cell, lng, lat } = getH3WithCoords(e);
        if (cell) addCell(cell, lng, lat);
      });
      canvas.addEventListener("mousemove",  (e) => {
        const { cell, lng, lat } = getH3WithCoords(e);
        setHoveredCell(cell);
        if (isDrawing.current && cell) addCell(cell, lng, lat);
      });
      canvas.addEventListener("mouseup",    ()  => { isDrawing.current = false; map.dragPan.enable(); });
      canvas.addEventListener("mouseleave", ()  => { isDrawing.current = false; map.dragPan.enable(); setHoveredCell(null); });
      canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        isDrawing.current = true;
        map.dragPan.disable();
        const { cell, lng, lat } = getH3WithCoords(e);
        if (cell) addCell(cell, lng, lat);
      }, { passive: false });
      canvas.addEventListener("touchmove",  (e) => {
        e.preventDefault();
        if (!isDrawing.current) return;
        const { cell, lng, lat } = getH3WithCoords(e);
        if (cell) addCell(cell, lng, lat);
      }, { passive: false });
      canvas.addEventListener("touchend",   ()  => { isDrawing.current = false; map.dragPan.enable(); });
    };

    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (pos) => initializeMap([pos.coords.longitude, pos.coords.latitude]),
        ()    => initializeMap([72.8777, 19.076]),
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
      );
    } else {
      initializeMap([72.8777, 19.076]);
    }

    return () => {
      if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; mapReady.current = false; }
    };
  }, [libsLoaded, addCell, updateStats]);

  const handleResChange = useCallback((newRes) => {
    setRes(newRes);
    stateRef.current.res = newRes;
    if (mapRef.current) mapRef.current.flyTo({ zoom: newRes === 11 ? 15.5 : 17, duration: 700 });
    resetAll();
    setInfo(`Switched to Res ${newRes} (~${newRes === 11 ? "25m" : "9m"} cells) · Draw your path on the map`);
  }, [resetAll]);

  return (
    <div style={{ fontFamily: "'DM Mono','Courier New',monospace", display: "flex", flexDirection: "column", height: "100vh", background: "#0a0e14", color: "#e2eaf5" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, padding: "9px 16px", background: "#0f1520", borderBottom: "1px solid rgba(99,140,200,0.18)", flexShrink: 0, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, letterSpacing: 2, textTransform: "uppercase", color: "#22d3ee", fontWeight: 700, whiteSpace: "nowrap" }}>
          H3 · Territory
        </span>
        <div style={{ width: 1, height: 22, background: "rgba(99,140,200,0.18)", flexShrink: 0 }} />
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ fontSize: 10, color: "#6b82a0", letterSpacing: 1, textTransform: "uppercase" }}>Res</span>
          {[11, 12].map((r) => (
            <button key={r} onClick={() => handleResChange(r)} style={{ fontSize: 11, padding: "4px 12px", borderRadius: 5, border: res === r ? "1.5px solid #3b82f6" : "1px solid rgba(99,140,200,0.25)", background: res === r ? "#3b82f6" : "transparent", color: res === r ? "#fff" : "#6b82a0", cursor: "pointer", fontFamily: "inherit", fontWeight: res === r ? 700 : 400, transition: "all 0.15s" }}>
              {r} <span style={{ opacity: 0.6, fontWeight: 400 }}>~{r === 11 ? "25m" : "9m"}</span>
            </button>
          ))}
        </div>
        <div style={{ width: 1, height: 22, background: "rgba(99,140,200,0.18)", flexShrink: 0 }} />
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {[["Path", stats.path], ["Circuits", stats.circuits], ["Captured", stats.captured], ["Max score", stats.maxScore]].map(([label, val]) => (
            <div key={label} style={{ background: "#141c2b", border: "1px solid rgba(99,140,200,0.18)", borderRadius: 6, padding: "4px 10px", fontSize: 11, color: "#6b82a0", whiteSpace: "nowrap" }}>
              {label}: <strong style={{ color: "#e2eaf5" }}>{val}</strong>
            </div>
          ))}
        </div>
        <button onClick={resetAll} style={{ marginLeft: "auto", fontSize: 11, padding: "5px 14px", borderRadius: 6, border: "1px solid rgba(239,68,68,0.4)", background: "rgba(239,68,68,0.08)", color: "#f87171", cursor: "pointer", fontFamily: "inherit", whiteSpace: "nowrap" }}>
          ↺ Reset
        </button>
      </div>

      <div style={{ position: "relative", flex: 1 }}>
        {!libsLoaded && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#6b82a0", fontSize: 13, zIndex: 10 }}>
            Loading map…
          </div>
        )}
        <div ref={mapContainerRef} style={{ width: "100%", height: "100%" }} />

        {hoveredCell && (
          <div style={{ position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)", background: "rgba(10,14,20,0.92)", border: "1px solid rgba(34,211,238,0.4)", backdropFilter: "blur(8px)", borderRadius: 8, padding: "5px 14px", fontSize: 11, color: "#22d3ee", pointerEvents: "none", zIndex: 25, whiteSpace: "nowrap" }}>
            H3 res-{res}: <strong>{hoveredCell}</strong>
          </div>
        )}

        <div style={{ position: "absolute", bottom: 28, left: "50%", transform: "translateX(-50%)", background: "rgba(10,14,20,0.88)", border: "1px solid rgba(99,140,200,0.2)", backdropFilter: "blur(10px)", borderRadius: 10, padding: "8px 18px", fontSize: 11, color: "#22d3ee", pointerEvents: "none", zIndex: 20, maxWidth: "90vw", textAlign: "center" }}>
          {info}
        </div>

        <div style={{ position: "absolute", bottom: 74, right: 14, background: "rgba(10,14,20,0.88)", border: "1px solid rgba(99,140,200,0.2)", backdropFilter: "blur(10px)", borderRadius: 10, padding: "10px 14px", zIndex: 20, fontSize: 10, color: "#6b82a0" }}>
          {[["rgba(239,68,68,0.55)", "#ef4444", "Path start"], ["rgba(59,130,246,0.32)", "#3b82f6", "Active path"], ["rgba(251,191,36,0.7)", "#f59e0b", "Current cell"]].map(([bg, border, label]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
              <div style={{ width: 13, height: 13, borderRadius: 3, background: bg, border: `1.5px solid ${border}`, flexShrink: 0 }} />
              {label}
            </div>
          ))}
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