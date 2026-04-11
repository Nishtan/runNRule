import { useState, useEffect, useRef, useCallback } from "react";

// ─── External lib loaders ────────────────────────────────────────────────────

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

// ─── Constants ───────────────────────────────────────────────────────────────

const MAPBOX_TOKEN =
  "pk.eyJ1IjoibmlzaHRhbiIsImEiOiJja3ZkcGNmZWg0d25wMm5xd2RkcDBzeHVsIn0.irJll1qHLs4XBFONtsVYFA";

const H3_RES = 12;
const BACKTRACK_TOL = 5;
const MAX_GAP_CELLS = 40;

const Q_ACCEL_VAR = 0.5;
const MAX_DT_S = 5;

// ← ZUPT: stillness detection thresholds
const ZUPT_WINDOW_MS = 300;   // stillness must persist this long
const ZUPT_ACCEL_TOL = 0.8;   // m/s² deviation from 9.8 allowed (gravity-included path)
const ZUPT_GYRO_TOL  = 0.05;  // rad/s max rotation allowed

// ─── 4×4 Matrix math ────────────────────────────────────────────────────────

function mat4_zero() { return new Float64Array(16); }

function mat4_identity() {
  const m = mat4_zero();
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4_mul(A, B) {
  const C = mat4_zero();
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      for (let k = 0; k < 4; k++)
        C[r + c * 4] += A[r + k * 4] * B[k + c * 4];
  return C;
}

function mat4_T(A) {
  const B = mat4_zero();
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      B[r + c * 4] = A[c + r * 4];
  return B;
}

function mat4_add(A, B) {
  const C = new Float64Array(16);
  for (let i = 0; i < 16; i++) C[i] = A[i] + B[i];
  return C;
}

function mat4_sub(A, B) {
  const C = new Float64Array(16);
  for (let i = 0; i < 16; i++) C[i] = A[i] - B[i];
  return C;
}

function mat2_inv(a, b, c, d) {
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) return [1, 0, 0, 1];
  return [d / det, -b / det, -c / det, a / det];
}

function vec2_sub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }

// ─── Coord helpers ───────────────────────────────────────────────────────────

function toMetric(lat, lng, originLat, originLng) {
  const R = 6_371_000;
  const dLat = ((lat - originLat) * Math.PI) / 180;
  const dLng = ((lng - originLng) * Math.PI) / 180;
  const x = dLng * R * Math.cos((originLat * Math.PI) / 180);
  const y = dLat * R;
  return [x, y];
}

function fromMetric(x, y, originLat, originLng) {
  const R = 6_371_000;
  const dLat = (y / R) * (180 / Math.PI);
  const dLng = (x / (R * Math.cos((originLat * Math.PI) / 180))) * (180 / Math.PI);
  return { lat: originLat + dLat, lng: originLng + dLng };
}

// ─── Kalman filter ───────────────────────────────────────────────────────────

class KalmanGPS {
  constructor(lat, lng, accuracy) {
    this.originLat = lat;
    this.originLng = lng;
    this.state = new Float64Array(4);
    const s2 = accuracy * accuracy;
    this.P = mat4_zero();
    this.P[0] = s2;
    this.P[5] = s2;
    this.P[10] = s2 * 10;
    this.P[15] = s2 * 10;
  }

  predict(dt) {
    const dtClamped = Math.min(dt, MAX_DT_S);
    const F = mat4_identity();
    F[0 + 2 * 4] = dtClamped;
    F[1 + 3 * 4] = dtClamped;
    const [px, py, vx, vy] = this.state;
    this.state[0] = px + vx * dtClamped;
    this.state[1] = py + vy * dtClamped;
    const FP = mat4_mul(F, this.P);
    const FPFt = mat4_mul(FP, mat4_T(F));
    const Q = this._buildQ(dtClamped);
    this.P = mat4_add(FPFt, Q);
    return this._stateToLatLng();
  }

  update(lat, lng, accuracy) {
    const [mx, my] = toMetric(lat, lng, this.originLat, this.originLng);
    const innov = vec2_sub([mx, my], [this.state[0], this.state[1]]);
    const s2 = accuracy * accuracy;
    const s00 = this.P[0] + s2;
    const s01 = this.P[4];
    const s10 = this.P[1];
    const s11 = this.P[5] + s2;
    const [si00, si01, si10, si11] = mat2_inv(s00, s01, s10, s11);
    const K = new Float64Array(8);
    for (let r = 0; r < 4; r++) {
      const p0 = this.P[r + 0 * 4];
      const p1 = this.P[r + 1 * 4];
      K[r + 0 * 4] = p0 * si00 + p1 * si10;
      K[r + 1 * 4] = p0 * si01 + p1 * si11;
    }
    for (let r = 0; r < 4; r++) {
      this.state[r] += K[r + 0 * 4] * innov[0] + K[r + 1 * 4] * innov[1];
    }
    const KH = mat4_zero();
    for (let r = 0; r < 4; r++) {
      KH[r + 0 * 4] = K[r + 0 * 4];
      KH[r + 1 * 4] = K[r + 1 * 4];
    }
    const I_KH = mat4_sub(mat4_identity(), KH);
    this.P = mat4_mul(I_KH, this.P);
    return this._stateToLatLng();
  }

  _stateToLatLng() {
    return fromMetric(this.state[0], this.state[1], this.originLat, this.originLng);
  }

  _buildQ(dt) {
    const q = Q_ACCEL_VAR;
    const dt2 = dt * dt;
    const dt3 = dt2 * dt;
    const dt4 = dt3 * dt;
    const Q = mat4_zero();
    Q[0] = q * dt4 / 4;
    Q[5] = q * dt4 / 4;
    Q[10] = q * dt2;
    Q[15] = q * dt2;
    Q[0 + 2 * 4] = q * dt3 / 2;
    Q[2 + 0 * 4] = q * dt3 / 2;
    Q[1 + 3 * 4] = q * dt3 / 2;
    Q[3 + 1 * 4] = q * dt3 / 2;
    return Q;
  }
}

// ─── Scoring colours ─────────────────────────────────────────────────────────

function scoreToFill(score) {
  const t = Math.min(score / 300, 1);
  const r = Math.round(110 + (5 - 110) * t);
  const g = Math.round(231 + (150 - 231) * t);
  const b = Math.round(183 + (105 - 183) * t);
  return `rgba(${r},${g},${b},${0.3 + 0.55 * t})`;
}

function scoreToStroke(score) {
  const t = Math.min(score / 300, 1);
  const g = Math.round(185 - 120 * t);
  return `rgba(16,${g},82,${0.7 + 0.3 * t})`;
}

const SCORE_STEPS = [20, 60, 100, 150, 200, 300];

// ─── H3 geometry helpers ─────────────────────────────────────────────────────

function cellBoundaryToGeoJSON(h3index) {
  const boundary = window.h3.cellToBoundary(h3index);
  const coords = boundary.map(([lat, lng]) => [lng, lat]);
  coords.push(coords[0]);
  return { type: "Polygon", coordinates: [coords] };
}

function bridgeTo(fromCell, toCell, fromLng, fromLat, toLng, toLat) {
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
    try { cell = h3.latLngToCell(lat, lng, H3_RES); } catch { continue; }
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
    if (!boundingDisk.has(cell))
      h3.gridDisk(cell, safeRadius).forEach((c) => boundingDisk.add(c));
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

// ─── GeoJSON builders ─────────────────────────────────────────────────────────

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

// ─── GPS options ──────────────────────────────────────────────────────────────

const GPS_OPTIONS = { enableHighAccuracy: true, timeout: 0, maximumAge: 0 };

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markerRef = useRef(null);
  const watchIdRef = useRef(null);
  const mapReady = useRef(false);
  const isRunningRef = useRef(false);

  const kalmanRef = useRef(null);
  const lastPingMsRef = useRef(null);
  const firstFixRef = useRef(false);

  // ← ZUPT: rolling IMU window + active flag
  const imuWindowRef = useRef([]);
  const zuptActiveRef = useRef(false);

  const stateRef = useRef({
    path: [],
    pathIndex: {},
    scoredCells: {},
    circuitCount: 0,
    totalScore: 0,
    lastCircuitScore: null,
    lastLng: null,
    lastLat: null,
    gpsCoords: [],
  });

  const [libsLoaded, setLibsLoaded] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [gpsReady, setGpsReady] = useState(false);
  const [gpsAccuracy, setGpsAccuracy] = useState(null);
  const [info, setInfo] = useState("Waiting for GPS signal…");
  const [stats, setStats] = useState({
    path: 0, circuits: 0, captured: 0, totalScore: 0, lastCircuitScore: null,
  });

  // ← ZUPT: track iOS permission state for the UI button
  const [imuPermission, setImuPermission] = useState("unknown"); // "unknown" | "granted" | "denied" | "notrequired"

  // ── Load external libs ──────────────────────────────────────────────────────
  useEffect(() => {
    loadLink("https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css");
    Promise.all([
      loadScript("https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"),
      loadScript("https://unpkg.com/h3-js@4.1.0/dist/h3-js.umd.js"),
    ]).then(() => setLibsLoaded(true));
  }, []);

  // ── Stats sync ──────────────────────────────────────────────────────────────
  const updateStats = useCallback(() => {
    const { path, scoredCells, circuitCount, totalScore, lastCircuitScore } = stateRef.current;
    setStats({
      path: path.length,
      circuits: circuitCount,
      captured: Object.keys(scoredCells).length,
      totalScore: Math.round(totalScore * 10) / 10,
      lastCircuitScore: lastCircuitScore != null ? Math.round(lastCircuitScore * 10) / 10 : null,
    });
  }, []);

  // ── Map redraw ──────────────────────────────────────────────────────────────
  const redraw = useCallback(() => {
    if (!mapReady.current || !mapRef.current) return;
    const m = mapRef.current;
    m.getSource("scored").setData({ type: "FeatureCollection", features: buildScoredFeatures(stateRef.current.scoredCells) });
    m.getSource("path").setData({ type: "FeatureCollection", features: buildPathFeatures(stateRef.current.path) });
    updateStats();
  }, [updateStats]);

  // ── Flash interior on circuit close ────────────────────────────────────────
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

  // ── Circuit close ───────────────────────────────────────────────────────────
  const closeCircuit = useCallback((loopStartIdx, closingCell) => {
    const st = stateRef.current;
    const loop = st.path.slice(loopStartIdx);
    const ringSet = new Set(loop);
    const interior = floodFillInterior(ringSet);

    const allCells = [...interior, ...loop];
    const A = allCells.reduce((sum, id) => {
      try { return sum + window.h3.cellArea(id, "m2"); } catch { return sum; }
    }, 0);
    const circuitScore = Math.sqrt(A);

    interior.forEach((id) => { st.scoredCells[id] = (st.scoredCells[id] || 0) + circuitScore; });
    loop.forEach((id) => { st.scoredCells[id] = (st.scoredCells[id] || 0) + circuitScore; });

    st.circuitCount++;
    st.totalScore = (st.totalScore || 0) + circuitScore;
    st.lastCircuitScore = circuitScore;

    const preLoop = st.path.slice(0, loopStartIdx);
    st.path = [...preLoop, closingCell];
    st.pathIndex = {};
    st.path.forEach((cell, i) => { st.pathIndex[cell] = i; });

    flashInterior(interior);
    setInfo(`Circuit #${st.circuitCount} · A=${Math.round(A).toLocaleString()}m² · +√A = +${circuitScore.toFixed(1)} · total ${st.totalScore.toFixed(1)}`);
    redraw();
  }, [flashInterior, redraw]);

  // ── Add cell to path ────────────────────────────────────────────────────────
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
    const bridge = bridgeTo(tail, rawCell, fromLng, fromLat, lng, lat);
    st.lastLng = lng;
    st.lastLat = lat;

    for (const cell of bridge) {
      const existingIdx = st.pathIndex[cell];

      if (existingIdx !== undefined) {
        const pathLen = st.path.length;
        const windowStart = Math.max(0, pathLen - BACKTRACK_TOL);

        if (existingIdx >= windowStart) {
          for (let i = existingIdx + 1; i < pathLen; i++) delete st.pathIndex[st.path[i]];
          st.path.splice(existingIdx + 1);
          redraw();
          return;
        } else {
          st.path.push(cell);
          closeCircuit(existingIdx, cell);
          return;
        }
      }

      st.pathIndex[cell] = st.path.length;
      st.path.push(cell);
    }

    redraw();
  }, [closeCircuit, redraw]);

  // ── Reset ───────────────────────────────────────────────────────────────────
  const resetAll = useCallback(() => {
    stateRef.current = {
      path: [], pathIndex: {}, scoredCells: {},
      circuitCount: 0, totalScore: 0, lastCircuitScore: null,
      lastLng: null, lastLat: null, gpsCoords: [],
    };
    kalmanRef.current = null;
    lastPingMsRef.current = null;
    // ← ZUPT: clear IMU state on reset too
    imuWindowRef.current = [];
    zuptActiveRef.current = false;
    if (mapReady.current && mapRef.current) {
      ["scored", "path", "flash"].forEach((src) =>
        mapRef.current.getSource(src).setData({ type: "FeatureCollection", features: [] })
      );
      mapRef.current.getSource("gpsTrail").setData({
        type: "Feature", geometry: { type: "LineString", coordinates: [] },
      });
    }
    updateStats();
    setInfo("Press Start Run to begin capturing territory with your GPS");
  }, [updateStats]);

  // ── Toggle run ──────────────────────────────────────────────────────────────
  const toggleRun = useCallback(() => {
    setIsRunning((prev) => {
      const next = !prev;
      isRunningRef.current = next;

      if (next) {
        const st = stateRef.current;
        st.path = [];
        st.pathIndex = {};
        st.lastLng = null;
        st.lastLat = null;
        st.gpsCoords = [];
        // ← ZUPT: clear IMU window when new run starts
        imuWindowRef.current = [];
        zuptActiveRef.current = false;
        if (mapReady.current && mapRef.current) {
          mapRef.current.getSource("path").setData({ type: "FeatureCollection", features: [] });
          mapRef.current.getSource("gpsTrail").setData({
            type: "Feature", geometry: { type: "LineString", coordinates: [] },
          });
        }
        setInfo("Run started — walk around to capture territory · close a loop to score!");
      } else {
        const st = stateRef.current;
        setInfo(`Run stopped · ${st.circuitCount} circuit(s) · ${Object.keys(st.scoredCells).length} cells · total ${(st.totalScore || 0).toFixed(1)}`);
      }
      return next;
    });
  }, []);

  // ── Re-center on user ───────────────────────────────────────────────────────
  const reCenterOnUser = useCallback(() => {
    if (!mapRef.current || !markerRef.current) return;
    const { lng, lat } = markerRef.current.getLngLat();
    mapRef.current.flyTo({ center: [lng, lat], zoom: 18, duration: 800 });
  }, []);

  // ── GPS position handler ────────────────────────────────────────────────────
  const handleGPSPosition = useCallback((pos) => {
    const { longitude: lng, latitude: lat, accuracy } = pos.coords;
    const nowMs = Date.now();

    setGpsAccuracy(Math.round(accuracy));

    if (!firstFixRef.current) {
      firstFixRef.current = true;
      kalmanRef.current = new KalmanGPS(lat, lng, accuracy);
      lastPingMsRef.current = nowMs;

      setGpsReady(true);
      if (markerRef.current) markerRef.current.setLngLat([lng, lat]);
      if (mapRef.current && !mapRef.current._hasCenteredOnUser) {
        mapRef.current.flyTo({ center: [lng, lat], zoom: 18, duration: 1200 });
        mapRef.current._hasCenteredOnUser = true;
      }
      setInfo("GPS acquired — press Start Run to capture territory");
      return;
    }

    const kf = kalmanRef.current;
    const dt = Math.max((nowMs - lastPingMsRef.current) / 1000, 0.01);
    lastPingMsRef.current = nowMs;

    kf.predict(dt);

    // ← ZUPT: if ZUPT is active, re-zero velocity after predict
    // (predict step rebuilds vx/vy via Q — we clamp it back down)
    if (zuptActiveRef.current) {
      kf.state[2] = 0;
      kf.state[3] = 0;
    }

    const { lat: kLat, lng: kLng } = kf.update(lat, lng, accuracy);

    if (markerRef.current) markerRef.current.setLngLat([kLng, kLat]);

    if (!isRunningRef.current) return;

    stateRef.current.gpsCoords.push([kLng, kLat]);
    if (mapReady.current && mapRef.current) {
      mapRef.current.getSource("gpsTrail").setData({
        type: "Feature",
        geometry: { type: "LineString", coordinates: stateRef.current.gpsCoords },
      });
    }

    const cell = (() => {
      try { return window.h3.latLngToCell(kLat, kLng, H3_RES); } catch { return null; }
    })();
    if (cell) addCell(cell, kLng, kLat);
  }, [addCell]);

  const handleGPSError = useCallback((err) => {
    console.warn("GPS error:", err);
    if (err.code !== err.TIMEOUT)
      setInfo(`GPS error: ${err.message}`);
  }, []);

  // ← ZUPT: DeviceMotion listener — runs independently of GPS
  useEffect(() => {
    const handleMotion = (e) => {
      const a  = e.acceleration;                 // gravity removed (may be null on some iOS)
      const ag = e.accelerationIncludingGravity;  // always available
      const r  = e.rotationRate;

      // Prefer gravity-removed if available, fall back to gravity-included
      const usingRaw = !a || (a.x == null && a.y == null && a.z == null);
      const ax = usingRaw ? (ag?.x ?? 0) : (a?.x ?? 0);
      const ay = usingRaw ? (ag?.y ?? 0) : (a?.y ?? 0);
      const az = usingRaw ? (ag?.z ?? 0) : (a?.z ?? 0);

      // rotationRate is in deg/s — convert to rad/s
      const gx = (r?.alpha ?? 0) * (Math.PI / 180);
      const gy = (r?.beta  ?? 0) * (Math.PI / 180);
      const gz = (r?.gamma ?? 0) * (Math.PI / 180);

      const now = Date.now();
      const accelMag = Math.sqrt(ax * ax + ay * ay + az * az);

      // Still check differs based on whether gravity is included or not:
      // - gravity-included: magnitude should be ~9.81 when phone is stationary
      // - gravity-removed:  magnitude should be ~0 when phone is stationary
      const isAccelStill = usingRaw
        ? Math.abs(accelMag - 9.81) < ZUPT_ACCEL_TOL
        : accelMag < ZUPT_ACCEL_TOL;

      const gyroMag = Math.sqrt(gx * gx + gy * gy + gz * gz);
      const isGyroStill = gyroMag < ZUPT_GYRO_TOL;

      // Push into rolling window, trim old samples
      imuWindowRef.current.push({ t: now, still: isAccelStill && isGyroStill });
      imuWindowRef.current = imuWindowRef.current.filter(s => now - s.t < ZUPT_WINDOW_MS);

      const windowFull = imuWindowRef.current.length > 3;  // need at least 3 samples
      const allStill   = imuWindowRef.current.every(s => s.still);

      if (windowFull && allStill) {
        if (!zuptActiveRef.current) {
          zuptActiveRef.current = true;

          // ← THE CORE ZUPT: zero velocity + tighten covariance
          if (kalmanRef.current) {
            kalmanRef.current.state[2] = 0;    // vx → 0
            kalmanRef.current.state[3] = 0;    // vy → 0
            // Tighten velocity variance so filter resists rebuilding speed
            // from the next noisy GPS ping
            kalmanRef.current.P[10] = 0.01;   // P[vx,vx] → near zero
            kalmanRef.current.P[15] = 0.01;   // P[vy,vy] → near zero
          }
        }
      } else {
        // User is moving again — release the brake
        zuptActiveRef.current = false;
      }
    };

    const startListening = () => {
      window.addEventListener("devicemotion", handleMotion);
    };

    // iOS 13+ requires explicit permission from a user gesture
    if (typeof DeviceMotionEvent !== "undefined" &&
        typeof DeviceMotionEvent.requestPermission === "function") {
      // iOS — we can't auto-request, needs user tap (see button in UI below)
      setImuPermission("needs-request");
    } else {
      // Android / desktop — just start listening
      setImuPermission("notrequired");
      startListening();
    }

    return () => window.removeEventListener("devicemotion", handleMotion);
  }, []);

  // ← ZUPT: iOS permission request — called from button tap
  const requestIOSMotionPermission = useCallback(async () => {
    try {
      const result = await DeviceMotionEvent.requestPermission();
      if (result === "granted") {
        setImuPermission("granted");
        window.addEventListener("devicemotion", (e) => {
          // Re-attach — the useEffect listener above may not have been added
          // We trigger a re-mount by toggling, but simpler: just re-add here.
          // In practice the useEffect cleanup + re-add handles this fine on Android;
          // on iOS we need this explicit path after the permission grant.
        });
        // Easiest: reload the effect by dispatching a custom event
        // Actually simplest — just reload the page after grant (one-time cost)
        window.location.reload();
      } else {
        setImuPermission("denied");
      }
    } catch {
      setImuPermission("denied");
    }
  }, []);

  // ── Map initialisation ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!libsLoaded || !mapContainerRef.current || mapRef.current) return;

    const initMap = (center) => {
      window.mapboxgl.accessToken = MAPBOX_TOKEN;
      const map = new window.mapboxgl.Map({
        container: mapContainerRef.current,
        style: "mapbox://styles/mapbox/dark-v11",
        center,
        zoom: center[0] === 0 && center[1] === 0 ? 2 : 18,
        minZoom: 2,
        maxZoom: 21,
        antialias: true,
      });
      map.addControl(new window.mapboxgl.NavigationControl({ showCompass: false }), "top-right");
      mapRef.current = map;

      map.on("load", () => {
        ["scored", "path", "flash"].forEach((id) =>
          map.addSource(id, { type: "geojson", data: { type: "FeatureCollection", features: [] } })
        );
        map.addSource("gpsTrail", {
          type: "geojson",
          data: { type: "Feature", geometry: { type: "LineString", coordinates: [] } },
        });

        map.addLayer({ id: "scored-fill", type: "fill", source: "scored", paint: { "fill-color": ["get", "fillColor"], "fill-opacity": 1 } });
        map.addLayer({ id: "scored-stroke", type: "line", source: "scored", paint: { "line-color": ["get", "strokeColor"], "line-width": 1.5 } });
        map.addLayer({ id: "path-fill", type: "fill", source: "path", paint: { "fill-color": ["get", "fillColor"], "fill-opacity": 1 } });
        map.addLayer({ id: "path-stroke", type: "line", source: "path", paint: { "line-color": ["get", "strokeColor"], "line-width": ["get", "strokeWidth"] } });
        map.addLayer({ id: "flash-fill", type: "fill", source: "flash", paint: { "fill-color": "rgba(255,255,120,0.6)", "fill-opacity": 1 } });
        map.addLayer({
          id: "gpsTrail-line",
          type: "line",
          source: "gpsTrail",
          paint: { "line-color": "#38bdf8", "line-width": 2.5, "line-opacity": 0.85 },
          layout: { "line-cap": "round", "line-join": "round" },
        });

        mapReady.current = true;
        updateStats();

        const el = document.createElement("div");
        el.style.cssText = `
          width:18px; height:18px; border-radius:50%;
          background:rgba(56,189,248,0.9);
          border:2.5px solid #fff;
          box-shadow:0 0 0 0 rgba(56,189,248,0.6);
          animation:gpsPulse 2s infinite;
          cursor:default;
        `;
        if (!document.getElementById("gpsPulseStyle")) {
          const style = document.createElement("style");
          style.id = "gpsPulseStyle";
          style.textContent = `
            @keyframes gpsPulse {
              0%  { box-shadow:0 0 0 0   rgba(56,189,248,0.55); }
              70% { box-shadow:0 0 0 14px rgba(56,189,248,0);   }
              100%{ box-shadow:0 0 0 0   rgba(56,189,248,0);    }
            }
            @keyframes gpsPulseRun {
              0%  { box-shadow:0 0 0 0   rgba(74,222,128,0.7);  }
              70% { box-shadow:0 0 0 18px rgba(74,222,128,0);   }
              100%{ box-shadow:0 0 0 0   rgba(74,222,128,0);    }
            }
          `;
          document.head.appendChild(style);
        }

        const marker = new window.mapboxgl.Marker({ element: el, anchor: "center" })
          .setLngLat([0, 0])
          .addTo(map);
        markerRef.current = marker;
        markerRef.current._el = el;
      });
    };

    if ("geolocation" in navigator) {
      const id = navigator.geolocation.watchPosition(handleGPSPosition, handleGPSError, GPS_OPTIONS);
      watchIdRef.current = id;
    }

    initMap([0, 0]);

    return () => {
      if (watchIdRef.current != null) navigator.geolocation.clearWatch(watchIdRef.current);
      if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; mapReady.current = false; }
    };
  }, [libsLoaded, handleGPSPosition, handleGPSError, updateStats]);

  // ── Marker colour while running ─────────────────────────────────────────────
  useEffect(() => {
    const el = markerRef.current?._el;
    if (!el) return;
    if (isRunning) {
      el.style.background = "rgba(74,222,128,0.95)";
      el.style.animation = "gpsPulseRun 1.4s infinite";
    } else {
      el.style.background = "rgba(56,189,248,0.9)";
      el.style.animation = "gpsPulse 2s infinite";
    }
  }, [isRunning]);

  const accuracyColor = !gpsReady ? "#fbbf24" : "#4ade80";

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div style={{
      fontFamily: "'DM Mono','Courier New',monospace",
      display: "flex", flexDirection: "column", height: "100vh",
      background: "#0a0e14", color: "#e2eaf5",
    }}>
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{
        display: "flex", alignItems: "center", gap: 14,
        padding: "9px 16px",
        background: "#0f1520",
        borderBottom: "1px solid rgba(99,140,200,0.18)",
        flexShrink: 0, flexWrap: "wrap",
      }}>
        <span style={{ fontSize: 12, letterSpacing: 2, textTransform: "uppercase", color: "#22d3ee", fontWeight: 700, whiteSpace: "nowrap" }}>
          H3 · Territory
        </span>
        <span style={{ fontSize: 10, color: "#3b5a80", letterSpacing: 1, whiteSpace: "nowrap" }}>
          RES {H3_RES} · Kalman GPS
        </span>

        <div style={{ width: 1, height: 22, background: "rgba(99,140,200,0.18)", flexShrink: 0 }} />

        {/* Stats pills */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {[["Path", stats.path], ["Circuits", stats.circuits], ["Captured", stats.captured], ["Score", stats.totalScore]].map(([label, val]) => (
            <div key={label} style={{
              background: "#141c2b", border: "1px solid rgba(99,140,200,0.18)",
              borderRadius: 6, padding: "4px 10px", fontSize: 11, color: "#6b82a0", whiteSpace: "nowrap",
            }}>
              {label}: <strong style={{ color: label === "Score" ? "#fbbf24" : "#e2eaf5" }}>{val}</strong>
            </div>
          ))}

          {stats.lastCircuitScore != null && (
            <div style={{
              background: "#141c2b", border: "1px solid rgba(251,191,36,0.35)",
              borderRadius: 6, padding: "4px 10px", fontSize: 11, color: "#fbbf24", whiteSpace: "nowrap",
            }}>
              Last +√A: <strong>+{stats.lastCircuitScore}</strong>
            </div>
          )}

          {/* GPS accuracy badge */}
          <div style={{
            background: "#141c2b", border: `1px solid ${accuracyColor}55`,
            borderRadius: 6, padding: "4px 10px", fontSize: 11, color: accuracyColor,
            whiteSpace: "nowrap", display: "flex", alignItems: "center", gap: 5,
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: accuracyColor, display: "inline-block", flexShrink: 0 }} />
            {!gpsReady ? "GPS…" : `GPS ±${gpsAccuracy}m · Kalman ✓`}
          </div>

          {/* ← ZUPT: IMU status badge */}
          <div style={{
            background: "#141c2b",
            border: `1px solid ${
              imuPermission === "granted" || imuPermission === "notrequired"
                ? "rgba(167,139,250,0.5)"
                : imuPermission === "denied"
                ? "rgba(239,68,68,0.4)"
                : "rgba(251,191,36,0.4)"
            }`,
            borderRadius: 6, padding: "4px 10px", fontSize: 11,
            color: imuPermission === "granted" || imuPermission === "notrequired"
              ? "#a78bfa"
              : imuPermission === "denied" ? "#f87171" : "#fbbf24",
            whiteSpace: "nowrap", display: "flex", alignItems: "center", gap: 5,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%", display: "inline-block", flexShrink: 0,
              background: imuPermission === "granted" || imuPermission === "notrequired"
                ? "#a78bfa" : imuPermission === "denied" ? "#f87171" : "#fbbf24",
            }} />
            {imuPermission === "notrequired" && "IMU ✓ ZUPT active"}
            {imuPermission === "granted"     && "IMU ✓ ZUPT active"}
            {imuPermission === "denied"      && "IMU denied"}
            {imuPermission === "needs-request" && "IMU needs permission"}
            {imuPermission === "unknown"     && "IMU…"}
          </div>
        </div>

        {/* Controls */}
        <button
          onClick={toggleRun}
          disabled={!gpsReady}
          style={{
            fontSize: 12, padding: "6px 18px", borderRadius: 7,
            fontFamily: "inherit", fontWeight: 700,
            cursor: gpsReady ? "pointer" : "not-allowed",
            transition: "all 0.2s", whiteSpace: "nowrap", letterSpacing: 0.5,
            border: isRunning ? "1.5px solid rgba(248,113,113,0.6)" : "1.5px solid rgba(74,222,128,0.55)",
            background: isRunning ? "rgba(239,68,68,0.15)" : "rgba(74,222,128,0.12)",
            color: isRunning ? "#f87171" : "#4ade80",
            boxShadow: isRunning ? "0 0 12px rgba(239,68,68,0.2)" : "0 0 12px rgba(74,222,128,0.15)",
            opacity: gpsReady ? 1 : 0.45,
          }}
        >
          {isRunning ? "⏹ Stop Run" : "▶ Start Run"}
        </button>

        {/* ← ZUPT: iOS permission button — only visible when needed */}
        {imuPermission === "needs-request" && (
          <button
            onClick={requestIOSMotionPermission}
            style={{
              fontSize: 11, padding: "5px 14px", borderRadius: 6,
              border: "1px solid rgba(167,139,250,0.5)", background: "rgba(167,139,250,0.1)",
              color: "#a78bfa", cursor: "pointer", fontFamily: "inherit", whiteSpace: "nowrap",
            }}
          >
            Enable IMU
          </button>
        )}

        <button
          onClick={reCenterOnUser}
          disabled={!gpsReady}
          title="Re-center on my location"
          style={{
            fontSize: 13, padding: "5px 10px", borderRadius: 6,
            border: "1px solid rgba(99,140,200,0.25)", background: "transparent",
            color: "#6b82a0", cursor: gpsReady ? "pointer" : "not-allowed",
            fontFamily: "inherit", opacity: gpsReady ? 1 : 0.4,
          }}
        >
          ◎
        </button>

        <button
          onClick={resetAll}
          disabled={isRunning}
          style={{
            fontSize: 11, padding: "5px 14px", borderRadius: 6,
            border: "1px solid rgba(239,68,68,0.4)", background: "rgba(239,68,68,0.08)",
            color: isRunning ? "#6b82a0" : "#f87171",
            cursor: isRunning ? "not-allowed" : "pointer",
            fontFamily: "inherit", whiteSpace: "nowrap", opacity: isRunning ? 0.45 : 1,
          }}
        >
          ↺ Reset
        </button>
      </div>

      {/* ── Map area ───────────────────────────────────────────────────────── */}
      <div style={{ position: "relative", flex: 1 }}>
        {!libsLoaded && (
          <div style={{
            position: "absolute", inset: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            color: "#6b82a0", fontSize: 13, zIndex: 10,
          }}>
            Loading map…
          </div>
        )}

        <div ref={mapContainerRef} style={{ width: "100%", height: "100%" }} />

        {/* Running pill */}
        {isRunning && (
          <div style={{
            position: "absolute", top: 12, left: "50%", transform: "translateX(-50%)",
            background: "rgba(10,14,20,0.92)", border: "1px solid rgba(74,222,128,0.5)",
            backdropFilter: "blur(8px)", borderRadius: 8, padding: "5px 16px",
            fontSize: 11, color: "#4ade80", pointerEvents: "none", zIndex: 25,
            whiteSpace: "nowrap", display: "flex", alignItems: "center", gap: 8,
          }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#4ade80", display: "inline-block", animation: "gpsPulseRun 1.4s infinite" }} />
            Tracking · Kalman + ZUPT
          </div>
        )}

        {/* Info bar */}
        <div style={{
          position: "absolute", bottom: 28, left: "50%", transform: "translateX(-50%)",
          background: "rgba(10,14,20,0.88)", border: "1px solid rgba(99,140,200,0.2)",
          backdropFilter: "blur(10px)", borderRadius: 10, padding: "8px 18px",
          fontSize: 11, color: "#22d3ee", pointerEvents: "none",
          zIndex: 20, maxWidth: "90vw", textAlign: "center",
        }}>
          {info}
        </div>

        {/* Legend */}
        <div style={{
          position: "absolute", bottom: 74, right: 14,
          background: "rgba(10,14,20,0.88)", border: "1px solid rgba(99,140,200,0.2)",
          backdropFilter: "blur(10px)", borderRadius: 10, padding: "10px 14px",
          zIndex: 20, fontSize: 10, color: "#6b82a0",
        }}>
          {[
            ["rgba(239,68,68,0.55)", "#ef4444", "Path start"],
            ["rgba(59,130,246,0.32)", "#3b82f6", "Active path"],
            ["rgba(251,191,36,0.7)", "#f59e0b", "Current cell"],
          ].map(([bg, border, label]) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
              <div style={{ width: 13, height: 13, borderRadius: 3, background: bg, border: `1.5px solid ${border}`, flexShrink: 0 }} />
              {label}
            </div>
          ))}
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
            <div style={{ width: 13, height: 3, borderRadius: 2, background: "#38bdf8", flexShrink: 0 }} />
            Kalman trail
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
            <div style={{ width: 13, height: 13, borderRadius: "50%", background: isRunning ? "rgba(74,222,128,0.9)" : "rgba(56,189,248,0.9)", border: "2px solid #fff", flexShrink: 0 }} />
            {isRunning ? "You (running)" : "You (idle)"}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 4 }}>
            {SCORE_STEPS.map((s) => (
              <div key={s} style={{ width: 13, height: 13, borderRadius: 3, background: scoreToFill(s), border: `1px solid ${scoreToStroke(s)}` }} title={`Score ${s}`} />
            ))}
            <span style={{ marginLeft: 4 }}>score (√A)</span>
          </div>
          <div style={{ marginTop: 6, fontSize: 9, color: "#3b5a80", lineHeight: 1.6 }}>
            Per circuit: +√A pts<br />
            A = Σ cellArea(hex) in m²
          </div>
        </div>
      </div>
    </div>
  );
}