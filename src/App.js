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

// Kalman process noise (m²/s²) — how much velocity can change per second.
// Tuned for walking pace; raise for cycling.
const Q_ACCEL_VAR = 0.5;

// Max dt (seconds) after which we clamp prediction to avoid covariance blow-up
// (handles tunnel / app-backgrounded gaps).
const MAX_DT_S = 5;

// ─── 4×4 Matrix math (column-major flat arrays, length 16) ──────────────────
// Layout: [m00,m10,m20,m30,  m01,m11,m21,m31,  m02,m12,m22,m32,  m03,m13,m23,m33]
// i.e. M[row + col*4]

function mat4_zero() { return new Float64Array(16); }

function mat4_identity() {
  const m = mat4_zero();
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** C = A × B */
function mat4_mul(A, B) {
  const C = mat4_zero();
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      for (let k = 0; k < 4; k++)
        C[r + c * 4] += A[r + k * 4] * B[k + c * 4];
  return C;
}

/** B = Aᵀ */
function mat4_T(A) {
  const B = mat4_zero();
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++)
      B[r + c * 4] = A[c + r * 4];
  return B;
}

/** C = A + B */
function mat4_add(A, B) {
  const C = new Float64Array(16);
  for (let i = 0; i < 16; i++) C[i] = A[i] + B[i];
  return C;
}

/** C = A - B */
function mat4_sub(A, B) {
  const C = new Float64Array(16);
  for (let i = 0; i < 16; i++) C[i] = A[i] - B[i];
  return C;
}

/** Invert a 2×2 matrix stored as [a,b,c,d] (row-major) → returns [a,b,c,d] */
function mat2_inv(a, b, c, d) {
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-12) return [1, 0, 0, 1]; // fallback
  return [d / det, -b / det, -c / det, a / det];
}

// ─── 2-vector helpers ────────────────────────────────────────────────────────

function vec2_sub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }

// ─── Coord helpers ───────────────────────────────────────────────────────────

/**
 * Convert lat/lng to local metric offsets (metres) from an origin.
 * Good for short distances (< a few km).
 */
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
//
// State vector  x = [px, py, vx, vy]ᵀ   (positions and velocities in metres)
//
// F (transition):
//   [1 0 dt  0]
//   [0 1  0 dt]
//   [0 0  1  0]
//   [0 0  0  1]
//
// H (observation — we only observe px, py):
//   [1 0 0 0]
//   [0 1 0 0]
//
// Q (process noise):
//   Discrete white-noise model for constant-velocity with acceleration noise q:
//   Q_pos = q·dt⁴/4,  Q_pos_vel = q·dt³/2,  Q_vel = q·dt²
//   Full 4×4:
//   [dt⁴/4  0     dt³/2  0    ]
//   [0      dt⁴/4  0     dt³/2]   × q
//   [dt³/2  0     dt²    0    ]
//   [0      dt³/2  0     dt²  ]
//
// R (measurement noise):
//   [σ²  0 ]   where σ = GPS accuracy in metres
//   [0   σ²]
//
// P (state covariance, 4×4 full matrix):
//   Initialised to diag(σ², σ², 0, 0) — we know position roughly,
//   velocity is completely unknown but we start it at zero and let it
//   grow via Q on the first prediction step.

class KalmanGPS {
  constructor(lat, lng, accuracy) {
    // Origin for metric projection
    this.originLat = lat;
    this.originLng = lng;

    // State vector [px, py, vx, vy] — all zeros at origin
    this.state = new Float64Array(4); // [0,0,0,0]

    // P — initial covariance
    // We're fairly sure about position (σ²_pos), completely unsure about velocity.
    // Off-diagonal pos-vel terms start at 0 (no correlation yet).
    const s2 = accuracy * accuracy;
    this.P = mat4_zero();
    this.P[0] = s2; // px-px
    this.P[5] = s2; // py-py
    // vx-vx, vy-vy: large uncertainty — let filter figure velocity out
    this.P[10] = s2 * 10;
    this.P[15] = s2 * 10;
    // All cross terms start at 0

    // H is fixed
    // [1 0 0 0]
    // [0 1 0 0]
    // We don't build it as a matrix object; we extract rows inline for speed.
  }

  /**
   * Predict step — advance state by dt seconds.
   * Returns predicted [lat, lng] (for optional display; not used for cell mapping).
   */
  predict(dt) {
    const dtClamped = Math.min(dt, MAX_DT_S);

    // F matrix
    const F = mat4_identity();
    F[0 + 2 * 4] = dtClamped; // px += vx*dt  → F[row0,col2]
    F[1 + 3 * 4] = dtClamped; // py += vy*dt  → F[row1,col3]

    // Apply F to state
    const [px, py, vx, vy] = this.state;
    this.state[0] = px + vx * dtClamped;
    this.state[1] = py + vy * dtClamped;
    // velocity unchanged

    // P = F·P·Fᵀ + Q
    const FP = mat4_mul(F, this.P);
    const FPFt = mat4_mul(FP, mat4_T(F));
    const Q = this._buildQ(dtClamped);
    this.P = mat4_add(FPFt, Q);

    return this._stateToLatLng();
  }

  /**
   * Update step — incorporate a GPS measurement.
   * @param {number} lat
   * @param {number} lng
   * @param {number} accuracy  GPS accuracy in metres
   * @returns {{ lat, lng }}   Filtered position
   */
  update(lat, lng, accuracy) {
    const [mx, my] = toMetric(lat, lng, this.originLat, this.originLng);

    // Innovation: z - H·x  (H extracts first two components)
    const innov = vec2_sub([mx, my], [this.state[0], this.state[1]]);

    // S = H·P·Hᵀ + R
    // H·P is just the first two rows of P.
    // (H·P·Hᵀ) is the top-left 2×2 of P.
    const s2 = accuracy * accuracy;
    // S (2×2, row-major: [s00, s01, s10, s11])
    const s00 = this.P[0] + s2; // P[px,px] + R[0,0]
    const s01 = this.P[4];        // P[px,py]
    const s10 = this.P[1];        // P[py,px]
    const s11 = this.P[5] + s2; // P[py,py] + R[1,1]

    // S⁻¹ (2×2)
    const [si00, si01, si10, si11] = mat2_inv(s00, s01, s10, s11);

    // Kalman gain K = P·Hᵀ·S⁻¹
    // P·Hᵀ is the first two columns of P (since Hᵀ picks cols 0 and 1).
    // Result K is 4×2.
    // K[r,0] = P[r,0]*si00 + P[r,1]*si10
    // K[r,1] = P[r,0]*si01 + P[r,1]*si11
    const K = new Float64Array(8); // 4 rows × 2 cols, K[row + col*4]
    for (let r = 0; r < 4; r++) {
      const p0 = this.P[r + 0 * 4]; // P[r, col0]
      const p1 = this.P[r + 1 * 4]; // P[r, col1]
      K[r + 0 * 4] = p0 * si00 + p1 * si10;
      K[r + 1 * 4] = p0 * si01 + p1 * si11;
    }

    // State update: x = x + K·innov
    for (let r = 0; r < 4; r++) {
      this.state[r] += K[r + 0 * 4] * innov[0] + K[r + 1 * 4] * innov[1];
    }

    // Covariance update: P = (I - K·H)·P
    // K·H is 4×4; (K·H)[r,c] = K[r,0]*H[0,c] + K[r,1]*H[1,c]
    // H[0,c] = 1 if c==0 else 0;  H[1,c] = 1 if c==1 else 0
    // So (K·H)[r,c] = K[r,0] if c==0, K[r,1] if c==1, else 0
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

  /** Discrete white-noise Q for constant-velocity model */
  _buildQ(dt) {
    const q = Q_ACCEL_VAR;
    const dt2 = dt * dt;
    const dt3 = dt2 * dt;
    const dt4 = dt3 * dt;
    const Q = mat4_zero();
    // px-px, py-py
    Q[0] = q * dt4 / 4;
    Q[5] = q * dt4 / 4;
    // vx-vx, vy-vy
    Q[10] = q * dt2;
    Q[15] = q * dt2;
    // px-vx, vx-px  (and py-vy, vy-py)
    Q[0 + 2 * 4] = q * dt3 / 2; // P[px, vx]
    Q[2 + 0 * 4] = q * dt3 / 2; // P[vx, px]
    Q[1 + 3 * 4] = q * dt3 / 2; // P[py, vy]
    Q[3 + 1 * 4] = q * dt3 / 2; // P[vy, py]
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

  // Kalman filter instance — created on first valid GPS fix
  const kalmanRef = useRef(null);
  // Timestamp (ms) of previous GPS ping, for computing dt
  const lastPingMsRef = useRef(null);
  // Has the user seen their first position on the map?
  const firstFixRef = useRef(false);

  // Game state kept in a ref (mutated directly for performance)
  const stateRef = useRef({
    path: [],
    pathIndex: {},
    scoredCells: {},
    circuitCount: 0,
    totalScore: 0,
    lastCircuitScore: null,
    lastLng: null,
    lastLat: null,
    gpsCoords: [],   // raw Kalman-filtered coords for trail
  });

  const [libsLoaded, setLibsLoaded] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [gpsReady, setGpsReady] = useState(false);
  const [gpsAccuracy, setGpsAccuracy] = useState(null);
  const [info, setInfo] = useState("Waiting for GPS signal…");
  const [stats, setStats] = useState({
    path: 0, circuits: 0, captured: 0, totalScore: 0, lastCircuitScore: null,
  });

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

    // Keep cells that came before the loop; closing cell becomes the new tail.
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
          // Backtrack — trim path
          for (let i = existingIdx + 1; i < pathLen; i++) delete st.pathIndex[st.path[i]];
          st.path.splice(existingIdx + 1);
          redraw();
          return;
        } else {
          // Loop detected — close circuit
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

    // ── FIRST FIX: show unconditionally — initialise Kalman ──────────────────
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

    // ── SUBSEQUENT FIXES: Kalman predict → update ─────────────────────────────
    const kf = kalmanRef.current;
    const dt = Math.max((nowMs - lastPingMsRef.current) / 1000, 0.01); // seconds, min 10ms
    lastPingMsRef.current = nowMs;

    kf.predict(dt);
    const { lat: kLat, lng: kLng } = kf.update(lat, lng, accuracy);

    // Move marker to Kalman-filtered position
    if (markerRef.current) markerRef.current.setLngLat([kLng, kLat]);

    if (!isRunningRef.current) return;

    // GPS trail (Kalman-filtered)
    stateRef.current.gpsCoords.push([kLng, kLat]);
    if (mapReady.current && mapRef.current) {
      mapRef.current.getSource("gpsTrail").setData({
        type: "Feature",
        geometry: { type: "LineString", coordinates: stateRef.current.gpsCoords },
      });
    }

    // Feed filtered position into H3 cell tracking
    const cell = (() => {
      try { return window.h3.latLngToCell(kLat, kLng, H3_RES); } catch { return null; }
    })();
    if (cell) addCell(cell, kLng, kLat);
  }, [addCell]);

  const handleGPSError = useCallback((err) => {
    console.warn("GPS error:", err);
    // Only surface non-timeout errors — timeouts are normal with maximumAge:0
    if (err.code !== err.TIMEOUT)
      setInfo(`GPS error: ${err.message}`);
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
        zoom: center[0] === 0 && center[1] === 0 ? 2 : 18, // world view until GPS
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

        // GPS marker element
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

        // Start marker at world-centre; it will jump to user on first fix
        const marker = new window.mapboxgl.Marker({ element: el, anchor: "center" })
          .setLngLat([0, 0])
          .addTo(map);
        markerRef.current = marker;
        markerRef.current._el = el;
      });
    };

    // GPS watch — always running (even before run starts) for live position dot
    if ("geolocation" in navigator) {
      const id = navigator.geolocation.watchPosition(handleGPSPosition, handleGPSError, GPS_OPTIONS);
      watchIdRef.current = id;
    }

    // Map always starts with world view; no GPS coords needed
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

  // ── Accuracy badge colour ───────────────────────────────────────────────────
  const accuracyColor = !gpsReady
    ? "#fbbf24"
    : "#4ade80";  // green once Kalman is active (filter handles the rest)

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
            {!gpsReady
              ? "GPS…"
              : `GPS ±${gpsAccuracy}m · Kalman ✓`}
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
            Tracking · Kalman-filtered GPS
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