/* GeoDuelo: utilidades compartidas entre el modo duelo (juego.js) y el explorador (explorar.js). */
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const COLOR = { jugador: "#38bdf8", modelo: "#f59e0b", real: "#22c55e" };
// Tiles de Esri (no requieren API key; CARTO ahora sí la pide).
const esri = (servicio) => (x, y, l) =>
  `https://server.arcgisonline.com/ArcGIS/rest/services/${servicio}/MapServer/tile/${l}/${y}/${x}`;
const TILES = {
  mapa: esri("World_Street_Map"),
  satelite: esri("World_Imagery"),
  oscuro: esri("Canvas/World_Dark_Gray_Base"),
};
const TEXTURA = "https://cdn.jsdelivr.net/npm/three-globe@2.45.6/example/img/";
const GEOJSON = "https://cdn.jsdelivr.net/gh/vasturiano/globe.gl@v2.46.2/example/datasets/ne_110m_admin_0_countries.geojson";
const CODIGOS_FALTANTES = { France: "FR", Norway: "NO", Kosovo: "XK", "N. Cyprus": "CY", Somaliland: "SO" };

/* ------------------------------------------------------------ utilidades */
const nombresPais = (() => {
  try { return new Intl.DisplayNames(["es"], { type: "region" }); } catch { return null; }
})();
function nombrePais(code) {
  if (!code) return "Océano";
  try { return nombresPais?.of(code) || code; } catch { return code; }
}
function bandera(code, ancho = 20) {
  if (!code || code.length !== 2) return "";
  const c = code.toLowerCase();
  return `<img src="https://flagcdn.com/w40/${c}.png" width="${ancho}" height="${Math.round(ancho * 0.75)}" alt="" loading="lazy">`;
}
const fmt = (n) => Math.round(n).toLocaleString("es-AR");
function fmtKm(km) {
  if (km == null) return "—";
  if (km < 1) return `${Math.round(km * 1000)} m`;
  if (km < 10) return `${km.toFixed(1)} km`;
  return `${fmt(km)} km`;
}
function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("visible");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("visible"), ms);
}
function animarNumero(el, hasta, ms = 1100) {
  const t0 = performance.now();
  const paso = (t) => {
    const k = Math.min(1, (t - t0) / ms);
    el.textContent = fmt(hasta * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(paso);
  };
  requestAnimationFrame(paso);
}
const guardar = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* sin storage */ } };
const leer = (k, def) => { try { return JSON.parse(localStorage.getItem(k)) ?? def; } catch { return def; } };

async function api(ruta, opciones) {
  const r = await fetch(ruta, opciones);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
  return r.json();
}

/* ------------------------------------------------------------ países (point-in-polygon) */
let paises = [];
fetch(GEOJSON).then((r) => r.json()).then((g) => {
  paises = g.features.map((f) => {
    const p = f.properties;
    const code = p.ISO_A2 && p.ISO_A2 !== "-99" ? p.ISO_A2 : CODIGOS_FALTANTES[p.NAME] || null;
    return { code, f, bbox: d3.geoBounds(f) };
  });
}).catch(() => console.warn("No se pudo cargar el GeoJSON de países"));

function paisDe(lat, lng) {
  if (lat == null) return null;
  for (const p of paises) {
    const [[x0, y0], [x1, y1]] = p.bbox;
    const dentroLon = x0 <= x1 ? lng >= x0 && lng <= x1 : lng >= x0 || lng <= x1;
    if (lat >= y0 && lat <= y1 && dentroLon && d3.geoContains(p.f, [lng, lat])) return p.code;
  }
  return null;
}

/* ------------------------------------------------------------ pins */
function crearPin(color, etiqueta) {
  const el = document.createElement("div");
  el.className = "pin";
  el.innerHTML = `
    <svg viewBox="0 0 30 40"><path d="M15 39C15 39 3 24.5 3 15a12 12 0 1 1 24 0c0 9.5-12 24-12 24Z"
      fill="${color}" stroke="#fff" stroke-width="2"/><circle cx="15" cy="15" r="4.6" fill="#fff"/></svg>
    ${etiqueta ? `<div class="etiqueta" style="color:${color}">${etiqueta}</div>` : ""}`;
  return el;
}

/* ------------------------------------------------------------ globo con tiles, pins, arcos y columnas */
function crearGlobo(cont) {
  const globo = new Globe(cont, { rendererConfig: { antialias: true, alpha: true } })
    .backgroundColor("rgba(0,0,0,0)")
    .globeTileEngineUrl(TILES.mapa)
    .globeTileEngineMaxLevel(16)
    .showAtmosphere(true).atmosphereColor("#7aa2ff").atmosphereAltitude(0.16)
    .htmlElementsData([]).htmlLat("lat").htmlLng("lng").htmlAltitude(0).htmlElement((d) => d.el)
    .htmlElementVisibilityModifier((el, visible) => { el.style.opacity = visible ? 1 : 0; })
    .arcsData([]).arcColor("color").arcStroke(0.55).arcDashLength(0.45).arcDashGap(0.12)
    .arcDashInitialGap(1).arcDashAnimateTime(2200).arcAltitudeAutoScale(0.35)
    .pointsData([]).pointLat("lat").pointLng("lon").pointColor(() => "rgba(245, 158, 11, 0.85)")
    .pointAltitude((d) => Math.max(0.006, d.prob * 0.7)).pointRadius(0.9).pointsMerge(false)
    .pointLabel((d) => `<b style="color:${COLOR.modelo}">${(d.prob * 100).toFixed(1)}%</b> ${nombrePais(paisDe(d.lat, d.lon))}`)
    .ringsData([]).ringLat("lat").ringLng("lng").ringColor(() => (t) => `rgba(34,197,94,${1 - t})`)
    .ringMaxRadius(4).ringPropagationSpeed(2.5).ringRepeatPeriod(1000);
  globo.controls().zoomSpeed = 1.4;
  globo.pointOfView({ lat: 15, lng: -30, altitude: 2.6 }, 0);
  new ResizeObserver(() => globo.width(cont.clientWidth).height(cont.clientHeight)).observe(cont);
  return globo;
}

// Clic propio: rayo desde la cámara contra la esfera (funciona igual con tiles, puntos o arcos encima).
function coordsDesdePantalla(globo, cx, cy) {
  const canvas = globo.renderer().domElement;
  const r = canvas.getBoundingClientRect();
  const nx = ((cx - r.left) / r.width) * 2 - 1;
  const ny = -((cy - r.top) / r.height) * 2 + 1;
  const cam = globo.camera();
  cam.updateMatrixWorld();
  const mul = (m, v) => [0, 1, 2, 3].map((i) => m[i] * v[0] + m[4 + i] * v[1] + m[8 + i] * v[2] + m[12 + i] * v[3]);
  const desproyectar = (z) => {
    let v = mul(cam.projectionMatrixInverse.elements, [nx, ny, z, 1]);
    v = v.map((c) => c / v[3]);
    return mul(cam.matrixWorld.elements, v);
  };
  const a = desproyectar(-1), b = desproyectar(1);
  let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const n = Math.hypot(...d);
  d = d.map((c) => c / n);
  const R = globo.getGlobeRadius();
  const od = a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
  const oo = a[0] ** 2 + a[1] ** 2 + a[2] ** 2;
  const disc = od * od - (oo - R * R);
  if (disc < 0) return null;
  const t = -od - Math.sqrt(disc);
  const p = { x: a[0] + t * d[0], y: a[1] + t * d[1], z: a[2] + t * d[2] };
  const g = globo.toGeoCoords(p);
  return { lat: g.lat, lng: ((g.lng + 540) % 360) - 180 };
}

// Llama a alClic({lat, lng}) en clics (no en arrastres) sobre el globo.
function alClicEnGlobo(globo, cont, alClic) {
  let inicio = null;
  cont.addEventListener("pointerdown", (e) => { inicio = { x: e.clientX, y: e.clientY }; });
  cont.addEventListener("pointerup", (e) => {
    if (!inicio) return;
    const mov = Math.hypot(e.clientX - inicio.x, e.clientY - inicio.y);
    inicio = null;
    if (mov > 5 || e.button !== 0) return;
    const c = coordsDesdePantalla(globo, e.clientX, e.clientY);
    if (c) alClic(c);
  });
}

function conectarEstiloMapa(globo, selector) {
  const botones = $$(`${selector} button`);
  botones.forEach((b) => b.addEventListener("click", () => {
    botones.forEach((x) => x.classList.toggle("sel", x === b));
    globo.globeTileEngineUrl(TILES[b.dataset.valor]);
    guardar("geoduelo.mapa", b.dataset.valor);
  }));
  const previo = leer("geoduelo.mapa", "mapa");
  if (TILES[previo] && previo !== "mapa") botones.find((b) => b.dataset.valor === previo)?.click();
}

function puntoMedio(puntos) {
  // Promedio en 3D (evita problemas con el antimeridiano).
  let x = 0, y = 0, z = 0;
  for (const p of puntos) {
    const la = (p.lat * Math.PI) / 180, lo = (p.lng * Math.PI) / 180;
    x += Math.cos(la) * Math.cos(lo); y += Math.cos(la) * Math.sin(lo); z += Math.sin(la);
  }
  return { lat: (Math.atan2(z, Math.hypot(x, y)) * 180) / Math.PI, lng: (Math.atan2(y, x) * 180) / Math.PI };
}

// Rellena #res-metricas, #res-top5 y #res-vistas con lo que pensó el modelo. Clic en una vista: el top-5 y las
// columnas del globo pasan a ser los de esa vista sola; otro clic (o "Promedio") vuelve a la predicción global.
function pintarDetalleModelo(globo, m, urlsVistas, { etiquetas = null, alAmpliar = null } = {}) {
  const metricas = [
    ["Confianza top-1", `${(m.prob * 100).toFixed(1)}%`],
    ["Celda real en el ranking", `#${m.rango_celda_real} de 70`],
    ["Incertidumbre", `${(m.entropia * 100).toFixed(0)}%`],
    ["Celda de mínima distancia esperada", `${fmtKm(m.bayes.km)} · ${fmt(m.bayes.puntos)} pts`],
  ];
  $("#res-metricas").innerHTML = metricas.map(([k, v]) => `<div class="metrica"><span>${k}</span><b>${v}</b></div>`).join("");

  const nombreVista = (k) => (etiquetas ? `vista ${etiquetas[k]}` : `vista ${k + 1}`);
  let seleccionada = null;

  function pintarTop(k) {
    seleccionada = k;
    const fuente = k == null ? m : m.vistas[k];
    $("#res-top5-titulo").innerHTML = k == null
      ? "Top-5 celdas · promedio de las 4 vistas"
      : `Top-5 celdas · solo ${nombreVista(k)} <button class="chip-volver" id="btn-promedio">← Promedio</button>`;
    $("#btn-promedio")?.addEventListener("click", () => pintarTop(null));
    $("#res-top5").innerHTML = fuente.top.map((c, i) => {
      const cp = paisDe(c.lat, c.lon);
      return `<li data-lat="${c.lat}" data-lon="${c.lon}"><div class="fondo-prob" style="width:${c.prob * 100}%"></div>
        <span class="n">${i + 1}</span><span class="lugar">${bandera(cp, 16)} ${nombrePais(cp)}</span>
        <span class="p">${(c.prob * 100).toFixed(1)}%</span><span class="k">${fmtKm(c.km)}</span></li>`;
    }).join("");
    $$("#res-top5 li").forEach((li) => li.addEventListener("click", () =>
      globo.pointOfView({ lat: +li.dataset.lat, lng: +li.dataset.lon, altitude: 0.9 }, 900)));
    $$("#res-vistas > div").forEach((d, i) => d.classList.toggle("sel", i === k));
    $("#res-top5").classList.toggle("de-vista", k != null);
    globo.pointsData(fuente.distribucion);
  }

  $("#res-vistas").innerHTML = m.vistas.map((v, k) => {
    const cp = paisDe(v.lat, v.lon);
    return `<div title="Clic: top-5 de esta vista · ${nombrePais(cp)} ${(v.prob * 100).toFixed(0)}%">
      <img src="${urlsVistas[k]}" alt="">
      ${alAmpliar ? '<button class="btn-ampliar" title="Ver en grande">⤢</button>' : ""}
      <span>${etiquetas ? `<b>${etiquetas[k]}</b> ` : ""}${bandera(cp, 14)} ${fmtKm(v.km)}</span></div>`;
  }).join("");
  $$("#res-vistas > div").forEach((d, k) => {
    d.addEventListener("click", () => pintarTop(seleccionada === k ? null : k));
    d.querySelector(".btn-ampliar")?.addEventListener("click", (e) => { e.stopPropagation(); alAmpliar(k); });
  });
  pintarTop(null);
}
