/* GeoDuelo: modo explorador. Street View (Maps JavaScript API) para caminar; al capturar, el servidor baja las 4
   vistas del panorama con la Street View Static API y corre el modelo. */
"use strict";

const est = {
  sv: null, // google.maps.StreetViewPanorama
  servicio: null, // google.maps.StreetViewService
  pos: null, // {lat, lng} del panorama actual
  volar: false, // si el próximo cambio de posición debe mover la cámara del globo
  orientacion: "norte", // norte | mirada
  capturando: false,
  capturas: [],
  activa: null,
  celdas: [],
  visorK: 0,
};
const DIRECCIONES = ["N", "E", "S", "O"];

/* ------------------------------------------------------------ globo */
const contGlobo = $("#globo");
const globo = crearGlobo(contGlobo);
conectarEstiloMapa(globo, "#estilo-mapa");
alClicEnGlobo(globo, contGlobo, (c) => { est.volar = false; irA(c, radioSegunZoom()); });

// Cuanto más alejado está el globo, más grande el radio de búsqueda de panoramas.
function radioSegunZoom() {
  const alt = globo.pointOfView().altitude;
  return Math.round(Math.min(50000, Math.max(150, alt * 20000)));
}

function pintarGlobo() {
  const pins = [], arcos = [];
  const c = est.activa;
  if (c) {
    const { real, modelo: m } = c;
    pins.push({ lat: real.lat, lng: real.lon, el: crearPin(COLOR.real, "Captura") },
      { lat: m.lat, lng: m.lon, el: crearPin(COLOR.modelo, "Modelo") });
    arcos.push({ startLat: m.lat, startLng: m.lon, endLat: real.lat, endLng: real.lon, color: [COLOR.modelo, COLOR.real] });
  }
  if (est.pos && !(c && Math.abs(c.real.lat - est.pos.lat) < 1e-6 && Math.abs(c.real.lon - est.pos.lng) < 1e-6)) {
    pins.push({ ...est.pos, el: crearPin(COLOR.jugador, "Estás acá") });
  }
  // Las columnas de probabilidad las maneja pintarDetalleModelo (promedio o la vista seleccionada).
  globo.htmlElementsData(pins).arcsData(arcos)
    .ringsData(c ? [{ lat: c.real.lat, lng: c.real.lon }] : []);
}

/* ------------------------------------------------------------ Google Maps */
async function iniciar() {
  api("/api/info").then((i) => { est.celdas = i.celdas; }).catch(() => {});
  let cfg;
  try {
    cfg = await api("/api/config");
  } catch {
    toast("No se pudo hablar con el servidor", 6000);
    return;
  }
  if (!cfg.google_maps_key) {
    $("#sv-vacio").classList.add("oculto");
    $("#sv-config").classList.remove("oculto");
    return;
  }
  window.gm_authFailure = () => toast("Google rechazó la API key: revisá que la Maps JavaScript API esté habilitada", 9000);
  window.__iniciarMapas = iniciarMapas;
  const s = document.createElement("script");
  s.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(cfg.google_maps_key)}` +
    "&v=weekly&language=es&callback=__iniciarMapas";
  s.async = true;
  s.onerror = () => toast("No se pudo cargar Google Maps", 6000);
  document.head.appendChild(s);
}

function iniciarMapas() {
  est.servicio = new google.maps.StreetViewService();
  est.sv = new google.maps.StreetViewPanorama($("#streetview"), {
    visible: false, addressControl: false, fullscreenControl: false, motionTracking: false,
    motionTrackingControl: false, enableCloseButton: false, zoomControl: true, panControl: true,
    linksControl: true, clickToGo: true, showRoadLabels: true,
  });
  est.sv.addListener("position_changed", actualizarPosicion);
  est.sv.addListener("pov_changed", actualizarBrujula);
  const ultimo = leer("geoduelo.explorador.ultimo", null);
  if (ultimo?.pano) {
    est.volar = true;
    est.sv.setPano(ultimo.pano);
    est.sv.setPov({ heading: ultimo.heading || 0, pitch: 0 });
    mostrarStreetView();
  }
}

function mostrarStreetView() {
  est.sv.setVisible(true);
  $("#sv-vacio").classList.add("oculto");
  $("#barra-captura").classList.remove("oculto");
}

async function irA(c, radio, silencioso = false) {
  if (!est.servicio) {
    if (!silencioso) toast("Google Maps todavía no cargó");
    return false;
  }
  const pedido = { location: c, radius: radio, preference: google.maps.StreetViewPreference.NEAREST };
  // Solo cobertura oficial de Google (la misma del dataset), no photospheres de usuarios.
  if (google.maps.StreetViewSource?.GOOGLE) pedido.sources = [google.maps.StreetViewSource.GOOGLE];
  else pedido.source = google.maps.StreetViewSource.OUTDOOR;
  try {
    const { data } = await est.servicio.getPanorama(pedido);
    est.volar = true;
    est.sv.setPano(data.location.pano);
    est.sv.setPov({ heading: est.sv.getPov()?.heading ?? 0, pitch: 0 });
    mostrarStreetView();
    return true;
  } catch {
    if (!silencioso) toast(`No hay Street View a menos de ${fmtKm(radio / 1000)} de ahí. Probá cerca de una ruta o alejá el globo.`, 4500);
    return false;
  }
}

function actualizarPosicion() {
  const p = est.sv.getPosition();
  if (!p) return;
  est.pos = { lat: p.lat(), lng: p.lng() };
  const cp = paisDe(est.pos.lat, est.pos.lng);
  $("#info-posicion").innerHTML = `${bandera(cp, 16)} ${nombrePais(cp)} · ${est.pos.lat.toFixed(5)}, ${est.pos.lng.toFixed(5)}`;
  if (est.volar) {
    est.volar = false;
    globo.pointOfView({ ...est.pos, altitude: Math.min(globo.pointOfView().altitude, 1.2) }, 1200);
  }
  pintarGlobo();
  guardar("geoduelo.explorador.ultimo", { pano: est.sv.getPano(), heading: est.sv.getPov()?.heading || 0 });
}

/* ------------------------------------------------------------ brújula */
{
  const g = $("#b-marcas");
  g.innerHTML = [0, 90, 180, 270].map((a) =>
    `<line x1="50" y1="5" x2="50" y2="13" class="b-marca" transform="rotate(${a} 50 50)"/>`).join("");
}
function headingBase() {
  return est.orientacion === "norte" ? 0 : ((est.sv?.getPov()?.heading ?? 0) + 360) % 360;
}
function actualizarBrujula() {
  const h = est.sv?.getPov()?.heading ?? 0;
  $("#b-aguja").setAttribute("transform", `rotate(${h} 50 50)`);
  const base = headingBase();
  $("#b-marcas").setAttribute("transform", `rotate(${base} 50 50)`);
  $("#rumbos").innerHTML = "Se van a mandar: " + [0, 1, 2, 3].map((k) => {
    const r = Math.round((base + 90 * k) % 360);
    return est.orientacion === "norte" ? `<b>${DIRECCIONES[k]}</b> ${r}°` : `<b>${r}°</b>`;
  }).join(" · ");
}
$$("#orientacion button").forEach((b) => b.addEventListener("click", () => {
  $$("#orientacion button").forEach((x) => x.classList.toggle("sel", x === b));
  est.orientacion = b.dataset.valor;
  actualizarBrujula();
  toast(est.orientacion === "norte"
    ? "El modelo va a recibir las vistas N, E, S y O, como en el dataset"
    : "El modelo va a recibir 4 vistas arrancando desde hacia donde mirás");
}));
actualizarBrujula();

/* ------------------------------------------------------------ lugar al azar */
const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
async function lugarAlAzar() {
  const btns = [$("#btn-azar"), $("#btn-azar-grande")];
  btns.forEach((b) => { b.disabled = true; });
  try {
    for (let i = 0; i < 25; i++) {
      // Mayormente cerca de los centros de celda (donde hay cobertura), a veces en cualquier lado.
      let c;
      if (est.celdas.length && Math.random() < 0.85) {
        const b = est.celdas[Math.floor(Math.random() * est.celdas.length)];
        c = { lat: Math.max(-75, Math.min(75, b.lat + gauss() * 3)), lng: ((b.lon + gauss() * 3 + 540) % 360) - 180 };
      } else {
        c = { lat: Math.random() * 130 - 55, lng: Math.random() * 360 - 180 };
      }
      if (paises.length && !paisDe(c.lat, c.lng)) continue;
      if (await irA(c, 50000, true)) return;
    }
    toast("No encontré cobertura, probá de nuevo");
  } finally {
    btns.forEach((b) => { b.disabled = false; });
  }
}

/* ------------------------------------------------------------ captura */
async function capturar() {
  if (!est.sv?.getVisible() || est.capturando || !est.pos) return;
  document.activeElement?.blur();
  const btn = $("#btn-capturar");
  const pano = est.sv.getPano();
  const base = headingBase();
  est.capturando = true;
  btn.classList.add("cargando-btn");
  btn.innerHTML = "⏳ Analizando…";
  let res;
  try {
    res = await api("/api/explorar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pano, lat: est.pos.lat, lon: est.pos.lng, heading: base }),
    });
  } catch (e) {
    toast(e.message, 9000);
    return;
  } finally {
    est.capturando = false;
    btn.classList.remove("cargando-btn");
    btn.innerHTML = "📸 Mostrarle al modelo <kbd>espacio</kbd>";
  }
  res.real.pais = paisDe(res.real.lat, res.real.lon);
  res.etiquetas = res.headings.map((h, k) =>
    est.orientacion === "norte" ? DIRECCIONES[k] : `${Math.round(h)}°`);
  res.heading = est.sv.getPov()?.heading ?? 0;
  est.capturas.unshift(res);
  mostrarCaptura(res);
}

function veredicto(km) {
  if (km < 25) return "¡Lo clavó! 🎯";
  if (km < 200) return "Muy cerca";
  if (km < 750) return "Bien encaminado";
  if (km < 2500) return "Más o menos";
  return "Perdidísimo";
}

function mostrarCaptura(res) {
  est.activa = res;
  const { real, modelo: m } = res;
  pintarGlobo();
  const encuadre = [{ lat: real.lat, lng: real.lon }, { lat: m.lat, lng: m.lon }];
  globo.pointOfView({ ...puntoMedio(encuadre), altitude: Math.min(2.6, Math.max(0.35, m.km / 3200)) }, 1400);

  $("#exp-ayuda").classList.add("oculto");
  $("#exp-resultado").classList.remove("oculto");
  const titulo = $("#res-titulo");
  titulo.className = "res-titulo gana-modelo";
  titulo.textContent = veredicto(m.km);
  $("#res-lugar").innerHTML = `${bandera(real.pais, 28)} <span>Capturaste en <b>${nombrePais(real.pais)}</b>
    <span style="color:var(--texto-3)">· ${real.lat.toFixed(4)}, ${real.lon.toFixed(4)}</span></span>`;
  $("#res-km").textContent = fmtKm(m.km);
  const cp = paisDe(m.lat, m.lon);
  $("#res-pais-modelo").innerHTML = `Dijo ${bandera(cp, 16)} <b>${nombrePais(cp)}</b> ` +
    (cp && cp === real.pais ? '<span style="color:var(--real);font-weight:700">✓ país</span>' : "");
  animarNumero($("#res-p-modelo"), m.puntos);
  $("#res-b-modelo").style.width = "0";
  requestAnimationFrame(() => requestAnimationFrame(() => { $("#res-b-modelo").style.width = `${(m.puntos / 5000) * 100}%`; }));

  pintarDetalleModelo(globo, m, res.vistas, { etiquetas: res.etiquetas, alAmpliar: abrirVisor });
  pintarHistorial();
  $("#exp-panel").scrollTop = 0;
}

function pintarHistorial() {
  $("#exp-historial-wrap").classList.toggle("oculto", est.capturas.length < 2);
  $("#exp-historial").innerHTML = est.capturas.map((c, i) =>
    `<li data-i="${i}" class="${c === est.activa ? "activa" : ""}">
      <span class="lugar">${bandera(c.real.pais, 16)} ${nombrePais(c.real.pais)}</span>
      <span class="k">${fmtKm(c.modelo.km)}</span><span class="p">${fmt(c.modelo.puntos)} pts</span></li>`).join("");
  $$("#exp-historial li").forEach((li) => li.addEventListener("click", () => {
    const c = est.capturas[+li.dataset.i];
    mostrarCaptura(c);
    if (est.sv && c.real.panoid !== est.sv.getPano()) {
      est.sv.setPano(c.real.panoid);
      est.sv.setPov({ heading: c.heading, pitch: 0 });
    }
  }));
}

/* ------------------------------------------------------------ visor */
function abrirVisor(k) {
  const c = est.activa;
  if (!c) return;
  est.visorK = (k + 4) % 4;
  $("#visor-img").src = c.vistas[est.visorK];
  $("#visor-k").textContent = `Vista ${c.etiquetas[est.visorK]} (heading ${c.headings[est.visorK]}°)`;
  $("#visor").classList.add("visible");
}
const cerrarVisor = () => $("#visor").classList.remove("visible");
$$(".visor-nav").forEach((b) => b.addEventListener("click", (e) => { e.stopPropagation(); abrirVisor(est.visorK + +b.dataset.d); }));
$("#visor").addEventListener("click", (e) => { if (e.target.id === "visor") cerrarVisor(); });

/* ------------------------------------------------------------ botones y teclado */
$("#btn-capturar").addEventListener("click", capturar);
$("#btn-azar").addEventListener("click", lugarAlAzar);
$("#btn-azar-grande").addEventListener("click", lugarAlAzar);
document.addEventListener("keydown", (e) => {
  if ($("#visor").classList.contains("visible")) {
    if (e.key === "Escape") cerrarVisor();
    if (e.key === "ArrowLeft") abrirVisor(est.visorK - 1);
    if (e.key === "ArrowRight") abrirVisor(est.visorK + 1);
    return;
  }
  if (e.key === " " && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
    e.preventDefault();
    capturar();
  }
});

iniciar();
