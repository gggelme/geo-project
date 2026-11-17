/* GeoDuelo: modo duelo. El servidor (servidor.py) elige el panorama, corre el modelo y calcula puntajes. */
"use strict";

const estado = {
  config: { rondas: 5, modo: "libre", tiempo: 0 },
  fase: "inicio", // inicio | cargando | adivinando | resultado | final
  ronda: null, // {token, vistas}
  pin: null, // {lat, lng}
  historial: [], // resultados de la partida
  fijado: false,
  timer: null,
  visorK: 0,
};

/* ------------------------------------------------------------ globo principal */
const contGlobo = $("#globo");
const globo = crearGlobo(contGlobo);
alClicEnGlobo(globo, contGlobo, (c) => { if (estado.fase === "adivinando") ponerPin(c); });

function ponerPin(c) {
  estado.pin = c;
  globo.htmlElementsData([{ ...c, el: crearPin(COLOR.jugador) }]);
  const btn = $("#btn-adivinar");
  btn.disabled = false;
  btn.innerHTML = `Adivinar <kbd>espacio</kbd>`;
}

conectarEstiloMapa(globo, "#estilo-mapa");

$("#btn-fijar").addEventListener("click", () => {
  estado.fijado = !estado.fijado;
  $("#btn-fijar").classList.toggle("activo", estado.fijado);
  if (estado.fase !== "resultado") modoPanel(estado.fijado ? "grande" : "mini");
});
function modoPanel(m) {
  const p = $("#panel-mapa");
  p.classList.remove("mini", "grande", "resultado-modo");
  p.classList.add(m);
}

/* ------------------------------------------------------------ globo decorativo de inicio */
const contInicio = $("#globo-inicio");
const globoInicio = new Globe(contInicio, { rendererConfig: { antialias: true, alpha: true } })
  .backgroundImageUrl(TEXTURA + "night-sky.png")
  .globeImageUrl(TEXTURA + "earth-blue-marble.jpg")
  .bumpImageUrl(TEXTURA + "earth-topology.png")
  .atmosphereColor("#7aa2ff").atmosphereAltitude(0.2)
  .pointsData([]).pointLat("lat").pointLng("lon").pointColor(() => "#fbbf24").pointAltitude(0.03).pointRadius(0.5)
  .ringsData([]).ringLat("lat").ringLng("lon").ringColor(() => (t) => `rgba(251,191,36,${0.8 * (1 - t)})`)
  .ringMaxRadius(3).ringPropagationSpeed(1.2).ringRepeatPeriod(2400);
globoInicio.controls().autoRotate = true;
globoInicio.controls().autoRotateSpeed = 0.5;
globoInicio.controls().enableZoom = false;
globoInicio.pointOfView({ lat: 12, lng: -40, altitude: 2.1 }, 0);
const ajustarInicio = () => {
  const w = window.innerWidth;
  globoInicio.width(w).height(window.innerHeight);
  // Corre el globo hacia la derecha para que no quede tapado por la tarjeta.
  contInicio.style.transform = w > 900 ? `translateX(${Math.round(w * 0.18)}px)` : "none";
};
window.addEventListener("resize", ajustarInicio);
ajustarInicio();

api("/api/info").then((info) => {
  globoInicio.pointsData(info.celdas).ringsData(info.celdas.filter((_, i) => i % 3 === 0));
  $("#historial-global").dataset.base =
    `${fmt(info.panoramas)} panoramas de test · ${info.paises} países · ${info.celdas.length} celdas`;
  pintarHistorialGlobal();
}).catch(() => toast("No se pudo hablar con el servidor"));

function pintarHistorialGlobal() {
  const h = leer("geoduelo.historial", { partidas: 0, ganadas: 0, mejor: 0 });
  const base = $("#historial-global").dataset.base || "";
  const extra = h.partidas
    ? ` · Le ganaste <b>${h.ganadas}</b> de <b>${h.partidas}</b> partidas · récord <b>${fmt(h.mejor)}</b>`
    : "";
  $("#historial-global").innerHTML = base + extra;
}

/* ------------------------------------------------------------ opciones de inicio */
$$(".segmentado[data-opcion]").forEach((grupo) => {
  const clave = grupo.dataset.opcion;
  grupo.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    grupo.querySelectorAll("button").forEach((x) => x.classList.toggle("sel", x === b));
    estado.config[clave] = clave === "modo" ? b.dataset.valor : Number(b.dataset.valor);
    guardar("geoduelo.config", estado.config);
  }));
});
{
  const c = leer("geoduelo.config", null);
  if (c) for (const [k, v] of Object.entries(c)) {
    $(`.segmentado[data-opcion="${k}"] button[data-valor="${v}"]`)?.click();
  }
}

function mostrarPantalla(id) {
  $$(".pantalla").forEach((p) => p.classList.toggle("activa", p.id === id));
  if (id === "inicio") { globoInicio.resumeAnimation(); pintarHistorialGlobal(); } else globoInicio.pauseAnimation();
  if (id === "juego") globo.resumeAnimation(); else globo.pauseAnimation();
}

/* ------------------------------------------------------------ partida */
function empezarPartida() {
  estado.historial = [];
  actualizarHud();
  mostrarPantalla("juego");
  siguienteRonda();
}

function actualizarHud() {
  const n = estado.historial.length + (estado.fase === "resultado" || estado.fase === "final" ? 0 : 1);
  const total = estado.config.rondas || "∞";
  $("#hud-ronda").textContent = `${Math.max(1, n)} / ${total}`;
  $("#hud-jugador").textContent = fmt(estado.historial.reduce((s, r) => s + r.jugador.puntos, 0));
  $("#hud-modelo").textContent = fmt(estado.historial.reduce((s, r) => s + r.modelo.puntos, 0));
}

async function siguienteRonda() {
  if (estado.fase === "cargando" || estado.fase === "enviando") return;
  document.activeElement?.blur();
  const { rondas } = estado.config;
  if (rondas && estado.historial.length >= rondas) return terminarPartida();
  estado.fase = "cargando";
  estado.pin = null;
  estado.ronda = null;
  actualizarHud();
  $("#resultado").classList.remove("visible");
  modoPanel(estado.fijado ? "grande" : "mini");
  globo.htmlElementsData([]).arcsData([]).pointsData([]).ringsData([]);
  globo.pointOfView({ lat: 15, lng: -30, altitude: 2.6 }, 900);
  const btn = $("#btn-adivinar");
  btn.disabled = true;
  btn.textContent = "Poné el pin en el globo";
  $$(".vista img").forEach((img) => { img.classList.remove("lista"); img.removeAttribute("src"); });
  $("#cargando").classList.add("visible");

  let ronda;
  try {
    ronda = await api(`/api/ronda?modo=${estado.config.modo}`);
  } catch (e) {
    toast(`Error pidiendo ronda: ${e.message}. Reintentando…`);
    estado.fase = "error";
    setTimeout(siguienteRonda, 1500);
    return;
  }
  estado.ronda = ronda;
  await Promise.all($$(".vista img").map((img, k) => new Promise((ok) => {
    img.onload = () => { img.classList.add("lista"); ok(); };
    img.onerror = ok;
    img.src = ronda.vistas[k];
  })));
  $("#cargando").classList.remove("visible");
  estado.fase = "adivinando";
  arrancarTimer();
}

function arrancarTimer() {
  clearInterval(estado.timer);
  const barra = $("#barra-tiempo");
  const seg = estado.config.tiempo;
  barra.classList.toggle("visible", seg > 0);
  if (!seg) return;
  const fin = performance.now() + seg * 1000;
  const tick = () => {
    const resta = Math.max(0, fin - performance.now()) / 1000;
    barra.querySelector("div").style.transform = `scaleX(${resta / seg})`;
    $("#texto-tiempo").textContent = `${Math.ceil(resta)} s`;
    barra.classList.toggle("urgente", resta < 10);
    if (resta <= 0) {
      clearInterval(estado.timer);
      if (estado.fase === "adivinando") { toast("¡Se acabó el tiempo!"); adivinar(); }
    }
  };
  tick();
  estado.timer = setInterval(tick, 200);
}

async function adivinar() {
  if (estado.fase !== "adivinando") return;
  document.activeElement?.blur();
  estado.fase = "enviando";
  clearInterval(estado.timer);
  $("#barra-tiempo").classList.remove("visible");
  const pin = estado.pin;
  let res;
  try {
    res = await api("/api/adivinar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: estado.ronda.token, lat: pin?.lat ?? null, lon: pin?.lng ?? null }),
    });
  } catch (e) {
    toast(`Error: ${e.message}`);
    estado.fase = "adivinando";
    return;
  }
  res.vistas = estado.ronda.vistas;
  estado.historial.push(res);
  estado.fase = "resultado";
  mostrarResultado(res);
}

function mostrarResultado(res) {
  const { real, jugador: j, modelo: m } = res;
  actualizarHud();
  modoPanel("resultado-modo");

  // --- Globo
  const pins = [{ lat: real.lat, lng: real.lon, el: crearPin(COLOR.real, "Real") },
    { lat: m.lat, lng: m.lon, el: crearPin(COLOR.modelo, "Modelo") }];
  const arcos = [{ startLat: m.lat, startLng: m.lon, endLat: real.lat, endLng: real.lon, color: [COLOR.modelo, COLOR.real] }];
  if (j.lat != null) {
    pins.push({ lat: j.lat, lng: j.lon, el: crearPin(COLOR.jugador, "Vos") });
    arcos.push({ startLat: j.lat, startLng: j.lon, endLat: real.lat, endLng: real.lon, color: [COLOR.jugador, COLOR.real] });
  }
  globo.htmlElementsData(pins).arcsData(arcos).pointsData(m.distribucion).ringsData([{ lat: real.lat, lng: real.lon }]);
  const encuadre = pins.map((p) => ({ lat: p.lat, lng: p.lng }));
  const maxKm = Math.max(m.km, j.km ?? 0);
  globo.pointOfView({ ...puntoMedio(encuadre), altitude: Math.min(2.8, Math.max(0.35, maxKm / 3200)) }, 1600);

  // --- Tarjeta
  const gana = j.puntos > m.puntos ? "jugador" : j.puntos < m.puntos ? "modelo" : "empate";
  const titulo = $("#res-titulo");
  titulo.className = "res-titulo " + (gana === "empate" ? "" : `gana-${gana}`);
  titulo.textContent = j.lat == null ? "Sin respuesta" :
    gana === "jugador" ? (j.puntos === 5000 ? "¡Perfecto! 🎯" : "¡Le ganaste a la red!") :
    gana === "modelo" ? "Esta la gana el modelo" : "Empate";
  const codigoReal = real.pais;
  $("#res-lugar").innerHTML = `${bandera(codigoReal, 28)} <span>Era en <b>${nombrePais(codigoReal)}</b>
    <span style="color:var(--texto-3)">· ${real.lat.toFixed(3)}, ${real.lon.toFixed(3)}</span></span>`;

  $$(".duelo-col").forEach((c) => c.classList.toggle("ganador", c.classList.contains(gana)));
  for (const [quien, r] of [["jugador", j], ["modelo", m]]) {
    animarNumero($(`#res-p-${quien}`), r.puntos);
    $(`#res-b-${quien}`).style.width = "0";
    requestAnimationFrame(() => requestAnimationFrame(() => { $(`#res-b-${quien}`).style.width = `${(r.puntos / 5000) * 100}%`; }));
    $(`#res-km-${quien}`).innerHTML = r.km == null ? "No pusiste pin" : `a <b>${fmtKm(r.km)}</b>`;
    const cp = r.lat == null ? null : paisDe(r.lat, r.lon);
    $(`#res-pais-${quien}`).innerHTML = r.lat == null ? "" :
      `${bandera(cp)} ${nombrePais(cp)} ${cp && cp === codigoReal ? '<span class="ok">✓ país</span>' : ""}`;
  }

  pintarDetalleModelo(globo, m, res.vistas, { alAmpliar: abrirVisor });

  const ultima = estado.config.rondas && estado.historial.length >= estado.config.rondas;
  $("#btn-siguiente").innerHTML = (ultima ? "Ver resultado final" : "Siguiente ronda") + " <kbd>espacio</kbd>";
  $("#resultado").scrollTop = 0;
  $("#resultado").classList.add("visible");
}

/* ------------------------------------------------------------ final */
function terminarPartida() {
  clearInterval(estado.timer);
  const h = estado.historial;
  if (!h.length) { estado.fase = "inicio"; mostrarPantalla("inicio"); return; }
  estado.fase = "final";
  const tj = h.reduce((s, r) => s + r.jugador.puntos, 0);
  const tm = h.reduce((s, r) => s + r.modelo.puntos, 0);
  const v = $("#fin-veredicto");
  v.textContent = tj > tm ? "¡Le ganaste a la red neuronal!" : tj < tm ? "Ganó la red neuronal" : "Empate técnico";
  v.style.color = tj > tm ? COLOR.jugador : tj < tm ? COLOR.modelo : "";
  animarNumero($("#fin-jugador"), tj, 1400);
  animarNumero($("#fin-modelo"), tm, 1400);
  $("#fin-tabla").innerHTML = h.map((r, i) => {
    const gj = r.jugador.puntos > r.modelo.puntos, gm = r.modelo.puntos > r.jugador.puntos;
    return `<tr><td class="num">${i + 1}</td><td>${bandera(r.real.pais, 18)} ${nombrePais(r.real.pais)}</td>
      <td class="num ${gj ? "gana j" : ""}">${fmt(r.jugador.puntos)}</td><td class="km">${fmtKm(r.jugador.km)}</td>
      <td class="num ${gm ? "gana m" : ""}">${fmt(r.modelo.puntos)}</td><td class="km">${fmtKm(r.modelo.km)}</td></tr>`;
  }).join("");

  const g = leer("geoduelo.historial", { partidas: 0, ganadas: 0, mejor: 0 });
  g.partidas += 1;
  g.ganadas += tj > tm ? 1 : 0;
  if (estado.config.rondas === 5) g.mejor = Math.max(g.mejor, tj);
  guardar("geoduelo.historial", g);
  mostrarPantalla("final");
}

/* ------------------------------------------------------------ visor */
function abrirVisor(k) {
  if (!estado.ronda) return;
  estado.visorK = (k + 4) % 4;
  $("#visor-img").src = estado.ronda.vistas[estado.visorK];
  $("#visor-k").textContent = `Vista ${estado.visorK + 1} de 4`;
  $("#visor").classList.add("visible");
}
const cerrarVisor = () => $("#visor").classList.remove("visible");
$$(".vista").forEach((v) => v.addEventListener("click", () => abrirVisor(+v.dataset.k)));
$$(".visor-nav").forEach((b) => b.addEventListener("click", (e) => { e.stopPropagation(); abrirVisor(estado.visorK + +b.dataset.d); }));
$("#visor").addEventListener("click", (e) => { if (e.target.id === "visor") cerrarVisor(); });

/* ------------------------------------------------------------ botones y teclado */
$("#btn-empezar").addEventListener("click", empezarPartida);
$("#btn-revancha").addEventListener("click", empezarPartida);
$("#btn-inicio").addEventListener("click", () => { estado.fase = "inicio"; mostrarPantalla("inicio"); });
$("#btn-adivinar").addEventListener("click", adivinar);
$("#btn-siguiente").addEventListener("click", siguienteRonda);
$("#btn-salir").addEventListener("click", () => {
  cerrarVisor();
  $("#resultado").classList.remove("visible");
  if (estado.historial.length) terminarPartida();
  else { estado.fase = "inicio"; clearInterval(estado.timer); mostrarPantalla("inicio"); }
});

document.addEventListener("keydown", (e) => {
  const visor = $("#visor").classList.contains("visible");
  if (visor) {
    if (e.key === "Escape") cerrarVisor();
    if (e.key === "ArrowLeft") abrirVisor(estado.visorK - 1);
    if (e.key === "ArrowRight") abrirVisor(estado.visorK + 1);
    return;
  }
  if (e.key === " " || e.key === "Enter") {
    if (estado.fase === "adivinando" && estado.pin) { e.preventDefault(); adivinar(); }
    else if (estado.fase === "resultado") { e.preventDefault(); siguienteRonda(); }
    else if (estado.fase === "inicio" && e.key === "Enter") { e.preventDefault(); empezarPartida(); }
  }
  if (estado.fase === "adivinando" && ["1", "2", "3", "4"].includes(e.key)) abrirVisor(+e.key - 1);
});

mostrarPantalla("inicio");
