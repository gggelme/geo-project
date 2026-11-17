// Genera presentacion_geoguessr.pptx a partir de figuras/ y figuras/datos_graficos.json.
// Uso (desde presentacion/): npm install && node armar_pptx.js
// Antes: ../.venv/Scripts/python.exe armar_figuras.py (genera las figuras y los datos).
// Tamaños pensados para proyectar en un aula: texto >= 16 pt, epígrafes >= 14 pt.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
const { applyTheme } = require("./apply_theme.js");

const FIG = path.join(__dirname, "figuras");
const D = JSON.parse(fs.readFileSync(path.join(FIG, "datos_graficos.json"), "utf8"));

// ---------------------------------------------------------------- tema
const THEME = {
  name: "Geolocalizacion",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16202B", lt1: "FFFFFF", dk2: "10263B", lt2: "EEF2F6",
    accent1: "E4572E", accent2: "2A78D6", accent3: "1BAF7A", accent4: "EDA100",
    accent5: "4F5B69", accent6: "7A8591", hlink: "2A78D6", folHlink: "4F5B69",
  },
};
const HEX = { tinta: "16202B", navy: "10263B", panel: "EEF2F6", pin: "E4572E", azul: "2A78D6", verde: "1BAF7A",
  amarillo: "EDA100", gris: "4F5B69", tenue: "8A94A0", blanco: "FFFFFF", grilla: "E3E7EC", naranja: "EB6834",
  claroOscuro: "CADCEB" };
const MODELOS = [["Celdas CE", HEX.azul], ["Celdas CE + aug", HEX.naranja], ["Celdas completo", HEX.verde],
  ["Países 101", HEX.amarillo]];
const NOMBRE_PAIS = { US: "Estados Unidos", IN: "India", BR: "Brasil", FR: "Francia", CA: "Canadá", AU: "Australia",
  RU: "Rusia", TH: "Tailandia", ID: "Indonesia", JP: "Japón", MX: "México", IT: "Italia" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Geolocalización de imágenes tipo GeoGuessr";
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, M = 0.6;

// ---------------------------------------------------------------- utilidades
function tamanoImagen(archivo) {
  const b = fs.readFileSync(archivo);
  if (b[0] === 0x89) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
  let i = 2;
  while (i < b.length) {
    const marca = b[i + 1], largo = b.readUInt16BE(i + 2);
    if (marca >= 0xc0 && marca <= 0xc3) return { h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7) };
    i += 2 + largo;
  }
  throw new Error("No se pudo leer el tamaño de " + archivo);
}
// Imagen entera dentro de la caja (x, y, w, h); devuelve el rectángulo ocupado.
function imagen(slide, nombre, x, y, w, h, opciones = {}) {
  const archivo = path.join(FIG, nombre);
  const t = tamanoImagen(archivo);
  const escala = Math.min(w / t.w, h / t.h);
  const iw = t.w * escala, ih = t.h * escala;
  const ix = opciones.alinear === "izq" ? x : x + (w - iw) / 2;
  const iy = opciones.alinearV === "arriba" ? y : y + (h - ih) / 2;
  slide.addImage({ path: archivo, x: ix, y: iy, w: iw, h: ih, altText: opciones.alt || nombre,
    objectName: opciones.nombre || nombre });
  return { x: ix, y: iy, w: iw, h: ih };
}
// Foto recortada para llenar la caja.
function foto(slide, nombre, x, y, w, h) {
  slide.addImage({ path: path.join(FIG, nombre), x, y, w, h, sizing: { type: "cover", w, h }, altText: nombre,
    objectName: nombre });
}
function texto(slide, contenido, opciones) {
  slide.addText(contenido, { isTextBox: true, fontSize: 18, color: C.text1, margin: 0, valign: "top", ...opciones });
}
// Cifra grande con etiqueta debajo.
function cifra(slide, valor, etiqueta, x, y, w, color = C.accent1, tam = 44) {
  texto(slide, valor, { x, y, w, h: 0.8, fontSize: tam, bold: true, color, valign: "bottom" });
  texto(slide, etiqueta, { x, y: y + 0.85, w, h: 0.7, fontSize: 17, color: C.text2 });
}
// Círculo numerado (motivo visual: el "pin" del mapa).
function pin(slide, n, x, y, d = 0.46, color = C.accent1) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 },
    objectName: `pin ${n}` });
  texto(slide, String(n), { x, y, w: d, h: d, fontSize: 18, bold: true, color: C.background1, align: "center",
    valign: "middle" });
}
function tarjeta(slide, x, y, w, h, color = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color },
    line: { color, width: 0 }, objectName: "tarjeta" });
}
function bullets(items, tam = 18) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1,
    paraSpaceAfter: 10, fontSize: tam } }));
}
const coma = (v) => String(v).replace(".", ",");
const fmt = (n) => n.toLocaleString("es-AR");
const EJES = { catAxisLabelColor: HEX.tinta, valAxisLabelColor: HEX.gris, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  catAxisLabelFontSize: 15, valAxisLabelFontSize: 14, dataLabelFontSize: 16, dataLabelColor: HEX.tinta,
  titleFontSize: 17, titleColor: HEX.tinta, legendFontSize: 15,
  valGridLine: { color: HEX.grilla, size: 0.75 }, catGridLine: { style: "none" }, catAxisLineShow: false,
  valAxisLineShow: false };

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "PORTADA", background: { color: THEME.colors.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.9, w: 6.3, h: 2.1, fontSize: 48, bold: true,
      color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.2, w: 6.1, h: 1.3, fontSize: 24,
      color: HEX.claroOscuro, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENIDO", background: { color: THEME.colors.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.3, w: W - 2 * M, h: 0.9, fontSize: 34,
      bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: H - 0.45, w: 0.5, h: 0.3, fontSize: 12, color: HEX.tenue, align: "right" },
});
pres.defineSlideMaster({
  title: "CIERRE", background: { color: THEME.colors.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: W - 2 * M, h: 0.9, fontSize: 38,
      bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
  ],
});
function nueva(titulo, seccion, layout = "CONTENIDO") {
  const s = pres.addSlide({ masterName: layout, sectionTitle: seccion });
  if (titulo) s.addText(titulo, { placeholder: "title" });
  return s;
}
const ds = D.dataset, test = D.test;
const g1 = test["Celdas CE"]["1 vista"], g4 = test["Celdas CE"]["4 vistas"];
const ganancia4 = fmt(g4.puntaje - g1.puntaje);
const of20 = D.overfitting["20k"], of2m = D.overfitting["2m"];

// ================================================================ 1. Portada
pres.addSection({ title: "Introducción" });
let s = nueva("¿Dónde fue tomada esta foto?", "Introducción", "PORTADA");
s.addText("Geolocalización de imágenes de Street View con redes convolucionales", { placeholder: "body" });
texto(s, "Redes Neuronales 2 · Trabajo final\nGrupo: [completar integrantes]", { x: M, y: 6.1, w: 6.2, h: 0.8,
  fontSize: 16, color: HEX.claroOscuro });
{
  const x0 = 7.25, y0 = 0.6, cw = 2.6, ch = 2.0, g = 0.1;
  for (let k = 0; k < 6; k++) foto(s, `portada_${k + 1}.jpg`, x0 + (k % 2) * (cw + g), y0 + Math.floor(k / 2) * (ch + g),
    cw, ch);
}
s.addNotes("[0:15] Presentación del grupo. Arrancamos con una pregunta: ¿dónde fue tomada esta foto? Ese es el " +
  "problema: dada una imagen de Street View, predecir en qué lugar del mundo fue tomada. Las fotos son del conjunto " +
  "de test: Noruega, Brasil, Tailandia, Senegal, EE. UU. y Ecuador.");

// ================================================================ 2. GeoGuessr
s = nueva("GeoGuessr: adiviná dónde estás", "Introducción");
foto(s, "ronda_foto.jpg", M, 1.4, 4.3, 4.3);
texto(s, "1. Te muestran una foto", { x: M, y: 5.85, w: 4.3, h: 0.45, fontSize: 18, bold: true });
imagen(s, "ronda_mapa.png", 5.1, 1.4, 4.3, 4.3);
texto(s, "2. Marcás un punto en el mapa", { x: 5.1, y: 5.85, w: 4.3, h: 0.45, fontSize: 18, bold: true });
tarjeta(s, 9.75, 1.4, 3.0, 4.9);
texto(s, "3. Puntaje según la distancia", { x: 10.0, y: 1.6, w: 2.6, h: 0.8, fontSize: 18, bold: true });
texto(s, "5000 · e^(−d / 1493 km)", { x: 10.0, y: 2.45, w: 2.6, h: 0.45, fontSize: 16, color: C.accent5,
  italic: true });
cifra(s, `${D.ronda.km} km`, "error del modelo en esta foto (Brasil)", 10.0, 3.0, 2.6, C.accent1, 40);
cifra(s, fmt(D.ronda.puntaje), "puntos (sobre 5000)", 10.0, 4.6, 2.6, C.text2, 40);
s.addNotes("[0:45] GeoGuessr es un juego online con decenas de millones de jugadores: te ponen en un lugar al azar " +
  "de Street View y tenés que marcar en el mapa dónde creés que estás. Cuanto más cerca, más puntos: 5000 si acertás " +
  "exacto, y el puntaje cae exponencialmente con la distancia. Esta es una ronda real con nuestro modelo final: una " +
  "foto de Brasil del conjunto de test; el modelo erró por 318 km y habría sacado 4.040 puntos. Por eso nuestra " +
  "métrica principal es la distancia en km (y el puntaje de GeoGuessr), no solo el accuracy.");

// ================================================================ 3. PIGEON
s = nueva("Estado del arte: PIGEON (Stanford, CVPR 2024)", "Introducción");
{
  const r = imagen(s, "pigeon_titulo.png", M, 1.35, 5.9, 5.45, { alinear: "izq", alt: "Primera página del paper PIGEON" });
  texto(s, "Haas, Skreta, Alberti y Finn · arXiv:2307.05845", { x: M, y: r.y + r.h + 0.08, w: 6, h: 0.35,
    fontSize: 14, color: C.accent6 });
}
[["Celdas geográficas", "El mundo se divide en regiones y se predice la región: clasificación, no regresión."],
  ["Haversine label smoothing", "La etiqueta reparte probabilidad entre las celdas cercanas, según la distancia."],
  ["4 vistas por lugar", "Usa las 4 imágenes del panorama para decidir."]].forEach(([t, d], i) => {
  const y = 1.4 + i * 1.33;
  pin(s, i + 1, 6.95, y + 0.02);
  texto(s, t, { x: 7.6, y, w: 5.15, h: 0.45, fontSize: 21, bold: true });
  texto(s, d, { x: 7.6, y: y + 0.48, w: 5.15, h: 0.8, fontSize: 17, color: C.accent5 });
});
tarjeta(s, 6.95, 5.45, 5.8, 1.4);
texto(s, [{ text: "Resultado: ", options: { bold: true } },
  { text: "le ganó 6 de 6 partidas a un jugador profesional; 40% de sus predicciones a menos de 25 km." }],
{ x: 7.2, y: 5.55, w: 5.3, h: 1.2, fontSize: 18, valign: "middle" });
s.addNotes("[0:45] Nuestra referencia es PIGEON, de Stanford, el estado del arte en geolocalización tipo GeoGuessr. " +
  "Tomamos tres ideas: (1) dividir el mundo en celdas geográficas y plantearlo como clasificación; (2) la loss con " +
  "haversine label smoothing, que no castiga igual equivocarse por la celda vecina que por otro continente; (3) usar " +
  "las 4 vistas de cada panorama. PIGEON usa CLIP preentrenado y muchísimo cómputo; nosotros lo hicimos con una " +
  "ResNet-18 y una GPU de 4 GB.");

// ================================================================ 4. Dataset
pres.addSection({ title: "Dataset y EDA" });
s = nueva("Dataset: World Street View 500k", "Dataset y EDA");
[[`${coma((ds.imagenes / 1e6).toFixed(2))} M`, "imágenes"], [`${Math.round(ds.panoramas / 1000)} k`, "panoramas"],
  [String(ds.paises), "países"], ["~185 GB", "en 393 archivos"]]
  .forEach(([v, e], i) => cifra(s, v, e, M + i * 3.05, 1.25, 2.85, i === 0 ? C.accent1 : C.text2, 46));
{
  const y = 3.25, cw = 2.9, g = 0.13;
  for (let k = 0; k < 4; k++) foto(s, `vista_${k + 1}.jpg`, M + k * (cw + g), y, cw, cw * 0.72);
  texto(s, "Un panorama = 4 vistas del mismo lugar, mirando en 4 direcciones (Noruega, test)",
    { x: M, y: y + cw * 0.72 + 0.15, w: 12.1, h: 0.45, fontSize: 17, color: C.accent5 });
  texto(s, "Separamos siempre por panorama: las 4 vistas quedan en el mismo conjunto.",
    { x: M, y: y + cw * 0.72 + 0.6, w: 12.1, h: 0.45, fontSize: 19, bold: true });
}
s.addNotes("[0:45] Usamos el dataset World Street View 500k de Hugging Face: casi 2 millones de imágenes de 117 " +
  "países, unos 185 GB. Lo importante es la estructura: son 500 mil panoramas, y cada uno tiene 4 fotos del mismo " +
  "lugar mirando en distintas direcciones. Eso tiene dos consecuencias: (1) las particiones siempre se hacen por " +
  "panorama, para que las vistas de un mismo lugar no queden repartidas entre train y test; (2) al final podemos " +
  "combinar las 4 vistas para predecir mejor.");

// ================================================================ 5. EDA desbalance
const pt = D.paises_top;
s = nueva(`Muy desbalanceado: 3 países tienen el ${coma(pt.top3)}% de las imágenes`, "Dataset y EDA");
s.addChart(pres.charts.BAR, [{ name: "% de imágenes", labels: pt.etiquetas.map((c) => `${NOMBRE_PAIS[c] || c} (${c})`),
  values: pt.valores }], {
  x: M - 0.1, y: 1.3, w: 5.3, h: 5.6, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [HEX.azul],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", valAxisHidden: true,
  showLegend: false, barGapWidthPct: 40, showTitle: true, title: "% de las imágenes", valAxisMaxVal: 30,
  ...EJES, catAxisLabelFontSize: 14, dataLabelFontSize: 14, objectName: "barras países" });
imagen(s, "eda_mapa_densidad.png", 5.45, 1.4, 7.35, 4.5, { alt: "Mapa de densidad de imágenes" });
texto(s, [{ text: `${pt.paises_menos_1000} países `, options: { bold: true, color: C.accent1 } },
  { text: "tienen menos de 1.000 imágenes. África y gran parte de Asia casi no aparecen." }],
{ x: 5.65, y: 6.05, w: 7.1, h: 0.85, fontSize: 18 });
s.addNotes("[0:45] El dataset está muy desbalanceado: Estados Unidos, India y Brasil juntan el 44% de las imágenes, " +
  "y 46 países tienen menos de mil. En el mapa se ve que la cobertura depende de dónde hay Street View: casi nada en " +
  "África, China o Medio Oriente. Como el objetivo es jugar GeoGuessr, donde los lugares no salen con esta " +
  "distribución, decidimos entrenar balanceado por región, para que el modelo no aprenda el sesgo del dataset.");

// ================================================================ 6. PCA
s = nueva("¿Alcanza con features simples para separar regiones?", "Dataset y EDA");
imagen(s, "pca_pixeles.png", M, 1.3, 6.0, 4.3, { alt: "PCA 2D de píxeles" });
imagen(s, "pca_clip.png", 6.85, 1.3, 5.9, 4.3, { alt: "PCA 2D de embeddings CLIP" });
texto(s, [{ text: "Píxeles: ", options: { bold: true } }, { text: "las regiones se superponen por completo." }],
  { x: M, y: 5.7, w: 6, h: 0.5, fontSize: 18 });
texto(s, [{ text: "Embeddings de CLIP: ", options: { bold: true } },
  { text: "aparece algo de estructura (América vs. Asia)." }], { x: 6.85, y: 5.7, w: 5.9, h: 0.5, fontSize: 18 });
tarjeta(s, M, 6.3, W - 2 * M, 0.7);
texto(s, "→ Hace falta aprender las features: una CNN entrenada de punta a punta.", { x: M + 0.25, y: 6.3,
  w: W - 2 * M - 0.5, h: 0.7, fontSize: 20, bold: true, valign: "middle" });
s.addNotes("[0:30] En el EDA probamos si con features simples se separan las regiones. El color de cada punto indica " +
  "la región (ver el mapa de referencia). Con los píxeles crudos (imágenes de 32x32), el PCA no muestra ninguna " +
  "estructura geográfica. Con embeddings de una red preentrenada (CLIP) ya se empiezan a separar América y Asia. " +
  "Conclusión: necesitamos una red profunda que aprenda sus propias features.");

// ================================================================ 7. Restricciones
pres.addSection({ title: "Recorrido" });
s = nueva("Restricción técnica: una GTX 1650 con 4 GB", "Recorrido");
tarjeta(s, M, 1.4, 5.8, 2.2);
texto(s, "Arquitectura", { x: M + 0.3, y: 1.55, w: 5.2, h: 0.45, fontSize: 21, bold: true });
cifra(s, "11,7 M", "parámetros · ResNet-18 ✔", M + 0.3, 1.9, 2.7, C.accent3, 42);
cifra(s, "138 M", "parámetros · VGG16 ✘", M + 3.05, 1.9, 2.6, C.accent6, 42);
texto(s, "VGG16 tiene 12 veces más parámetros: demasiado pesada para 4 GB de memoria y para iterar rápido. La " +
  "ResNet además entrena mejor gracias a las conexiones residuales.", { x: M, y: 3.85, w: 5.8, h: 2.0, fontSize: 18,
  color: C.accent5 });
tarjeta(s, 6.9, 1.4, 5.83, 2.2);
texto(s, "Trabajar sobre muestras", { x: 7.2, y: 1.55, w: 5.3, h: 0.45, fontSize: 21, bold: true });
cifra(s, "20 k", "imágenes de train, balanceadas", 7.2, 1.9, 2.7, C.accent1, 42);
cifra(s, "5,3 k", "imágenes de validación fija", 9.95, 1.9, 2.6, C.text2, 42);
texto(s, bullets(["Train balanceado: la misma cantidad de imágenes por región.",
  "Validación con la distribución real del dataset.", "Cada experimento: menos de 2 h en la GPU local."]),
{ x: 6.9, y: 3.85, w: 5.85, h: 2.6, color: C.accent5 });
s.addNotes("[0:45] La restricción principal fue el hardware: teníamos una GTX 1650 con 4 GB. Por eso elegimos " +
  "ResNet-18: 11,7 millones de parámetros, contra 138 millones de VGG16, que descartamos. Y no podíamos entrenar " +
  "con los 2 millones de imágenes en cada prueba, así que trabajamos sobre muestras: un train de 20 mil imágenes " +
  "balanceado por región y una validación fija de 5.300 imágenes con la distribución real, que usamos para comparar " +
  "todos los experimentos. Cada corrida tardaba menos de 2 horas.");

// ================================================================ 8. País vs celdas
const pvc = D.pais_vs_celdas_20k;
s = nueva("¿Clasificar países o celdas geográficas?", "Recorrido");
const rc = imagen(s, "mapa_celdas.png", M, 1.35, 7.9, 4.6, { alinear: "izq", alinearV: "arriba",
  alt: "Mapa de las 70 celdas" });
texto(s, "70 celdas con K-means esférico sobre las coordenadas", { x: M, y: rc.y + rc.h + 0.1, w: 7.9, h: 0.4,
  fontSize: 16, color: C.accent5 });
texto(s, "Nos quedamos con celdas", { x: 8.8, y: 1.35, w: 3.95, h: 0.5, fontSize: 23, bold: true, color: C.accent1 });
texto(s, bullets(["Granularidad: divide los países grandes (EE. UU. = 12 celdas, Brasil = 7) y agrupa los chicos.",
  "Menos error en km con el mismo accuracy."], 17), { x: 8.8, y: 1.95, w: 3.95, h: 2.3 });
tarjeta(s, 8.8, 4.3, 3.95, 2.6);
texto(s, `Prueba con 20k (misma muestra y misma validación de ${fmt(pvc.imagenes_val)} imágenes)`,
  { x: 9.0, y: 4.4, w: 3.6, h: 0.6, fontSize: 14, color: C.accent5 });
s.addTable([
  [{ text: "", options: {} }, { text: "Países", options: { bold: true } }, { text: "Celdas", options: { bold: true } }],
  ["accuracy", `${coma(pvc.acc_pais)}%`, `${coma(pvc.acc_celdas)}%`],
  ["km mediana", fmt(pvc.km_pais), { text: fmt(pvc.km_celdas), options: { bold: true, color: HEX.verde } }],
], { x: 9.0, y: 5.1, w: 3.55, colW: [1.45, 1.05, 1.05], fontSize: 17, color: HEX.tinta, align: "center",
  rowH: 0.48, border: { type: "solid", color: "D5DBE2", pt: 0.75 }, objectName: "tabla país vs celdas" });
s.addNotes("[1:00] Primera decisión: ¿clasificar por país o por regiones geográficas? Probamos las dos con la misma " +
  "muestra de 20k y la misma validación. Las celdas salen de un K-means esférico sobre las coordenadas: 70 regiones. " +
  "Nos quedamos con celdas por dos razones. Granularidad: EE. UU. queda dividido en 12 celdas y Brasil en 7, " +
  "mientras que países chicos vecinos comparten celda. Y rendimiento en distancia: con accuracy casi igual (28,4% " +
  "contra 27,5%), el error mediano fue 1.820 km con países contra 1.629 km con celdas.");

// ================================================================ 9. Más datos
s = nueva("Lección temprana: más datos, mucho mejor", "Recorrido");
s.addChart(pres.charts.BAR, [{ name: "Accuracy de país", labels: ["20k imágenes", "50k imágenes"],
  values: D.mas_datos_paises.valores }], {
  x: M, y: 1.4, w: 5.6, h: 5.3, barDir: "col", chartColors: [HEX.azul], showValue: true, dataLabelPosition: "outEnd",
  dataLabelFormatCode: "0.0\"%\"", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 50, showLegend: false,
  barGapWidthPct: 70, showTitle: true, title: "Clasificador de países (ResNet-18 desde cero)", ...EJES,
  catAxisLabelFontSize: 17, dataLabelFontSize: 20, objectName: "barras más datos" });
cifra(s, "+12 pts", "de accuracy con 2,5 veces más imágenes", 7.0, 1.8, 5.7, C.accent1, 60);
texto(s, "Ya en las primeras pruebas se vio que el modelo estaba limitado por la cantidad de datos. Esa idea marcó " +
  "todo lo que siguió: regularizar mientras trabajábamos con muestras chicas y, al final, usar el dataset completo.",
{ x: 7.0, y: 3.6, w: 5.7, h: 2.6, fontSize: 20, color: C.accent5 });
s.addNotes("[0:30] Una lección temprana: con el mismo clasificador de países, pasar de 20 mil a 50 mil imágenes " +
  "subió el accuracy de 28% a 41%. El modelo estaba muy limitado por la cantidad de datos, y eso guió todo lo que " +
  "vino después.");

// ================================================================ 10. Recorrido 20k
const rec = D.recorrido_20k;
const etiquetasRec = ["Desde cero", "+ ImageNet", "Celdas más finas", "+ Loss haversine", "+ Augmentation",
  "+ Snapshot ensemble", "+ SAM", "SAM + snapshots"];
s = nueva("Con la muestra de 20k: cada técnica, un paso adelante", "Recorrido");
s.addChart(pres.charts.BAR, [{ name: "km mediana", labels: etiquetasRec, values: rec.km }], {
  x: M - 0.1, y: 1.3, w: 8.3, h: 5.35, barDir: "bar", catAxisOrientation: "maxMin",
  chartColors: [HEX.azul, HEX.azul, HEX.tenue, HEX.azul, HEX.azul, HEX.azul, HEX.azul, HEX.pin], showValue: true,
  dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0\" km\"", valAxisHidden: true, valAxisMinVal: 0,
  valAxisMaxVal: 2700, showLegend: false, barGapWidthPct: 30, showTitle: true,
  title: "Error mediano sobre la validación fija (menos es mejor)", ...EJES, catAxisLabelFontSize: 16,
  objectName: "barras recorrido" });
texto(s, `Accuracy: ${coma(rec.top1[0])}% → ${coma(rec.top1.at(-1))}%`, { x: M, y: 6.7, w: 8, h: 0.4, fontSize: 16,
  color: C.accent5 });
tarjeta(s, 8.95, 1.3, 3.78, 5.65);
texto(s, "El problema: overfitting", { x: 9.15, y: 1.42, w: 3.4, h: 0.5, fontSize: 21, bold: true, color: C.accent1 });
s.addChart(pres.charts.LINE, [{ name: "train", labels: of20.x.map(String), values: of20.train },
  { name: "validación", labels: of20.x.map(String), values: of20.val }], {
  x: 9.05, y: 1.95, w: 3.6, h: 2.55, chartColors: [HEX.tenue, HEX.pin], lineSize: 3, lineDataSymbol: "none",
  showLegend: true, legendPos: "b", valAxisMaxVal: 100, valAxisMinVal: 0, showTitle: true,
  title: "accuracy (%) por época", ...EJES, titleFontSize: 14, catAxisLabelFontSize: 13, valAxisLabelFontSize: 13,
  legendFontSize: 14, objectName: "curva overfitting 20k" });
texto(s, "Por eso lo que más sumó fue regularizar. Descartamos focal loss (sin mejora) y temperature scaling " +
  "(solo calibra).", { x: 9.15, y: 4.65, w: 3.4, h: 2.2, fontSize: 16, color: C.accent5 });
s.addNotes("[1:30] Sobre la muestra de 20k probamos técnicas de a una; cada barra suma una técnica a la anterior. " +
  "Desde cero: 2.211 km. Con pesos de ImageNet: 1.213 km. Después pasamos a celdas calculadas con todo el train " +
  "(barra gris): son más finas donde hay datos, la tarea es más difícil, por eso el error no baja, pero el piso de " +
  "error es menor. Loss haversine, como PIGEON: 1.100 km. Data augmentation: 888 km, el salto más grande. Snapshot " +
  "ensemble: 846. SAM, que busca mínimos más planos: 826. SAM + snapshots: 805 km. El problema dominante era el " +
  "overfitting: a la derecha, sin augmentation el train llega a 81% y la validación se queda en 30%. Por eso todo lo " +
  "que regularizaba mejoraba la validación. También probamos focal loss (no cambió el error), temperature scaling " +
  "(solo mejora la calibración) y ensambles de semillas (ayudan, pero cuestan varias corridas).");

// ================================================================ 11. Configuración
s = nueva("La configuración que llevamos al dataset completo", "Recorrido");
[["Arquitectura", "ResNet-18 preentrenada en ImageNet. Primero se entrena solo la última capa, después toda la red."],
  ["Clases", "70 celdas geográficas, con K-means esférico sobre las coordenadas de todo el train."],
  ["Loss", "Haversine label smoothing (τ = 300 km) con focal loss (γ = 2)."],
  ["Regularización", "Data augmentation (recortes y color), dropout 0,3 y SAM."],
  ["Ensamble", "Snapshot ensemble: ciclos cortos de learning rate al final del entrenamiento."],
  ["Muestreo", "Balanceado por celda: todas las regiones pesan lo mismo."]].forEach(([t, d], i) => {
  const x = M + (i % 2) * 6.15, y = 1.4 + Math.floor(i / 2) * 1.85;
  tarjeta(s, x, y, 5.95, 1.65);
  pin(s, i + 1, x + 0.25, y + 0.25);
  texto(s, t, { x: x + 0.9, y: y + 0.22, w: 4.85, h: 0.5, fontSize: 21, bold: true, valign: "middle" });
  texto(s, d, { x: x + 0.9, y: y + 0.75, w: 4.85, h: 0.85, fontSize: 17, color: C.accent5 });
});
s.addNotes("[0:45] De todas esas pruebas salió una configuración, y es la que llevamos al dataset completo: " +
  "ResNet-18 preentrenada, 70 celdas, loss haversine con focal, augmentation, dropout y SAM como regularización, " +
  "snapshot ensemble al final y muestreo balanceado por celda. Todo lo que había funcionado con 20k.");

// ================================================================ 12. Cómo armamos los conjuntos
s = nueva("Cómo armamos los conjuntos del dataset completo", "Recorrido");
[["Train", "1.932.636", "483 k panoramas", "shards 0 a 380", "Para entrenar", HEX.azul],
  ["Validación", "20.344", "5.086 panoramas", "shards 381 a 384", "Early stopping y elegir el modelo", HEX.verde],
  ["Test", "40.687", "10.172 panoramas", "shards 385 a 392", "Una sola vez, al final", HEX.pin]]
  .forEach(([n, img, panos, shards, uso, color], i) => {
    const x = M + i * 4.1;
    tarjeta(s, x, 1.4, 3.85, 3.4);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.65, w: 0.32, h: 0.32, fill: { color }, line: { color, width: 0 },
      objectName: `marca ${n}` });
    texto(s, n, { x: x + 0.75, y: 1.58, w: 2.9, h: 0.45, fontSize: 22, bold: true, valign: "middle" });
    texto(s, img, { x: x + 0.3, y: 2.15, w: 3.3, h: 0.75, fontSize: 40, bold: true, color, valign: "bottom" });
    texto(s, "imágenes", { x: x + 0.3, y: 2.92, w: 3.3, h: 0.4, fontSize: 17, color: C.text2 });
    texto(s, `${panos} · ${shards}`, { x: x + 0.3, y: 3.42, w: 3.4, h: 0.4, fontSize: 16, color: C.accent5 });
    texto(s, uso, { x: x + 0.3, y: 3.95, w: 3.4, h: 0.7, fontSize: 17, bold: true });
  });
texto(s, bullets(["Partición por archivo del dataset (shard): ningún panorama queda en dos conjuntos.",
  "Validación y test siguen la distribución real del dataset, como el juego (no están balanceados).",
  "Train: 97% de las imágenes; validación 1% y test 2% alcanzan para medir con precisión."]),
{ x: M, y: 5.15, w: W - 2 * M, h: 1.9, color: C.accent5 });
s.addNotes("[0:30] Para el dataset completo armamos la partición por archivo del dataset: 1,93 millones de imágenes " +
  "de train, 20 mil de validación y 40 mil de test. Ningún panorama queda partido entre dos conjuntos, y validación " +
  "y test respetan la distribución real. La validación la usamos para el early stopping y para elegir el modelo; " +
  "el test lo usamos una sola vez, al final.");

// ================================================================ 13. Entrenamiento 2M
s = nueva("Entrenar con 1,9 M de imágenes", "Recorrido");
[["Ronda de 200k", "imágenes balanceadas por celda; cada imagen a lo sumo una vez por ronda"],
  ["Evaluar", "en validación después de cada ronda"], ["Early stopping", "y bajar el learning rate si no mejora"]]
  .forEach(([t, d], i) => {
    const x = M + i * 4.1;
    tarjeta(s, x, 1.35, 3.75, 1.75);
    pin(s, i + 1, x + 0.25, 1.55);
    texto(s, t, { x: x + 0.85, y: 1.52, w: 2.8, h: 0.5, fontSize: 20, bold: true, valign: "middle" });
    texto(s, d, { x: x + 0.25, y: 2.15, w: 3.35, h: 0.9, fontSize: 16, color: C.accent5 });
    if (i < 2) texto(s, "→", { x: x + 3.77, y: 1.9, w: 0.33, h: 0.6, fontSize: 28, color: C.accent6, align: "center" });
  });
texto(s, bullets(["Google Colab con una GPU L4; checkpoints en Drive por si se corta la sesión.",
  "Imágenes recomprimidas a 384 px para que entren en el disco (de 181 a ~79 GB).",
  "Con la configuración completa: 8,6 h de entrenamiento."]), { x: M, y: 3.5, w: 5.9, h: 3.3, color: C.accent5 });
const esc = D.escala_datos;
s.addChart(pres.charts.BAR, [{ name: "km mediana", labels: ["20k", "200k", "1,9 M"], values: esc.km }], {
  x: 6.85, y: 3.3, w: 5.9, h: 3.65, barDir: "col", chartColors: [HEX.tenue, HEX.tenue, HEX.azul], showValue: true,
  dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0\" km\"", valAxisHidden: true, valAxisMinVal: 0,
  valAxisMaxVal: 950, showLegend: false, barGapWidthPct: 55, showTitle: true,
  title: "Error mediano según cantidad de imágenes", ...EJES, catAxisLabelFontSize: 17,
  objectName: "barras escala" });
s.addNotes("[0:45] Entrenar con 1,9 millones de imágenes cambia el esquema: en vez de épocas, entrenamos por rondas " +
  "de 200 mil imágenes balanceadas por celda, y después de cada ronda evaluamos en validación para el early " +
  "stopping. Corrimos en Colab con una L4, con checkpoints para sobrevivir a los cortes de sesión, y recomprimimos " +
  "las imágenes para que entraran en el disco. Resultado: en la misma validación fija de siempre, el error bajó de " +
  "805 km con 20k imágenes a 486 km con el dataset completo.");

// ================================================================ 14. Baselines
pres.addSection({ title: "Resultados" });
s = nueva("Sorpresa: los baselines simples le ganan al modelo completo", "Resultados");
texto(s, "Nos sobraba tiempo y teníamos créditos de Colab: corrimos baselines con el mismo pipeline, sin las técnicas " +
  "de regularización.", { x: M, y: 1.25, w: W - 2 * M, h: 0.7, fontSize: 18, color: C.accent5 });
const nombres = MODELOS.map(([n]) => n);
const etiquetasMod = ["Celdas CE simple", "Celdas CE + aug", "Celdas completo", "Países (101)"];
const barrasModelos = (valores, titulo, formato, extra, x) => s.addChart(pres.charts.BAR,
  [{ name: titulo, labels: etiquetasMod, values: valores }], {
    x, y: 2.0, w: 6.05, h: 4.35, barDir: "bar", catAxisOrientation: "maxMin", chartColors: MODELOS.map(([, c]) => c),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: formato, valAxisHidden: true,
    showLegend: false, barGapWidthPct: 35, showTitle: true, title: titulo, ...EJES, catAxisLabelFontSize: 16,
    dataLabelFontSize: 17, ...extra, objectName: titulo });
barrasModelos(nombres.map((n) => test[n]["4 vistas"].puntaje), "Puntaje GeoGuessr medio en test (más es mejor)",
  "#,##0", { valAxisMinVal: 0, valAxisMaxVal: 4400 }, M - 0.1);
barrasModelos(nombres.map((n) => test[n]["4 vistas"].km_mediana), "Error mediano en test (menos es mejor)",
  "#,##0\" km\"", { valAxisMinVal: 0, valAxisMaxVal: 730 }, 6.75);
texto(s, "CE simple: cross-entropy sin augmentation, SAM ni snapshots (2,3 h). Completo: la configuración de 20k " +
  "(8,6 h). Test, combinando las 4 vistas.", { x: M, y: 6.45, w: W - 2 * M, h: 0.6, fontSize: 15, color: C.accent6 });
s.addNotes("[1:00] Como nos sobraba tiempo y descubrimos que teníamos créditos de Colab, corrimos baselines con el " +
  "mismo pipeline: celdas con cross-entropy simple, la misma con augmentation, y un clasificador de 101 países. La " +
  "sorpresa: el baseline más simple, sin ninguna regularización, le ganó al modelo completo: 381 km contra 403 y 61 " +
  "puntos más por ronda (diferencia significativa por bootstrap), en un cuarto del tiempo. Y el de países pierde unos " +
  "500 puntos por ronda: confirma la elección de celdas.");

// ================================================================ 15. Por qué
s = nueva("¿Por qué? Con 1,9 M de imágenes no hay overfitting", "Resultados");
[[of20, "Con 20k imágenes (por época)", M], [of2m, "Con 1,9 M imágenes (por ronda)", 6.85]].forEach(([o, t, x]) => {
  s.addChart(pres.charts.LINE, [{ name: "train", labels: o.x.map(String), values: o.train },
    { name: "validación", labels: o.x.map(String), values: o.val }], {
    x, y: 1.3, w: 5.85, h: 3.75, chartColors: [HEX.tenue, HEX.pin], lineSize: 3, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 100, showTitle: true,
    title: `Accuracy (%) · ${t}`, ...EJES, catAxisLabelFrequency: o.x.length > 12 ? 5 : 1, objectName: `curva ${t}` });
});
const trenVal = (o) => `${Math.round(o.train.at(-1))}% vs ${Math.round(o.val.at(-1))}%`;
cifra(s, trenVal(of20), "accuracy en train vs. validación: el modelo memoriza el train", M, 5.15, 5.85, C.accent1, 40);
cifra(s, trenVal(of2m), "accuracy en train vs. validación: casi no hay diferencia", 6.85, 5.15, 5.85, C.accent3, 40);
texto(s, "La regularización ayudaba porque faltaban datos; con 100 veces más datos ya no aporta. Con más recursos, " +
  "habríamos probado más configuraciones directamente sobre el conjunto completo.", { x: M, y: 6.6, w: W - 2 * M,
  h: 0.6, fontSize: 17, bold: true });
s.addNotes("[0:45] ¿Por qué? Las curvas lo muestran. Con 20k el modelo memorizaba: 81% de accuracy en train contra " +
  "30% en validación, una diferencia enorme. Con 1,9 millones casi no hay diferencia: 75% contra 67%. Las técnicas " +
  "que ganábamos en la muestra chica eran todas de regularización, y si no hay overfitting que combatir, no aportan. " +
  "La lección es que las conclusiones con datos chicos no siempre escalan; con más recursos habríamos repetido más " +
  "pruebas directamente sobre el conjunto completo.");

// ================================================================ 16. Ganador: métricas
s = nueva("Modelo final en test: celdas + CE, combinando las 4 vistas", "Resultados");
[[`${g4.km_mediana} km`, "error mediano", C.accent1], [fmt(g4.puntaje), "puntos GeoGuessr por ronda", C.text2],
  [`${coma(g4.pct_750)}%`, "a menos de 750 km", C.text2], ["78,3%", "país correcto", C.text2]]
  .forEach(([v, e, c], i) => cifra(s, v, e, M + i * 3.05, 1.2, 2.9, c, 46));
imagen(s, "test_cdf.png", M, 2.95, 8.3, 4.0, { alinear: "izq", alt: "Distribución acumulada del error en km" });
const encabezado = (t) => ({ text: t, options: { bold: true, fill: { color: HEX.panel } } });
s.addTable([
  [encabezado(""), encabezado("1 vista"), encabezado("4 vistas")],
  ["km mediana", fmt(g1.km_mediana), fmt(g4.km_mediana)],
  ["puntaje", fmt(g1.puntaje), fmt(g4.puntaje)],
  ["< 750 km", `${coma(g1.pct_750)}%`, `${coma(g4.pct_750)}%`],
  ["< 2500 km", `${coma(g1.pct_2500)}%`, `${coma(g4.pct_2500)}%`],
], { x: 9.1, y: 3.05, w: 3.65, colW: [1.45, 1.1, 1.1], fontSize: 17, color: HEX.tinta, align: "center",
  border: { type: "solid", color: HEX.grilla, pt: 0.75 }, rowH: 0.46, objectName: "tabla 1 vs 4 vistas" });
texto(s, `Combinar las 4 vistas suma +${ganancia4} puntos por ronda, sin reentrenar.`, { x: 9.1, y: 5.5, w: 3.65,
  h: 1.3, fontSize: 18, bold: true, color: C.accent1 });
s.addNotes("[1:00] Este es el modelo final, evaluado una sola vez en test: 40 mil imágenes que ningún modelo vio. Con " +
  "las 4 vistas: 381 km de error mediano, 3.664 puntos de GeoGuessr por ronda en promedio, 83% de las predicciones " +
  "a menos de 750 km y el país correcto en el 78% de los casos. Los números de test coinciden con los de " +
  "validación. La curva muestra la fracción de predicciones por debajo de cada distancia; el modelo de países " +
  "(amarillo) queda claramente atrás. Promediar las predicciones de las 4 vistas del panorama, sin reentrenar nada, " +
  "suma " + ganancia4 + " puntos por ronda: la mejora más grande del proyecto.");

// ================================================================ 17. Dónde acierta y dónde falla
s = nueva("¿Dónde acierta y dónde se equivoca?", "Resultados");
const re = imagen(s, "test_ejemplos.png", M - 0.1, 1.3, 8.6, 5.0, { alinear: "izq",
  alt: "30 panoramas de test al azar" });
texto(s, "30 panoramas de test al azar: ubicación real → predicción del modelo", { x: M, y: re.y + re.h + 0.05,
  w: 8.3, h: 0.4, fontSize: 15, color: C.accent6 });
[["73%", "de los errores caen en una celda vecina: reconoce la región y falla en el corte fino."],
  ["Canadá → EE. UU.", "se confunden países chicos con un vecino grande (también Irlanda → Reino Unido)."],
  ["~320 km", "es el piso de error de 70 celdas: para bajar de ahí hace falta una grilla más fina."]]
  .forEach(([v, d], i) => {
    const y = 1.3 + i * 1.9;
    tarjeta(s, 8.85, y, 3.9, 1.75);
    texto(s, v, { x: 9.05, y: y + 0.12, w: 3.55, h: 0.55, fontSize: 26, bold: true, color: C.accent1 });
    texto(s, d, { x: 9.05, y: y + 0.7, w: 3.55, h: 1.0, fontSize: 16, color: C.accent5 });
  });
s.addNotes("[0:30] ¿Dónde se equivoca? En el mapa, 30 panoramas de test al azar: la mayoría de las líneas son " +
  "cortas, verdes. Tres observaciones: (1) el 73% de los errores cae en una celda vecina: el modelo reconoce la " +
  "región y falla en el corte fino; (2) se confunden países chicos con un vecino grande, como Canadá con EE. UU.; " +
  "(3) con 70 celdas hay un piso de unos 320 km: para mejorar mucho más haría falta una grilla más fina o refinar " +
  "la coordenada dentro de la celda.");

// ================================================================ 18. Conclusiones
s = nueva("Conclusiones", "Resultados", "CIERRE");
[["Los datos fueron la clave", "De 20k a 1,9 M imágenes: de 805 a 455 km en la misma validación."],
  ["La regularización depende del régimen", "Ayudó mucho con 20k imágenes; con 1,9 M dejó de sumar."],
  ["Celdas mejor que países para jugar", "Unos 500 puntos más por ronda: la granularidad importa."],
  ["Las 4 vistas, sin costo", `Promediar las vistas suma +${ganancia4} puntos sin reentrenar.`]].forEach(([t, d], i) => {
  const y = 1.45 + i * 1.22;
  pin(s, i + 1, M, y + 0.03, 0.52);
  texto(s, t, { x: M + 0.8, y, w: 7.1, h: 0.5, fontSize: 23, bold: true, color: C.background1 });
  texto(s, d, { x: M + 0.8, y: y + 0.52, w: 7.1, h: 0.62, fontSize: 18, color: HEX.claroOscuro });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.75, y: 1.45, w: 3.98, h: 4.6, rectRadius: 0.08,
  fill: { color: "1B3954" }, line: { color: "1B3954", width: 0 }, objectName: "tarjeta resultado" });
texto(s, "Modelo final (test)", { x: 9.0, y: 1.65, w: 3.5, h: 0.45, fontSize: 19, color: HEX.claroOscuro });
texto(s, `${g4.km_mediana} km`, { x: 9.0, y: 2.1, w: 3.5, h: 0.95, fontSize: 58, bold: true, color: C.accent1 });
texto(s, "de error mediano", { x: 9.0, y: 3.05, w: 3.5, h: 0.45, fontSize: 19, color: C.background1 });
texto(s, `${fmt(g4.puntaje)} puntos`, { x: 9.0, y: 3.75, w: 3.5, h: 0.75, fontSize: 36, bold: true,
  color: C.background1 });
texto(s, "por ronda de GeoGuessr (sobre 5000)", { x: 9.0, y: 4.5, w: 3.5, h: 0.9, fontSize: 19,
  color: HEX.claroOscuro });
texto(s, "Próximos pasos: celdas más finas, refinar la coordenada dentro de la celda y probar configuraciones " +
  "directamente sobre el dataset completo.", { x: M, y: 6.35, w: W - 2 * M, h: 0.8, fontSize: 17,
  color: HEX.claroOscuro });
s.addNotes("[0:45] Para cerrar, cuatro conclusiones. Primero, los datos fueron la clave: en la misma validación " +
  "pasamos de 805 a 455 km solo por usar el dataset completo. Segundo, las técnicas de regularización sirven cuando " +
  "hay overfitting; con el dataset completo dejaron de sumar. Tercero, para jugar conviene predecir celdas y no " +
  "países. Cuarto, combinar las 4 vistas es la mejora más barata. El modelo final saca 3.664 puntos por ronda con " +
  "381 km de error mediano. ¡Gracias! ¿Preguntas?");

// ---------------------------------------------------------------- escritura
// pptxgenjs escribe las categorías de los gráficos como multinivel; Google Slides y otros visores no las leen y
// muestran 1, 2, 3... Se reescriben como categorías simples (lo mismo que escribe PowerPoint).
async function categoriasSimples(archivo) {
  const zip = await JSZip.loadAsync(fs.readFileSync(archivo));
  for (const nombre of Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    let xml = await zip.file(nombre).async("string");
    xml = xml.replace(/<c:multiLvlStrRef>/g, "<c:strRef>").replace(/<\/c:multiLvlStrRef>/g, "</c:strRef>")
      .replace(/<c:multiLvlStrCache>/g, "<c:strCache>").replace(/<\/c:multiLvlStrCache>/g, "</c:strCache>")
      .replace(/<c:lvl>/g, "").replace(/<\/c:lvl>/g, "");
    zip.file(nombre, xml);
  }
  // Se copia solo los archivos (sin entradas de carpeta: los servicios en la nube no abren pptx que las tienen).
  // zip.remove() de una carpeta borraría también su contenido.
  const limpio = new JSZip();
  for (const [nombre, entrada] of Object.entries(zip.files)) {
    if (!entrada.dir) limpio.file(nombre, await entrada.async("nodebuffer"), { createFolders: false });
  }
  fs.writeFileSync(archivo, await limpio.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}
(async () => {
  const salida = path.join(__dirname, "presentacion_geoguessr.pptx");
  await pres.writeFile({ fileName: salida });
  await applyTheme(salida, THEME);
  await categoriasSimples(salida);
  console.log("Escrito:", salida);
})();
