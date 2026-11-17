"""GeoDuelo: un GeoGuessr contra el modelo de celdas de 2M (outputs/entrenamiento_2m/baseline_celdas_2m).

Cada ronda es un panorama del split de test (4 vistas). El jugador pone un pin en el globo; el servidor devuelve la
ubicación real, la distancia y el puntaje del jugador y la predicción del modelo (promedio de las probabilidades de
las 4 vistas), con su distancia, puntaje, top-5 de celdas y la predicción de cada vista por separado.

Las coordenadas reales no salen del servidor hasta que el jugador adivina.

Modo explorador (/explorar.html): el usuario elige un lugar y camina en Street View; al capturar, el servidor baja las
4 vistas con la Street View Static API (N/E/S/O, FOV 90, 512×512, como el dataset) y muestra qué predice el modelo.
Necesita GOOGLE_MAPS_API_KEY en .env con "Maps JavaScript API" y "Street View Static API" habilitadas.

Uso: .venv/Scripts/python.exe -m src.geoguessr.servidor [--puerto 8765] [--datos ../world-streetview-500k/data]
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import queue
import secrets
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.entrenamiento.datos import transformaciones  # noqa: E402
from src.entrenamiento.entrenamiento import construir_modelo, haversine_np  # noqa: E402

ESTATICOS = Path(__file__).resolve().parent / "static"
CORRIDA = RAIZ / "outputs/entrenamiento_2m/baseline_celdas_2m"
TEST_FILAS = RAIZ / "outputs/entrenamiento_2m/test_filas.parquet"
ESCALA_PUNTAJE_KM = 1492.7  # Mapa mundial de GeoGuessr: 5000 · exp(-d / 1492.7)
MODOS = ("libre", "paises")


class ErrorExplorar(Exception):
    pass


def leer_google_key() -> str | None:
    """GOOGLE_MAPS_API_KEY desde el entorno o desde el .env de la raíz del repo (ignorado por git)."""
    if os.environ.get("GOOGLE_MAPS_API_KEY"):
        return os.environ["GOOGLE_MAPS_API_KEY"].strip()
    env = RAIZ / ".env"
    if env.exists():
        for linea in env.read_text(encoding="utf-8").splitlines():
            clave, _, valor = linea.partition("=")
            if clave.strip() == "GOOGLE_MAPS_API_KEY" and valor.strip():
                return valor.strip().strip("\"'")
    return None


def puntaje(km: float) -> int:
    return int(round(5000 * np.exp(-km / ESCALA_PUNTAJE_KM)))


def recomprimir(im: Image.Image, lado: int = 384, calidad: int = 90) -> Image.Image:
    """La misma recompresión que tuvieron train/val (ver scripts/inferir_test.py)."""
    ancho, alto = im.size
    escala = lado / min(ancho, alto)
    if escala >= 1:
        return im
    im = im.resize((round(ancho * escala), round(alto * escala)), Image.BICUBIC)
    salida = io.BytesIO()
    im.save(salida, "JPEG", quality=calidad)
    return Image.open(io.BytesIO(salida.getvalue())).convert("RGB")


class Juego:
    def __init__(self, datos: Path, semilla: int | None):
        self.datos = datos
        self.rng = np.random.default_rng(semilla)
        self.lock_modelo = threading.Lock()
        self.rondas: OrderedDict[str, dict] = OrderedDict()  # token -> ronda (LRU acotado)
        self.lock_rondas = threading.Lock()
        self.colas = {m: queue.Queue(maxsize=2) for m in MODOS}
        self.google_key = leer_google_key()

        test = pd.read_parquet(TEST_FILAS)
        test = test.sort_values(["archivo", "fila"])
        tamanos = test.groupby("panoid").size()
        test = test[test.panoid.isin(tamanos[tamanos == 4].index)]
        self.panos = (test.groupby("panoid", sort=False)
                      .agg(archivo=("archivo", "first"), filas=("fila", list), lat=("latitude", "first"),
                           lon=("longitude", "first"), pais=("country_code", "first"))
                      .reset_index())
        self.por_pais = self.panos.groupby("pais").indices
        print(f"Test: {len(self.panos)} panoramas de 4 vistas, {len(self.por_pais)} países")

        clases = pd.read_csv(CORRIDA / "clases.csv").sort_values("clase")
        self.celdas = clases[["lat", "lon"]].to_numpy()
        self.dist_celdas = haversine_np(self.celdas[:, None, 0], self.celdas[:, None, 1],
                                        self.celdas[None, :, 0], self.celdas[None, :, 1])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.loads((CORRIDA / "config.json").read_text(encoding="utf-8"))
        estado = torch.load(CORRIDA / "mejor.pt", map_location="cpu", weights_only=False)["modelo"]
        self.modelo = construir_modelo(estado["fc.1.weight"].shape[0], cfg["dropout"], preentrenado=False)
        self.modelo.load_state_dict(estado)
        self.modelo.to(self.device).eval()
        _, self.transform = transformaciones(cfg["tamano_imagen"])
        print(f"Modelo: {CORRIDA.name}/mejor.pt ({len(self.celdas)} celdas) en {self.device}")
        print("Modo explorador: " + ("activo (GOOGLE_MAPS_API_KEY encontrada)" if self.google_key
                                     else "sin GOOGLE_MAPS_API_KEY en .env, no va a funcionar"))

        for modo in MODOS:
            threading.Thread(target=self._precargar, args=(modo,), daemon=True).start()

    # ------------------------------------------------------------------ rondas
    def _elegir(self, modo: str) -> pd.Series:
        if modo == "paises":  # Uniforme sobre países, después uniforme dentro del país.
            pais = self.rng.choice(list(self.por_pais))
            return self.panos.iloc[int(self.rng.choice(self.por_pais[pais]))]
        return self.panos.iloc[int(self.rng.integers(len(self.panos)))]

    def _leer_vistas(self, archivo: str, filas: list[int]) -> list[bytes]:
        with pq.ParquetFile(self.datos / archivo) as pf:
            limites = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
            salida = {}
            for g in sorted({int(np.searchsorted(limites, f, side="right") - 1) for f in filas}):
                col = pf.read_row_group(g, columns=["image"]).column(0)
                for f in filas:
                    if limites[g] <= f < limites[g + 1]:
                        salida[f] = col[int(f - limites[g])].as_py()["bytes"]
        return [salida[f] for f in filas]

    def _predecir(self, vistas: list[bytes]) -> np.ndarray:
        x = torch.stack([self.transform(recomprimir(Image.open(io.BytesIO(b)).convert("RGB"))) for b in vistas])
        with self.lock_modelo, torch.inference_mode():
            return torch.softmax(self.modelo(x.to(self.device)).float(), dim=1).cpu().numpy()

    def _armar(self, modo: str) -> dict:
        p = self._elegir(modo)
        vistas = self._leer_vistas(p.archivo, p.filas)
        return {"panoid": p.panoid, "lat": float(p.lat), "lon": float(p.lon), "pais": str(p.pais),
                "vistas": vistas, "probs": self._predecir(vistas), "creada": time.time()}

    def _precargar(self, modo: str) -> None:
        while True:
            try:
                self.colas[modo].put(self._armar(modo))
            except Exception as e:  # noqa: BLE001
                print(f"Error precargando ronda ({modo}): {e}", flush=True)
                time.sleep(1)

    def nueva_ronda(self, modo: str) -> dict:
        ronda = self.colas[modo if modo in MODOS else "libre"].get(timeout=60)
        token = secrets.token_urlsafe(12)
        with self.lock_rondas:
            self.rondas[token] = ronda
            while len(self.rondas) > 200:
                self.rondas.popitem(last=False)
        return {"token": token, "vistas": [f"/api/imagen/{token}/{k}" for k in range(len(ronda["vistas"]))]}

    def imagen(self, token: str, k: int) -> bytes | None:
        with self.lock_rondas:
            ronda = self.rondas.get(token)
        return ronda["vistas"][k] if ronda and 0 <= k < len(ronda["vistas"]) else None

    def info(self) -> dict:
        return {"corrida": CORRIDA.name, "panoramas": len(self.panos), "paises": len(self.por_pais),
                "celdas": [{"lat": float(a), "lon": float(b)} for a, b in self.celdas]}

    # ------------------------------------------------------------------ resultado
    def _celda(self, c: int, prob: float, lat: float, lon: float) -> dict:
        clat, clon = self.celdas[c]
        km = float(haversine_np(lat, lon, clat, clon))
        return {"celda": int(c), "lat": float(clat), "lon": float(clon), "prob": float(prob), "km": km,
                "puntos": puntaje(km)}

    def _distribucion(self, p: np.ndarray, minimo: float = 0.003) -> list[dict]:
        return [{"lat": float(self.celdas[c, 0]), "lon": float(self.celdas[c, 1]), "prob": float(p[c])}
                for c in np.argsort(-p) if p[c] >= minimo]

    def _prediccion_vista(self, p: np.ndarray, lat: float, lon: float) -> dict:
        orden = np.argsort(-p)
        return {**self._celda(int(orden[0]), float(p[orden[0]]), lat, lon),
                "top": [self._celda(int(c), float(p[c]), lat, lon) for c in orden[:5]],
                "distribucion": self._distribucion(p)}

    def adivinar(self, token: str, lat: float | None, lon: float | None) -> dict | None:
        with self.lock_rondas:
            ronda = self.rondas.get(token)
            if ronda is None or ronda.get("respondida"):
                return None
            ronda["respondida"] = True
        if lat is None or lon is None:
            jugador = {"lat": None, "lon": None, "km": None, "puntos": 0}
        else:
            km = float(haversine_np(ronda["lat"], ronda["lon"], lat, lon))
            jugador = {"lat": lat, "lon": lon, "km": km, "puntos": puntaje(km)}
        return self._evaluar(ronda, jugador)

    # ------------------------------------------------------------------ modo explorador
    def explorar(self, pano: str, lat: float, lon: float, heading: float) -> dict:
        """Baja de la Street View Static API las 4 vistas que vería el modelo (headings heading + 0/90/180/270,
        FOV 90, 512×512, igual que el dataset), las predice y evalúa contra la posición del panorama."""
        if not self.google_key:
            raise ErrorExplorar("Falta GOOGLE_MAPS_API_KEY en .env")
        vistas = []
        for k in range(4):
            params = urlencode({"size": "512x512", "pano": pano, "heading": f"{(heading + 90 * k) % 360:.2f}",
                                "fov": 90, "pitch": 0, "return_error_code": "true", "source": "outdoor",
                                "key": self.google_key})
            try:
                with urlopen(f"https://maps.googleapis.com/maps/api/streetview?{params}", timeout=20) as r:
                    vistas.append(r.read())
            except HTTPError as e:
                detalle = e.read().decode(errors="replace")[:300]
                raise ErrorExplorar(f"Street View Static API respondió {e.code}: {detalle}") from e
            except URLError as e:
                raise ErrorExplorar(f"No se pudo conectar con Google: {e.reason}") from e
        ronda = {"panoid": pano, "lat": lat, "lon": lon, "pais": None, "vistas": vistas,
                 "probs": self._predecir(vistas), "creada": time.time(), "respondida": True}
        token = secrets.token_urlsafe(12)
        with self.lock_rondas:
            self.rondas[token] = ronda
            while len(self.rondas) > 200:
                self.rondas.popitem(last=False)
        return {**self._evaluar(ronda, None), "vistas": [f"/api/imagen/{token}/{k}" for k in range(4)],
                "headings": [round((heading + 90 * k) % 360, 1) for k in range(4)]}

    def _evaluar(self, ronda: dict, jugador: dict | None) -> dict:
        rlat, rlon = ronda["lat"], ronda["lon"]
        probs = ronda["probs"]
        media = probs.mean(axis=0)
        orden = np.argsort(-media)
        top = [self._celda(c, media[c], rlat, rlon) for c in orden[:5]]
        # Celda que minimiza la distancia esperada bajo la distribución del modelo (alternativa al argmax).
        c_bayes = int(np.argmin(self.dist_celdas @ media))
        celda_real = int(np.argmin(haversine_np(rlat, rlon, self.celdas[:, 0], self.celdas[:, 1])))
        rango = int(np.where(orden == celda_real)[0][0]) + 1
        entropia = float(-(media * np.log(media + 1e-12)).sum() / np.log(len(media)))
        return {
            "real": {"lat": rlat, "lon": rlon, "pais": ronda["pais"], "panoid": ronda["panoid"]},
            "jugador": jugador,
            "modelo": {**top[0], "top": top, "rango_celda_real": rango, "entropia": entropia,
                       "bayes": self._celda(c_bayes, media[c_bayes], rlat, rlon),
                       "distribucion": self._distribucion(media),
                       "vistas": [self._prediccion_vista(p, rlat, rlon) for p in probs]},
        }


def crear_handler(juego: Juego):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # Silencioso
            pass

        def _enviar(self, codigo: int, cuerpo: bytes, tipo: str, cache: bool = False):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)

        def _json(self, obj, codigo: int = 200):
            self._enviar(codigo, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            partes = [p for p in url.path.split("/") if p]
            try:
                if partes[:2] == ["api", "info"]:
                    return self._json(juego.info())
                if partes[:2] == ["api", "config"]:  # La key va igual al navegador para la Maps JavaScript API.
                    return self._json({"google_maps_key": juego.google_key})
                if partes[:2] == ["api", "ronda"]:
                    modo = parse_qs(url.query).get("modo", ["libre"])[0]
                    return self._json(juego.nueva_ronda(modo))
                if partes[:2] == ["api", "imagen"] and len(partes) == 4:
                    datos = juego.imagen(partes[2], int(partes[3]))
                    return self._enviar(200, datos, "image/jpeg", cache=True) if datos else self._json({}, 404)
                ruta = (ESTATICOS / ("/".join(partes) or "index.html")).resolve()
                if ESTATICOS in ruta.parents and ruta.is_file():
                    tipo = mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
                    return self._enviar(200, ruta.read_bytes(), tipo)
                return self._json({"error": "no encontrado"}, 404)
            except queue.Empty:
                return self._json({"error": "todavía cargando rondas"}, 503)
            except (ValueError, IndexError):
                return self._json({"error": "pedido inválido"}, 400)

        def do_POST(self):
            ruta = urlparse(self.path).path
            if ruta == "/api/explorar":
                try:
                    c = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    pano = str(c["pano"])
                    if not pano.replace("-", "").replace("_", "").isalnum():
                        raise ValueError
                    return self._json(juego.explorar(pano, float(c["lat"]), float(c["lon"]),
                                                     float(c.get("heading", 0))))
                except ErrorExplorar as e:
                    return self._json({"error": str(e)}, 502)
                except (ValueError, TypeError, KeyError):
                    return self._json({"error": "pedido inválido"}, 400)
            if ruta != "/api/adivinar":
                return self._json({"error": "no encontrado"}, 404)
            try:
                cuerpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                lat, lon = cuerpo.get("lat"), cuerpo.get("lon")
                if lat is not None:
                    lat, lon = float(lat), ((float(lon) + 180) % 360) - 180
                resultado = juego.adivinar(str(cuerpo.get("token", "")), lat, lon)
            except (ValueError, TypeError):
                return self._json({"error": "pedido inválido"}, 400)
            return self._json(resultado) if resultado else self._json({"error": "ronda inexistente"}, 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--datos", default=str(RAIZ.parent / "world-streetview-500k/data"))
    ap.add_argument("--semilla", type=int, default=None)
    ap.add_argument("--sin-navegador", action="store_true")
    args = ap.parse_args()
    juego = Juego(Path(args.datos), args.semilla)
    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto), crear_handler(juego))
    url = f"http://127.0.0.1:{args.puerto}"
    print(f"GeoDuelo listo en {url}  (Ctrl+C para cortar)", flush=True)
    if not args.sin_navegador:
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
