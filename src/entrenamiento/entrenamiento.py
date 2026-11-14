"""Loop de entrenamiento por rondas, con checkpoints y retomado automático.

Fases (el estado se guarda en el checkpoint y al relanzar se sigue desde donde quedó):
1. base: rondas hasta early stopping (CE haversine de validación), con ReduceLROnPlateau. Las primeras
   `rondas_cabeza` rondas entrenan solo la capa final (backbone congelado y BatchNorm en eval).
2. ciclos: por cada variante de `variantes_ciclos`, desde el MISMO mejor checkpoint de la fase base, ciclos de
   LR coseno con reinicio; al final de cada ciclo se guarda un snapshot. Las rondas siguen pidiéndose al mismo
   planificador (datos nuevos, mismo balance).
3. evaluacion_final: logits de validación completa (4 vistas) y de la validación global de 20k para el mejor
   modelo y cada snapshot (para analizar ensamble y SWA en un notebook).

Checkpoint `checkpoints/ultimo.pt` (y `anterior.pt` como respaldo): modelo, optimizador, scheduler, estado de la
corrida (fase, ronda, archivos ya consumidos de la ronda en curso, contadores de early stopping, historial) y
estados aleatorios. Se escribe de forma atómica cada `minutos_checkpoint`, al final de cada ronda, al
interrumpir (Ctrl+C / SIGTERM) y al llegar a `max_horas`.
"""
from __future__ import annotations

import json
import math
import os
import random
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from .config import Config
from .datos import FlujoMuestras, Lector, ram_usada_gb, tareas_por_archivo, transformaciones
from .sampler import Planificador

RADIO_TIERRA_KM = 6371.0
RAIZ = Path(__file__).resolve().parents[2]


class Pausa(Exception):
    """Corte prolijo (max_horas o señal): el checkpoint ya está guardado."""


def resolver(ruta: str | Path) -> Path:
    p = Path(ruta).expanduser()
    return (p if p.is_absolute() else RAIZ / p).resolve()


# ------------------------------------------------------------------ geometría
def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_torch(lat, lon, plat, plon):
    lat, lon, plat, plon = (torch.deg2rad(t) for t in (lat[:, None], lon[:, None], plat[None], plon[None]))
    a = torch.sin((plat - lat) / 2) ** 2 + torch.cos(lat) * torch.cos(plat) * torch.sin((plon - lon) / 2) ** 2
    return 2 * RADIO_TIERRA_KM * torch.arcsin(torch.sqrt(a.clamp(0, 1)))


def to_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


# ------------------------------------------------------------------ modelo, loss y SAM
def construir_modelo(num_clases: int, dropout: float, preentrenado: bool = True) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if preentrenado else None)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_clases))
    return m


def loss_haversine(logits, distancias, tau, gamma=0.0):
    # CE contra el target blando q = softmax(-d/τ); con gamma > 0, focal: cada término por (1 - p_k)^γ.
    q = torch.softmax(-distancias / tau, dim=1)
    log_p = torch.log_softmax(logits.float(), dim=1)
    if gamma:
        return -(q * (1 - log_p.exp()) ** gamma * log_p).sum(dim=1).mean()
    return -(q * log_p).sum(dim=1).mean()


class SAM:
    # Sharpness-Aware Minimization (Foret et al., 2021) sobre AdamW, igual que en los notebooks de 20k.
    def __init__(self, optimizador_base, rho):
        self.base, self.rho, self.perturbaciones = optimizador_base, rho, {}

    @torch.no_grad()
    def primer_paso(self):
        params = [p for g in self.base.param_groups for p in g["params"] if p.grad is not None]
        norma = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
        escala = self.rho / (norma + 1e-12)
        for p in params:
            e = p.grad * escala
            p.add_(e)
            self.perturbaciones[p] = e
        self.base.zero_grad(set_to_none=True)

    @torch.no_grad()
    def segundo_paso(self):
        for p, e in self.perturbaciones.items():
            p.sub_(e)
        self.perturbaciones = {}
        self.base.step()
        self.base.zero_grad(set_to_none=True)

    @torch.no_grad()
    def deshacer(self):
        # Si la segunda pasada da loss no finita: volver a w sin dar el paso.
        for p, e in self.perturbaciones.items():
            p.sub_(e)
        self.perturbaciones = {}
        self.base.zero_grad(set_to_none=True)


@contextmanager
def bn_sin_actualizar(red):
    bns = [m for m in red.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momentos = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, momento in zip(bns, momentos):
            m.momentum = momento


# ------------------------------------------------------------------ métricas
def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metricas(logits: np.ndarray, celda: np.ndarray, dist: np.ndarray, dist_proto: np.ndarray, tau: float) -> dict:
    p = softmax_np(logits.astype(np.float64))
    pred = p.argmax(axis=1)
    pred_mbr = (p @ dist_proto).argmin(axis=1)
    idx = np.arange(len(celda))
    q = softmax_np(-dist / tau)
    log_p = np.log(np.clip(p, 1e-12, None))
    conf, acierto = p.max(axis=1), pred == celda
    bins = np.clip(np.digitize(conf, np.linspace(0, 1, 11)[1:-1]), 0, 9)
    ece = sum((bins == b).mean() * abs(acierto[bins == b].mean() - conf[bins == b].mean())
              for b in range(10) if (bins == b).any())
    return {"ce_haversine": float(-(q * log_p).sum(axis=1).mean()), "ce": float(-log_p[idx, celda].mean()),
            "top1": float(acierto.mean()), "top5": float((np.argsort(-p, axis=1)[:, :5] == celda[:, None]).any(1).mean()),
            "km_mediana": float(np.median(dist[idx, pred])), "km_media": float(dist[idx, pred].mean()),
            "km_mediana_mbr": float(np.median(dist[idx, pred_mbr])), "ece": float(ece)}


# ------------------------------------------------------------------ tablas de datos
class Tabla:
    """Filas de un conjunto: archivo, fila, coordenadas, celda y panorama."""

    def __init__(self, df: pd.DataFrame):
        self.archivo = df["archivo"].to_numpy().astype(str)
        self.fila = df["fila"].to_numpy(np.int64)
        self.lat = df["latitude"].to_numpy(np.float64)
        self.lon = df["longitude"].to_numpy(np.float64)
        self.celda = df["ind_cluster"].to_numpy(np.int64)
        self.panoid = df["panoid"].to_numpy().astype(str)

    def __len__(self):
        return len(self.fila)


def rango_shards(texto: str | None, numeros: pd.Series) -> pd.Series:
    if not texto:
        return pd.Series(True, index=numeros.index)
    a, _, b = texto.partition("-")
    return (numeros >= int(a)) & (numeros <= int(b or a))


def cargar_tablas(cfg: Config, prototipos: np.ndarray) -> dict[str, Tabla]:
    splits = pq.read_table(resolver(cfg.splits), columns=["split", "archivo", "fila", "panoid", "ind_cluster"]).to_pandas()
    coords = pq.read_table(resolver(cfg.manifest), columns=["archivo", "fila", "latitude", "longitude"]).to_pandas()
    df = splits.merge(coords, on=["archivo", "fila"], how="left", validate="one_to_one")
    if df[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Filas de splits sin coordenadas en el manifest.")
    tablas = {}
    tr = df[df["split"] == "train"]
    numeros = tr["archivo"].str.extract(r"train-(\d+)-")[0].astype(int)
    tr = tr[rango_shards(cfg.shards_train, numeros)]
    tablas["train"] = Tabla(tr.sort_values(["panoid", "archivo", "fila"]).reset_index(drop=True))
    va = df[df["split"] == "val"].sort_values(["archivo", "fila"]).reset_index(drop=True)
    if cfg.limite_val:  # Solo para pruebas rápidas: los primeros N panoramas.
        va = va[va["panoid"].isin(va["panoid"].drop_duplicates().head(cfg.limite_val))].reset_index(drop=True)
    tablas["val"] = Tabla(va)
    tablas["val_1vista"] = Tabla(va.drop_duplicates("panoid", keep="first").reset_index(drop=True))
    if cfg.val_global:
        ruta = resolver(cfg.val_global)
        vg = pq.read_table(ruta, columns=["panoid", "latitude", "longitude"]).to_pandas()
        vg["archivo"], vg["fila"] = str(ruta), np.arange(len(vg))
        if cfg.limite_val:
            vg = vg[vg["panoid"].isin(vg["panoid"].drop_duplicates().head(cfg.limite_val))]
        vg["ind_cluster"] = (to_xyz(vg["latitude"].to_numpy(), vg["longitude"].to_numpy())
                             @ to_xyz(prototipos[:, 0], prototipos[:, 1]).T).argmax(axis=1)
        tablas["val_global"] = Tabla(vg)
    return tablas


# ------------------------------------------------------------------ checkpoints
def guardar_atomico(objeto, destino: Path, respaldo: Path | None = None) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp")
    torch.save(objeto, tmp)
    with open(tmp, "r+b") as f:  # Forzar a disco antes del reemplazo (en Windows, fsync pide escritura).
        os.fsync(f.fileno())
    if respaldo is not None and destino.exists():
        os.replace(destino, respaldo)
    os.replace(tmp, destino)


def commit_git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ corrida
class Corrida:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.carpeta = resolver(cfg.carpeta_corrida)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.ruta_ultimo = self.carpeta / "checkpoints" / "ultimo.pt"
        self.ruta_anterior = self.carpeta / "checkpoints" / "anterior.pt"
        self.inicio_sesion = time.time()
        self.ultimo_guardado = time.time()
        self.senal_recibida = False
        self.backbone_congelado = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._registrar_config()

        ruta_proto = resolver(cfg.prototipos)
        import hashlib

        if hashlib.sha256(ruta_proto.read_bytes()).hexdigest() != cfg.sha256_prototipos:
            raise ValueError(f"{ruta_proto} no coincide con los prototipos del manifest.")
        proto = pd.read_csv(ruta_proto).sort_values("ind_cluster")
        self.prototipos = proto[["lat", "lon"]].to_numpy(np.float64)
        self.num_clases = len(self.prototipos)
        self.dist_proto = haversine_np(self.prototipos[:, None, 0], self.prototipos[:, None, 1],
                                       self.prototipos[None, :, 0], self.prototipos[None, :, 1])
        self.proto_t = torch.tensor(self.prototipos, dtype=torch.float32, device=self.device)

        self.tablas = cargar_tablas(cfg, self.prototipos)
        self.lector = Lector(cfg.fuente, resolver(cfg.datos_local), cfg.hf_repo, cfg.hf_revision, cfg.hf_subcarpeta,
                             cfg.hf_cache, cfg.borrar_shards_hf)
        # Los parquets de validación se usan en cada ronda: con fuente HF se bajan una vez y no se borran.
        self.lector_eval = Lector(cfg.fuente, resolver(cfg.datos_local), cfg.hf_repo, cfg.hf_revision,
                                  cfg.hf_subcarpeta, cfg.hf_cache, borrar=False)
        self.transform_train, self.transform_val = transformaciones(cfg.tamano_imagen)
        tr = self.tablas["train"]
        _, self.pano_idx = np.unique(tr.panoid, return_inverse=True)
        n_panos = self.pano_idx.max() + 1
        self.inicio_filas = np.searchsorted(self.pano_idx, np.arange(n_panos))
        self.n_filas = np.bincount(self.pano_idx)
        self.celda_pano = tr.celda[self.inicio_filas]
        # Tamaño de cada celda en train (imágenes), para monitorear sobreajuste en celdas chicas.
        imgs_celda = np.bincount(tr.celda, minlength=self.num_clases)
        self.grupo_celda = np.digitize(imgs_celda, np.quantile(imgs_celda, [1 / 3, 2 / 3]))  # 0 chica, 2 grande
        self.planificador = self._nuevo_planificador()

        self.modelo = construir_modelo(self.num_clases, cfg.dropout).to(self.device)
        self.optimizador = self._nuevo_optimizador()
        self.scheduler = self._nuevo_plateau()
        self.estado = {"fase": "base", "ronda_base": 0, "ronda_en_curso": None, "paso": 0,
                       "mejor_loss": float("inf"), "mejor_ronda": 0, "sin_mejora": 0, "ronda_fin_base": None,
                       "variante": 0, "ciclo": 0, "rondas_hechas_variante": 0, "ronda_en_ciclo": 0,
                       "segundos": 0.0, "fallas_imagen": 0, "losses_no_finitas": 0, "historial": [],
                       "huella_planificador": {}, "sesiones": 0}
        self._retomar()
        self.estado["sesiones"] += 1
        self._log({"tipo": "inicio_sesion", "sesion": self.estado["sesiones"], "fase": self.estado["fase"],
                   "dispositivo": str(self.device), "train_imagenes": len(tr), "train_panoramas": int(n_panos),
                   "commit": commit_git()})

    # ---------------------------------------------------------- utilidades
    def _registrar_config(self):
        ruta = self.carpeta / "config.json"
        datos = {**self.cfg.a_dict(), "huella": self.cfg.huella()}
        if ruta.exists():
            previa = json.loads(ruta.read_text(encoding="utf-8"))
            if previa.get("huella") not in self.cfg.huellas_validas():
                raise ValueError(f"{self.carpeta} es de otra configuración (huella {previa.get('huella')} vs "
                                 f"{datos['huella']}). Usá otra carpeta_corrida o la misma configuración.")
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _log(self, registro: dict):
        registro = {"fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"), **registro}
        with open(self.carpeta / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
        if registro["tipo"] != "paso":
            print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in registro.items() if k != "fecha"), flush=True)

    def _nuevo_planificador(self) -> Planificador:
        return Planificador(self.celda_pano, self.inicio_filas, self.n_filas, self.num_clases,
                            self.cfg.tamano_ronda, self.cfg.semilla, self.cfg.r_max)

    def _nuevo_optimizador(self, escala_lr: float = 1.0):
        m = self.modelo
        return torch.optim.AdamW([
            {"params": m.fc.parameters(), "lr": self.cfg.lr_cabeza * escala_lr},
            {"params": [p for n, p in m.named_parameters() if not n.startswith("fc.")],
             "lr": self.cfg.lr_backbone * escala_lr},
        ], weight_decay=self.cfg.weight_decay)

    def _nuevo_plateau(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizador, factor=self.cfg.factor_plateau,
                                                          patience=self.cfg.paciencia_plateau)

    def _nuevo_coseno(self, pasos_totales: int):
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizador, lambda paso: 0.5 * (1 + math.cos(math.pi * min(paso, pasos_totales) / pasos_totales)))

    def _pasos_ronda(self) -> int:
        return math.ceil(self.cfg.tamano_ronda / self.cfg.batch_size)

    def plan_ronda(self, numero: int) -> np.ndarray:
        """Filas de la ronda `numero` (1, 2, ...) del planificador; reconstruye el estado si hace falta."""
        if self.planificador.rondas > numero - 1:
            self.planificador = self._nuevo_planificador()
        self.planificador.avanzar_hasta(numero - 1)
        huella = self.planificador.huella()
        esperada = self.estado["huella_planificador"].get(str(numero - 1))
        if esperada is not None and esperada != huella:
            raise RuntimeError(f"La reconstrucción del planificador no coincide en la ronda {numero - 1}.")
        self.estado["huella_planificador"][str(numero - 1)] = huella
        return self.planificador.siguiente_ronda()

    def fijar_backbone(self, congelado: bool):
        for n, p in self.modelo.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = not congelado
        self.backbone_congelado = congelado

    # ---------------------------------------------------------- checkpoints
    def guardar(self, motivo: str):
        estado_rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        guardar_atomico({"modelo": self.modelo.state_dict(), "optimizador": self.optimizador.state_dict(),
                         "scheduler": self.scheduler.state_dict(),
                         "tipo_scheduler": type(self.scheduler).__name__, "estado": self.estado, "rng": estado_rng,
                         "huella_config": self.cfg.huella(), "torch": torch.__version__, "commit": commit_git()},
                        self.ruta_ultimo, self.ruta_anterior)
        self.ultimo_guardado = time.time()
        self._log({"tipo": "checkpoint", "motivo": motivo, "fase": self.estado["fase"], "paso": self.estado["paso"]})

    def _retomar(self):
        for ruta in (self.ruta_ultimo, self.ruta_anterior):
            if not ruta.exists():
                continue
            try:
                ck = torch.load(ruta, map_location="cpu", weights_only=False)
            except Exception as e:  # noqa: BLE001 — archivo dañado: probar con el respaldo.
                print(f"No se pudo leer {ruta.name} ({e}); se prueba el respaldo.")
                continue
            if ck["huella_config"] not in self.cfg.huellas_validas():
                raise ValueError("El checkpoint es de otra configuración.")
            self.estado = ck["estado"]
            fijadas = self.estado.get("variantes_ciclos")
            if self.estado["fase"] != "base" and fijadas is None:  # Checkpoint anterior al cambio: hash legado.
                if ck["huella_config"] != self.cfg.huella(legado=True):
                    raise ValueError("El checkpoint es de otra configuración (variantes_ciclos distintas).")
            elif self.estado["fase"] != "base" and fijadas != self.cfg.variantes_ciclos:
                raise ValueError(f"variantes_ciclos ya quedó fijado al terminar la fase base ({fijadas}); "
                                 f"no se puede cambiar a {self.cfg.variantes_ciclos}.")
            self.modelo.load_state_dict(ck["modelo"])
            if self.estado["fase"] == "ciclos" and ck["tipo_scheduler"] == "LambdaLR":
                self.optimizador = self._nuevo_optimizador()
                self.optimizador.load_state_dict(ck["optimizador"])
                self.scheduler = self._nuevo_coseno(self.estado["pasos_ciclo"])
            else:
                self.optimizador.load_state_dict(ck["optimizador"])
                self.scheduler = self._nuevo_plateau()
            self.scheduler.load_state_dict(ck["scheduler"])
            rng = ck["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            print(f"Retomando desde {ruta.name}: fase {self.estado['fase']}, paso {self.estado['paso']}, "
                  f"ronda base {self.estado['ronda_base']}")
            return
        print("Corrida nueva.")

    def _controlar_corte(self):
        cfg = self.cfg
        if self.senal_recibida:
            self.guardar("senal")
            raise Pausa("Señal de corte recibida.")
        if cfg.max_horas and time.time() - self.inicio_sesion > cfg.max_horas * 3600:
            self.guardar("max_horas")
            raise Pausa(f"Se alcanzó max_horas = {cfg.max_horas}.")
        if time.time() - self.ultimo_guardado > cfg.minutos_checkpoint * 60:
            self.guardar("periodico")

    # ---------------------------------------------------------- una ronda de entrenamiento
    def entrenar_ronda(self, numero_plan: int, etiqueta: str):
        cfg, tr = self.cfg, self.tablas["train"]
        filas = self.plan_ronda(numero_plan)
        if len(filas) == 0:
            return None
        en_curso = self.estado["ronda_en_curso"]
        if en_curso is None or en_curso["numero_plan"] != numero_plan:
            objetivo = pd.Series(tr.archivo[filas]).value_counts().to_dict()
            en_curso = self.estado["ronda_en_curso"] = {"numero_plan": numero_plan, "etiqueta": etiqueta,
                                                         "objetivo": objetivo, "consumidos": {},
                                                         "suma_loss": 0.0, "aciertos": 0, "n": 0,
                                                         "aciertos_grupo": [0, 0, 0], "n_grupo": [0, 0, 0]}
        terminados = {a for a, n in en_curso["consumidos"].items() if n >= en_curso["objetivo"][a]}
        pendientes = ~np.isin(tr.archivo[filas], list(terminados))
        orden = list(np.random.default_rng([cfg.semilla, 2, numero_plan]).permutation(sorted(en_curso["objetivo"])))
        tareas = tareas_por_archivo(tr.archivo[filas][pendientes], tr.fila[filas][pendientes],
                                    filas[pendientes], orden)
        flujo = FlujoMuestras(tareas, self.lector, self.transform_train, cfg.buffer_mezcla,
                              semilla=cfg.semilla * 100_000 + numero_plan, tamano=cfg.tamano_imagen,
                              corrupta_cada=cfg.prueba_corrupta_cada)
        loader = DataLoader(flujo, batch_size=cfg.batch_size, num_workers=min(cfg.num_workers, len(tareas)),
                            pin_memory=self.device.type == "cuda")
        lat_t, lon_t = torch.tensor(tr.lat, dtype=torch.float32), torch.tensor(tr.lon, dtype=torch.float32)
        sam = SAM(self.optimizador, cfg.rho_sam)
        self.modelo.train()
        if self.backbone_congelado:
            for nombre, modulo in self.modelo.named_children():
                if nombre != "fc":
                    modulo.eval()
        inicio, t_datos, t_ultimo, imgs_ventana = time.time(), 0.0, time.time(), 0
        for imagenes, ids, ok in loader:
            t_datos += time.time() - t_ultimo
            ids_np = ids.numpy()
            for a, n in pd.Series(tr.archivo[ids_np]).value_counts().items():
                en_curso["consumidos"][a] = en_curso["consumidos"].get(a, 0) + int(n)
            if not ok.all():
                self.estado["fallas_imagen"] += int((~ok).sum())
                self._log({"tipo": "imagenes_fallidas", "cantidad": int((~ok).sum()),
                           "total": self.estado["fallas_imagen"]})
                if self.estado["fallas_imagen"] > cfg.max_fallas_imagen:
                    self.guardar("demasiadas_fallas")
                    raise RuntimeError(f"Más de {cfg.max_fallas_imagen} imágenes no decodificaron.")
                imagenes, ids = imagenes[ok], ids[ok]
                if len(ids) < 2:
                    t_ultimo = time.time()
                    continue
            imagenes = imagenes.to(self.device, non_blocking=True)
            dist = haversine_torch(lat_t[ids], lon_t[ids], self.proto_t[:, 0].cpu(), self.proto_t[:, 1].cpu())
            dist = dist.to(self.device, non_blocking=True)
            celdas = torch.as_tensor(tr.celda[ids.numpy()], device=self.device)
            self.estado["paso"] += 1
            logits = self.modelo(imagenes)
            loss = loss_haversine(logits, dist, cfg.tau_km, cfg.gamma_focal)
            if cfg.prueba_nan_en_paso and self.estado["paso"] == cfg.prueba_nan_en_paso:
                loss = loss * float("nan")
            if not torch.isfinite(loss):
                self._loss_no_finita()
                t_ultimo = time.time()
                continue
            loss.backward()
            sam.primer_paso()
            with bn_sin_actualizar(self.modelo):
                loss_perturbada = loss_haversine(self.modelo(imagenes), dist, cfg.tau_km, cfg.gamma_focal)
            if not torch.isfinite(loss_perturbada):
                sam.deshacer()
                self._loss_no_finita()
                t_ultimo = time.time()
                continue
            loss_perturbada.backward()
            sam.segundo_paso()
            self.estado["losses_no_finitas"] = 0
            if isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR):
                self.scheduler.step()
                self.estado["paso_en_ciclo"] = self.estado.get("paso_en_ciclo", 0) + 1
            n = len(ids)
            pred = logits.detach().argmax(dim=1)
            en_curso["suma_loss"] += float(loss.detach()) * n
            en_curso["aciertos"] += int((pred == celdas).sum())
            en_curso["n"] += n
            grupos = self.grupo_celda[celdas.cpu().numpy()]
            for g in range(3):
                en_curso["aciertos_grupo"][g] += int(((pred == celdas).cpu().numpy() & (grupos == g)).sum())
                en_curso["n_grupo"][g] += int((grupos == g).sum())
            imgs_ventana += n
            if self.estado["paso"] % cfg.pasos_log == 0:
                transcurrido = time.time() - inicio
                self._log({"tipo": "paso", "paso": self.estado["paso"], "ronda": etiqueta,
                           "loss": en_curso["suma_loss"] / max(en_curso["n"], 1),
                           "img_s": imgs_ventana / max(transcurrido, 1e-9), "frac_espera_datos": t_datos / transcurrido,
                           "lr_cabeza": self.optimizador.param_groups[0]["lr"], "ram_gb": ram_usada_gb(),
                           "gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None})
                print(f"  [{etiqueta}] paso {self.estado['paso']}: loss {en_curso['suma_loss'] / en_curso['n']:.4f}, "
                      f"{imgs_ventana / transcurrido:.0f} img/s, espera de datos {100 * t_datos / transcurrido:.0f}%, "
                      f"{en_curso['n']}/{len(filas)} muestras", flush=True)
            self.estado["segundos"] += time.time() - t_ultimo
            self._controlar_corte()
            t_ultimo = time.time()
        resultado = {"loss_train": en_curso["suma_loss"] / max(en_curso["n"], 1),
                     "top1_train": en_curso["aciertos"] / max(en_curso["n"], 1),
                     **{f"top1_train_celdas_{nombre}": en_curso["aciertos_grupo"][g] / max(en_curso["n_grupo"][g], 1)
                        for g, nombre in enumerate(("chicas", "medianas", "grandes"))},
                     "muestras": en_curso["n"], "minutos": (time.time() - inicio) / 60}
        self.estado["ronda_en_curso"] = None
        return resultado

    def _loss_no_finita(self):
        self.optimizador.zero_grad(set_to_none=True)
        self.estado["losses_no_finitas"] += 1
        self._log({"tipo": "loss_no_finita", "paso": self.estado["paso"], "seguidas": self.estado["losses_no_finitas"]})
        if self.estado["losses_no_finitas"] >= self.cfg.max_losses_no_finitas:
            self.guardar("losses_no_finitas")
            raise RuntimeError(f"{self.cfg.max_losses_no_finitas} losses no finitas seguidas.")

    # ---------------------------------------------------------- evaluación
    @torch.inference_mode()
    def logits(self, nombre_tabla: str, modelo: nn.Module | None = None) -> np.ndarray:
        t = self.tablas[nombre_tabla]
        ids = np.arange(len(t))
        flujo = FlujoMuestras(tareas_por_archivo(t.archivo, t.fila, ids), self.lector_eval, self.transform_val,
                              tamano=self.cfg.tamano_imagen)
        loader = DataLoader(flujo, batch_size=64, num_workers=min(self.cfg.num_workers, len(set(t.archivo))),
                            pin_memory=self.device.type == "cuda")
        modelo = modelo or self.modelo
        modelo.eval()
        salida = np.full((len(t), self.num_clases), np.nan, dtype=np.float32)
        for imagenes, i, ok in loader:
            salida[i.numpy()] = modelo(imagenes.to(self.device, non_blocking=True)).float().cpu().numpy()
            if not ok.all():
                self._log({"tipo": "imagenes_fallidas_eval", "tabla": nombre_tabla, "cantidad": int((~ok).sum())})
        if np.isnan(salida).any():
            raise RuntimeError(f"Faltan predicciones en {nombre_tabla}.")
        return salida

    def evaluar(self, nombre_tabla: str, modelo=None) -> tuple[dict, np.ndarray]:
        t = self.tablas[nombre_tabla]
        z = self.logits(nombre_tabla, modelo)
        dist = haversine_np(t.lat[:, None], t.lon[:, None], self.prototipos[None, :, 0], self.prototipos[None, :, 1])
        m = metricas(z, t.celda, dist, self.dist_proto, self.cfg.tau_km)
        acierto = z.argmax(axis=1) == t.celda
        for g, nombre in enumerate(("chicas", "medianas", "grandes")):
            sel = self.grupo_celda[t.celda] == g
            m[f"top1_celdas_{nombre}"] = float(acierto[sel].mean()) if sel.any() else float("nan")
        return m, z

    def evaluar_ronda(self, etiqueta: str, resultado_train: dict) -> dict:
        val, _ = self.evaluar("val_1vista")
        fila = {"ronda": etiqueta, "fase": self.estado["fase"], "paso": self.estado["paso"], **resultado_train,
                **{f"val_{k}": v for k, v in val.items()},
                "lr_cabeza": self.optimizador.param_groups[0]["lr"], "horas_acumuladas": self.estado["segundos"] / 3600}
        if "val_global" in self.tablas:
            vg, _ = self.evaluar("val_global")
            fila.update({f"valglobal_{k}": vg[k] for k in ("top1", "km_mediana", "ce_haversine")})
        self.estado["historial"].append(fila)
        pd.DataFrame(self.estado["historial"]).to_csv(self.carpeta / "historial.csv", index=False)
        self._log({"tipo": "ronda", **{k: v for k, v in fila.items() if not k.startswith("top1_train_celdas")}})
        return fila

    # ---------------------------------------------------------- fases
    def fase_base(self):
        cfg, e = self.cfg, self.estado
        while e["fase"] == "base":
            numero = e["ronda_base"] + 1
            cabeza = numero <= cfg.rondas_cabeza
            self.fijar_backbone(cabeza)
            resultado = self.entrenar_ronda(numero, f"base_{numero}")
            if resultado is None:
                self._log({"tipo": "evento", "detalle": "No quedan datos para planificar más rondas."})
                fila = None
            else:
                fila = self.evaluar_ronda(f"base_{numero}", resultado)
                loss_val = fila["val_ce_haversine"]
                if not math.isfinite(loss_val):
                    raise RuntimeError("Loss de validación no finita.")
                if loss_val < e["mejor_loss"]:
                    e["mejor_loss"], e["mejor_ronda"], e["sin_mejora"] = loss_val, numero, 0
                    guardar_atomico({"modelo": self.modelo.state_dict(), "ronda": numero, "metricas_val": fila,
                                     "config": cfg.a_dict()}, self.carpeta / "mejor.pt")
                elif not cabeza:
                    e["sin_mejora"] += 1
                if not cabeza:
                    self.scheduler.step(loss_val)
            e["ronda_base"] = numero
            if fila is None or e["sin_mejora"] >= cfg.paciencia or numero >= cfg.max_rondas_base:
                e["fase"], e["ronda_fin_base"] = "ciclos", numero
                e["variantes_ciclos"] = cfg.variantes_ciclos  # Desde acá ya no se pueden cambiar.
                self._log({"tipo": "fin_base", "mejor_ronda": e["mejor_ronda"], "mejor_loss": e["mejor_loss"],
                           "rondas": numero})
            self.guardar("fin_ronda")

    def fase_ciclos(self):
        cfg, e = self.cfg, self.estado
        self.fijar_backbone(False)
        while e["fase"] == "ciclos" and e["variante"] < len(cfg.variantes_ciclos):
            var = cfg.variantes_ciclos[e["variante"]]
            rondas_ciclo = max(1, round(var["fraccion"] * e["mejor_ronda"]))
            carpeta_snap = self.carpeta / "snapshots"
            while e["ciclo"] < var["ciclos"]:
                if e["ronda_en_ciclo"] == 0 and e.get("paso_en_ciclo", 0) == 0 and e["ronda_en_curso"] is None:
                    if e["ciclo"] == 0:  # Cada variante arranca del mismo mejor checkpoint de la fase base.
                        self.modelo.load_state_dict(torch.load(self.carpeta / "mejor.pt", map_location="cpu",
                                                               weights_only=False)["modelo"])
                    self.optimizador = self._nuevo_optimizador()
                    e["pasos_ciclo"] = rondas_ciclo * self._pasos_ronda()
                    self.scheduler = self._nuevo_coseno(e["pasos_ciclo"])
                    e["paso_en_ciclo"] = 0
                while e["ronda_en_ciclo"] < rondas_ciclo:
                    numero_plan = e["ronda_fin_base"] + e["rondas_hechas_variante"] + 1
                    etiqueta = f"{var['nombre']}_c{e['ciclo'] + 1}_r{e['ronda_en_ciclo'] + 1}"
                    resultado = self.entrenar_ronda(numero_plan, etiqueta)
                    if resultado is None:
                        raise RuntimeError("No quedan datos para los ciclos.")
                    self.evaluar_ronda(etiqueta, resultado)
                    e["rondas_hechas_variante"] += 1
                    e["ronda_en_ciclo"] += 1
                    self.guardar("fin_ronda")
                nombre = f"snapshot_{var['nombre']}_{e['ciclo'] + 1}.pt"
                guardar_atomico({"modelo": self.modelo.state_dict(), "variante": var, "ciclo": e["ciclo"] + 1,
                                 "rondas_por_ciclo": rondas_ciclo, "config": cfg.a_dict()}, carpeta_snap / nombre)
                self._log({"tipo": "snapshot", "archivo": nombre, "rondas_por_ciclo": rondas_ciclo})
                e["ciclo"] += 1
                e["ronda_en_ciclo"], e["paso_en_ciclo"] = 0, 0
                self.guardar("fin_ciclo")
            e["variante"] += 1
            e["ciclo"], e["rondas_hechas_variante"] = 0, 0
            self.guardar("fin_variante")
        if e["fase"] == "ciclos":
            e["fase"] = "evaluacion_final"
            self.guardar("fin_ciclos")

    def fase_evaluacion_final(self):
        carpeta = self.carpeta / "logits"
        carpeta.mkdir(exist_ok=True)
        modelos = [self.carpeta / "mejor.pt", *sorted((self.carpeta / "snapshots").glob("snapshot_*.pt"))]
        tablas = ["val"] + (["val_global"] if "val_global" in self.tablas else [])
        resumen = []
        for ruta in modelos:
            m = construir_modelo(self.num_clases, self.cfg.dropout, preentrenado=False).to(self.device)
            m.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=False)["modelo"])
            for tabla in tablas:
                destino = carpeta / f"{ruta.stem}__{tabla}.npy"
                if not destino.exists():
                    met, z = self.evaluar(tabla, m)
                    np.save(destino, z)
                else:
                    t = self.tablas[tabla]
                    dist = haversine_np(t.lat[:, None], t.lon[:, None], self.prototipos[None, :, 0],
                                        self.prototipos[None, :, 1])
                    met = metricas(np.load(destino), t.celda, dist, self.dist_proto, self.cfg.tau_km)
                resumen.append({"modelo": ruta.stem, "tabla": tabla, **met})
                self._controlar_corte()
        pd.DataFrame(resumen).to_csv(self.carpeta / "metricas_finales.csv", index=False)
        self.estado["fase"] = "terminado"
        self.guardar("terminado")

    def ejecutar(self) -> str:
        def al_recibir_senal(*_):
            self.senal_recibida = True

        try:
            signal.signal(signal.SIGTERM, al_recibir_senal)
        except (ValueError, AttributeError):  # Fuera del hilo principal (algunos entornos de notebook).
            pass
        try:
            if self.estado["fase"] == "base":
                self.fase_base()
            if self.estado["fase"] == "ciclos":
                self.fase_ciclos()
            if self.estado["fase"] == "evaluacion_final":
                self.fase_evaluacion_final()
        except Pausa as p:
            self._log({"tipo": "pausa", "detalle": str(p)})
            return "pausado"
        except KeyboardInterrupt:
            self.guardar("interrupcion")
            self._log({"tipo": "pausa", "detalle": "Interrumpido por el usuario."})
            return "pausado"
        self._log({"tipo": "fin", "horas_entrenamiento": self.estado["segundos"] / 3600})
        return self.estado["fase"]


def entrenar(cfg: Config) -> str:
    """Entrena (o retoma) la corrida de `cfg.carpeta_corrida`. Devuelve 'terminado' o 'pausado'."""
    random.seed(cfg.semilla)
    np.random.seed(cfg.semilla)
    torch.manual_seed(cfg.semilla)
    return Corrida(cfg).ejecutar()
