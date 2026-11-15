"""Configuración de una corrida. Se guarda en la carpeta de la corrida y su hash se controla al retomar."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# Campos que se pueden cambiar entre sesiones sin invalidar la corrida (no cambian qué se entrena).
CAMPOS_SIN_HASH = {"carpeta_corrida", "num_workers", "minutos_checkpoint", "max_horas", "pasos_log",
                   "prueba_nan_en_paso", "prueba_corrupta_cada", "fuente", "datos_local", "hf_cache",
                   "borrar_shards_hf", "val_global", "manifest", "splits", "prototipos",
                   "optimizaciones_velocidad", "channels_last"}

# Campos agregados después de la corrida de 2M: en su valor por defecto (el de esa corrida) no entran en el hash,
# así las corridas anteriores conservan su huella y se pueden seguir analizando o retomando.
CAMPOS_SIN_HASH_POR_DEFECTO = {"tarea", "loss", "augmentation", "conteo_paises", "min_filas_pais"}


@dataclass
class Config:
    carpeta_corrida: str  # Todo lo de la corrida (checkpoints, logs, snapshots). En Colab, una carpeta de Drive.

    # Datos.
    manifest: str = "datasets/manifest_2m/manifest.parquet"
    splits: str = "datasets/manifest_2m/splits.parquet"
    prototipos: str = "src/notebooks/preprocessing/assets/prototipos_train-completo_k70.csv"
    sha256_prototipos: str = "e3fcfd92cb1f3d1dd61806167f73e1249c6a8bc53741e54c576f3151197f693c"
    fuente: str = "local"  # "local" (parquets en disco) o "hf" (se bajan del repo de HF, de a uno).
    datos_local: str = "../world-streetview-500k/data"
    hf_repo: str = "josefbednar/world-streetview-500k"
    hf_revision: str | None = None  # Fijar un commit para la corrida de 2M (que el dataset no cambie a mitad).
    hf_subcarpeta: str = "data"
    hf_cache: str = "/content/shards"
    borrar_shards_hf: bool = True  # Borrar cada parquet bajado al terminarlo (el disco de Colab es limitado).
    shards_train: str | None = None  # Rango "a-b" de números de shard para train (prueba); None = todos.
    val_global: str | None = "datasets/samples/data/validation/validation.parquet"  # Validación de los 20k.
    limite_val: int | None = None  # Solo pruebas rápidas: evaluar con los primeros N panoramas de cada validación.

    # Qué se predice. "celdas": las K celdas de `prototipos`. "paises": los países con al menos `min_filas_pais`
    # filas en `conteo_paises` (el mismo criterio que el clasificador por país de 20k: 71 países); el train se
    # restringe a esos países, las rondas se balancean por país y las imágenes de validación de otros países
    # cuentan como error. Para medir km, cada país se representa con su medoide (ver clases_y_tablas).
    tarea: str = "celdas"
    conteo_paises: str = "src/notebooks/preprocessing/assets/countries_count.json"
    min_filas_pais: int = 1000

    # Muestreo (ver sampler.py).
    tamano_ronda: int = 40_000
    # Tope opcional de pasadas por panorama en toda la corrida. None = sin tope: el único límite es una aparición
    # por ronda (con tope, las celdas chicas desaparecerían de las últimas rondas y de los ciclos de snapshot).
    r_max: int | None = None
    buffer_mezcla: int = 4_000  # Imágenes (bytes JPEG) en el shuffle buffer, sumando todos los workers.

    # Modelo y optimización (receta de S42 + focal haversine).
    loss: str = "haversine"  # "haversine": CE contra softmax(-d/τ) (focal si gamma_focal > 0). "ce": CE simple.
    augmentation: bool = True  # False: train con las mismas transformaciones que validación.
    tamano_imagen: int = 224
    dropout: float = 0.3
    tau_km: float = 300.0
    gamma_focal: float = 2.0
    rho_sam: float = 0.05  # 0 = sin SAM (un solo forward/backward por paso).
    lr_cabeza: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    rondas_cabeza: int = 1  # Rondas iniciales con el backbone congelado.
    paciencia_plateau: int = 2
    factor_plateau: float = 0.1
    paciencia: int = 4  # Early stopping, en rondas sin mejora de la CE haversine de validación.
    max_rondas_base: int = 200

    # Snapshots: variantes que arrancan todas del mismo mejor checkpoint. fraccion = largo de cada ciclo como
    # fracción de las rondas que tardó el modelo base en llegar a su mejor ronda. Lista vacía = sin snapshots.
    variantes_ciclos: list = field(default_factory=lambda: [
        {"nombre": "corto", "ciclos": 4, "fraccion": 0.15},
        {"nombre": "largo", "ciclos": 3, "fraccion": 0.40},
    ])

    # Velocidad sin cambiar la receta: cudnn.benchmark, AdamW fusionado e imágenes en uint8 normalizadas en la
    # GPU (mismas cuentas que ToTensor + Normalize). Se puede cambiar entre sesiones.
    optimizaciones_velocidad: bool = True
    # channels_last aparte y apagado: en fp32 sin tensor cores (GTX 1650) hizo el entrenamiento ~7x más lento.
    # Solo conviene con AMP o TF32 en GPUs que lo aprovechen; medirlo antes de activarlo.
    channels_last: bool = False

    # Robustez y operación.
    num_workers: int = 3
    minutos_checkpoint: float = 15.0
    max_horas: float | None = None  # Por sesión: al llegar, guarda y sale (se retoma relanzando).
    max_fallas_imagen: int = 200  # Imágenes que no decodifican antes de abortar.
    max_losses_no_finitas: int = 10  # Seguidas, antes de abortar.
    pasos_log: int = 200
    semilla: int = 42

    # Inyección de fallas para probar la robustez (None en corridas reales).
    prueba_nan_en_paso: int | None = None
    prueba_corrupta_cada: int | None = None

    def __post_init__(self):
        # Validar al cargar: un error acá recién explotaría al terminar la fase base, horas después.
        if self.tarea not in ("celdas", "paises"):
            raise ValueError(f"tarea debe ser 'celdas' o 'paises', no {self.tarea!r}")
        if self.loss not in ("haversine", "ce"):
            raise ValueError(f"loss debe ser 'haversine' o 'ce', no {self.loss!r}")
        if self.rho_sam < 0:
            raise ValueError(f"rho_sam debe ser >= 0, no {self.rho_sam}")
        v = self.variantes_ciclos
        if not isinstance(v, list):
            raise ValueError(f"variantes_ciclos debe ser una lista (vacía = sin snapshots), no {v!r}")
        nombres = set()
        for var in v:
            if (not isinstance(var, dict) or set(var) != {"nombre", "ciclos", "fraccion"}
                    or not isinstance(var["nombre"], str) or not isinstance(var["ciclos"], int)
                    or var["ciclos"] < 1 or not isinstance(var["fraccion"], (int, float)) or var["fraccion"] <= 0):
                raise ValueError(f"Variante inválida: {var!r} (se espera {{nombre: str, ciclos: int >= 1, "
                                 f"fraccion: número > 0}})")
            if var["nombre"] in nombres:
                raise ValueError(f"Nombre de variante repetido: {var['nombre']}")
            nombres.add(var["nombre"])

    def a_dict(self) -> dict:
        return asdict(self)

    def huella(self, legado: bool = False) -> str:
        """Hash de lo que define la corrida. variantes_ciclos queda afuera: se puede cambiar mientras la corrida
        siga en la fase base y se fija al empezar los ciclos (lo controla Corrida). legado=True reproduce el hash
        anterior a ese cambio, para retomar corridas empezadas antes."""
        excluidos = CAMPOS_SIN_HASH if legado else CAMPOS_SIN_HASH | {"variantes_ciclos"}
        por_defecto = {f.name: f.default for f in fields(self) if f.name in CAMPOS_SIN_HASH_POR_DEFECTO}
        excluidos = excluidos | {k for k, d in por_defecto.items() if getattr(self, k) == d}
        relevantes = {k: v for k, v in self.a_dict().items() if k not in excluidos}
        return hashlib.sha256(json.dumps(relevantes, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def huellas_validas(self) -> set[str]:
        return {self.huella(), self.huella(legado=True)}

    @classmethod
    def desde_json(cls, ruta: str | Path, **cambios) -> "Config":
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        datos.update(cambios)
        conocidos = {f.name for f in fields(cls)}
        desconocidos = set(datos) - conocidos
        if desconocidos:
            raise ValueError(f"Campos desconocidos en la configuración: {sorted(desconocidos)}")
        return cls(**datos)
