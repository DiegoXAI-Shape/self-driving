# AGENTS.md — Reglas y Convenciones del Proyecto Helioskrill

Este documento establece las directrices técnicas, estándares de coordenadas y arquitectura del proyecto **Helioskrill**.

---

## 1. Stack Tecnológico y Dependencias Clave
* **Lenguajes de programación:** Python, C++ (aún sin uso).
* **Framework Principal:** PyTorch 2.0+ con aceleración por GPU (CUDA).
* **Backbone Visual:** Meta DINOv2 Small (`dinov2_vits14`) adaptado mediante **LoRA** ($r=8, \alpha=16$) enfocado en capas `qkv`.
* **Recurrencia Temporal:** Mamba SSM (`mamba-ssm`), utilizando `TemporalMamba` sobre cuadrículas BEV reducidas ($100 \times 100$).
* **Entorno de Simulación:** CARLA 0.9.15 (ejecutado en Windows con cliente en WSL 2 Ubuntu).

---

## 2. Convenciones de Arquitectura y Modularización

El código debe mantenerse limpio y estrictamente modularizado en `models/`:

* **`models/utils/perception_blocks.py`:**
  * `DINOv2EncoderLoRA`: Extracción de características visuales 2D ($384$ canales).
  * `CameraBEVProjectionV2`: Proyección IPM de 2D a espacio BEV 3D.
  * `CameraBEVNeck`: Cuello convolucional de refinamiento BEV exclusivo de cámaras ($64 \to 128$ canales).
* **`models/utils/mamba_blocks.py`:**
  * `TemporalMambaBlock` y `TemporalMamba`: Bloques de recurrencia secuencial 1D para secuencias temporales ($S=5$).
* **`models/modules/BEV_perception.py`:**
  * Red principal `BEVPerceptionNetV2`: Pipeline **exclusivo de cámaras** (sin fusión LiDAR en la red neuronal).
* **`models/modules/BEV_planning.py`:**
  * `MultiHeadBEVPlanningHead`: Cabeza multitarea con 3 salidas:
    1. Polinomio cinemático de 5to grado (10 puntos futuros $X, Y, Z$).
    2. Orientación trigonométrica ($Yaw: \sin \theta, \cos \theta$).
    3. Pedales (`throttle`, `brake`) y velocidad objetivo (`speed_mps`).

---

## 3. Estándares de Coordenadas y Proyección BEV

* **Sistema de CARLA (Vehículo Ego):** Mano Izquierda ($+X$ Adelante, $+Y$ Derecha, $+Z$ Arriba).
* **Sistema de Cámara (OpenCV / PyTorch):** Mano Derecha ($+X$ Derecha, $+Y$ Abajo, $+Z$ Profundidad).
* **Conversión de Sistemas:** Matriz $R_{\text{default}}$ para transformar de Mano Izquierda (CARLA) a Mano Derecha (Cámara).
* **Grilla BEV (Asimétrica Estilo Tesla):** Matriz de $200 \times 200$ casillas con cobertura asimétrica de $X \in [-10\text{m}, +40\text{m}]$ (40m hacia adelante) y $Y \in [-25\text{m}, +25\text{m}]$ a una resolución de $0.25\text{m}/\text{píxel}$.
  * La posición central del vehículo $(0.0, 0.0)$ mapea **estrictamente a la columna 100, fila 160 (`col=100, row=160`)**.

---

## 4. Estándar de Datos y Recolección (DAgger / HydraSkrill)

* **Arreglo de Cámaras:** 8 Cámaras RGB + 8 Cámaras de Profundidad + 8 Cámaras Semánticas ($800 \times 600$, $\text{FOV}=100^\circ$) montadas según configuración Tesla Model 3.
* **Formato de Control (`control.csv`):** `[frame, timestamp_sec, throttle, brake, steer, hand_brake, reverse, is_recovery]`.
* **Perturbaciones Suaves DAgger:** Desvíos laterales acotados ($\pm 0.15$ a $\pm 0.25$ rad) para registrar correcciones de carril fluidas sin colisiones fatales.
* **Distancia de Seguimiento en CARLA Traffic Manager:** Configurar `distance_to_leading_vehicle` a $3.5\text{m}$ (equivalente a $2-3\text{m}$ de margen real entre parachoques).
* **Estrategia de Recolección Secuencial:** Recolectar datos por lotes independientes (`--mode normal` para $75-80\%$ de datos expertos limpios y `--mode dagger` para $20-25\%$ de maniobras de recuperación).
* **Ponderación de Pérdida:** Multiplicador de $1.5\times / 3.0\times$ en el cargador de datos (`dataset.py`) para giros, frenado y muestras de recuperación (`is_recovery == 1.0`).

---

## 5. Reglas de Inferencia en Bucle Cerrado (Closed-Loop)

* **Red Neuronal:** Funciona $100\%$ solo con imágenes de cámara (sin entrada LiDAR a PyTorch).
* **Escudo de Freno de Emergencia:** En `run_carla_closed_loop.py`, el sensor LiDAR físico permanece montado en CARLA **exclusivamente como regla determinista de seguridad** (frena al 100% si detecta obstáculos a menos de $3.5\text{m}$).

## 6. Reglas de modificaciones

* **Cambios**: SIEMPRE que vayas a realizar cambios, platícame sobre ellos y dime el por qué, intenta también explicarmelos lo más abstracto posibe
* **Refactorización**: SIEMPRE que vayas a realizar cambios en el código, revisa antes otros archivos para ver si hay algo similar que puedas usar como base y adaptarlo a tus necesidades, no reinventes la rueda.
* **Cita recursos**: SIEMPRE que vayas a realizar cambios en el código, intenta citar recursos de donde sacaste la idea, ya sean otros repositorios, papers, documentación, etc. para tener un mayor entendimiento sobre lo que estás haciendo.  
* **GitHub**: Siempre que vayas a realizar un cambio genera commits atómicos y bien explicados, es decir, si modificas un archivo, me explicas el qué cambiaste y así lo subo al GitHub yo.
