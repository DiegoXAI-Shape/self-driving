# Helioskrill — Experimento No. 2: DINOv2 + LoRA + GroupNorm + Temporal Mamba

## 1. Resumen Ejecutivo
En el Experimento 2 se logró un avance crítico en el proyecto Helioskrill al transicionar de un encoder visual entrenado desde cero (Experimento 1) hacia un modelo con **pesos pre-entrenados Meta DINOv2 Small (`dinov2_vits14`)** adaptado mediante **LoRA ($r=8, \alpha=16$)** y normalizado con **`GroupNorm(num_groups=16)`**.

### Métricas Clave Obtenidas (Validación Cruzada en Rutas Desconocidas)
* **Mean ADE (Error Medio de Trayectoria):** **0.49 metros** (Sub-métrico: 49 cm en 5.0 segundos).
* **Mean FDE (Error Final a 5.0s):** **0.87 metros** (Menos de 1 metro al final de la trayectoria).
* **Mean Velocity Error:** **0.21 m/s** (Excelente consistencia dinámica).
* **Mean Acceleration Error:** **1.35 m/s²**.
* **Mean Yaw Error (Orientación):** **52.15 grados**.
* **Mejora frente al Experimento 1:** **¡17.1x de reducción en el error de trayectoria!** ($8.42\text{m} \to 0.49\text{m}$).
* **Consumo de VRAM:** **~3.8 GB** (Cero memoria PCIe swapped a RAM).
* **Parámetros Entrenables:** **1,644,168** ($6.94\%$ del total de $23.7\text{M}$).

---

## 2. Arquitectura Utilizada (`BEVPerceptionNetV2`)

1. **Backbone Visual:** `dinov2_vits14` pre-entrenado. Módulos LoRA inyectados únicamente en las proyecciones `qkv` del mecanismo de atención de los 12 bloques ViT.
2. **Proyección BEV (IPM):** Transforma mapas de características 2D de $N=8$ cámaras periféricas al plano 3D de Vista de Pájaro ($400 \times 400$ grilla a $0.25\text{m/pixel}$).
3. **Módulo Temporal:** `TemporalMamba` ($dim=64, L=2$) aplicando modelos de espacio de estados (SSM 1D) a lo largo de la dimensión temporal.
4. **Fusión & Regresión:** Fusión con grilla LiDAR BEV de 5 canales utilizando `GroupNorm(16)` y cabeza de regresión `BEVPlanningHead` $\to [B, 10, 4]$.

---

## 3. Diagnóstico Técnico y Aprendizajes (Post-Mortem)

### Lo que funcionó extraordinariamente bien:
* **Comprensión Espacial Inmediata:** DINOv2 aportó primitivas visuales de geometría 3D y límites de carril deslumbrantes. El auto se mantiene en su carril a nivel sub-métrico.
* **Estabilidad Estricta a Batch Size $B=1$:** `GroupNorm(16)` erradicó por completo las oscilaciones en diente de sierra que producía `BatchNorm2d` en el Experimento 1.

### Los 2 Cuellos de Botella Detectados:
1. **Inercia en Giros de $90^\circ$ (Fenómeno de Repetición Temporal):**
   * *Causa:* Para mantener la VRAM ligera en $\sim 3.8\text{ GB}$, se proyectó el fotograma actual sobre la secuencia temporal $S=5$ (`repeat(1, S=5, 1, 1, 1)`).
   * *Efecto:* `TemporalMamba` recibió 5 fotos estáticas idénticas, calculando velocidad $= 0 \text{ m/s}$ y sin flujo óptico entre frames. Al no ver movimiento, la red memorizó una trayectoria recta promedio por defecto para giros pronunciados.
2. **Sesgo de Escala en Yaw ($52.8^\circ$):**
   * *Causa:* La función `HuberLoss` calculó la pérdida sumando posiciones en metros ($X,Y \in [0, 50]$) y ángulos en radianes ($Yaw \in [-\pi, \pi]$).
   * *Efecto:* La red priorizó aplastantemente la precisión de posición en metros ($0.49\text{m}$) a costa de descuidar la rotación del vehículo.

---

## 4. Hoja de Ruta — Experimento No. 3 (Para Discusión)

Para solucionar estos dos cuellos de botella manteniendo el entrenamiento rápido:

1. **Movimiento Temporal Real en BEV:**
   Procesar los mapas de características BEV de los 5 fotogramas pasados reales (o pasar el historial temporal del LiDAR) para que `TemporalMamba` calcule velocidad real y detecte giros en intersecciones.
2. **Cabeza de Orientación Trigonométrica:**
   Predecir el ángulo de guiñada mediante pares acotados `(sin(θ), cos(θ))` para evitar discontinuidades de envoltura en $\pm 180^\circ$.
3. **Pérdida Ponderada Multitarea:**
   Aplicar una pérdida balanceada `L_total = L_pos + 0.1 * L_yaw` para que el error en metros no tape el error de rotación.
