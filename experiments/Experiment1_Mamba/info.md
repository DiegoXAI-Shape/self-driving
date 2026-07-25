# Guía Conceptual: Mamba, Vision Mamba (ViM) y Percepción BEV (Bird's Eye View)

Este documento resume los conceptos clave explicados durante el Pair Programming para guiar tu implementación de Vision Mamba (ViM) y su integración con sistemas de percepción de vista de pájaro (como el de Tesla).

---

## 1. Glosario de Dimensiones y Variables

Tener claras las dimensiones de los tencesor es la clave para entender y programar cualquier modelo de Deep Learning:
*   **$B$ (Batch Size):** Tamaño del lote de datos procesados en paralelo.
*   **$M$ (Sequence Length / Tokens):** Número de tokens en la secuencia. En visión, es la cantidad de parches (patches) en los que se divide la imagen (por ejemplo, para una imagen con $16 \times 16$ parches, $M = 256$).
*   **$D$ (Model Dimension):** El número de canales o dimensión de embedding del modelo.
*   **$E$ (Expanded Dimension):** Dimensión expandida dentro del bloque Mamba. Típicamente $E = 2 \times D$.
*   **$N$ (State Dimension):** El tamaño del espacio de estados interno del SSM (State Space Model). Suele ser un valor pequeño constante (como $16$).

---

## 2. Análisis del Algoritmo 1: Bloque Vision Mamba (ViM)

A diferencia de Mamba clásico (diseñado para secuencias unidireccionales de texto), **Vision Mamba (ViM)** procesa la secuencia en ambas direcciones (bidireccional) para capturar relaciones espaciales bidimensionales.

```
Imagen 2D -> Patch Embedding -> Tokens (B, M, D)
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Normalización   │
                              └────────┬────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼ (Rama de procesamiento)   ▼ (Rama de gating)
                     Proyección x                Proyección z
                    (B, M, E)                   (B, M, E)
                         │                           │
            ┌────────────┴────────────┐              │
            ▼ (Forward)               ▼ (Backward)   │
         Conv1d + SiLU             [Voltear M]       │
            │                     Conv1d + SiLU      │
         Proyectar                 [Desvoltear]      │
        B_f, C_f, Δ_f                 │              │
            │                     Proyectar          │
        Discretizar             B_b, C_b, Δ_b        │
        A_bar, B_bar                  │              │
            │                    Discretizar         │
        SSM Scan                 A_bar, B_bar        │
        (Recurrencia)                 │              │
            │                     SSM Scan           │
            │                   (Recurrencia)        │
            │                         │              │
            └────────────┬────────────┘              │
                         ▼                           ▼
                    y_forward                   SiLU(z)
                    y_backward                       │
                         │                           │
                         ├───────────────────────────┘
                         ▼ (Multiplicación Gating)
                     Multiplicar por SiLU(z)
                         │
                         ▼
                     Suma y Proyección de salida (E -> D)
                         │
                         ▼ (Residual)
                    Salida + Entrada
```

### Paso a Paso de las Operaciones Matemáticas:

1.  **Normalización e Input Projection (Líneas 1-4):**
    *   La entrada $T_{l-1}$ se normaliza y se proyecta a dos ramas: $x$ (procesamiento) y $z$ (puerta/gating) usando capas lineales.
    *   Tanto $x$ como $z$ tienen dimensiones $(B, M, E)$.
2.  **Procesamiento Bidireccional (Líneas 5-23):**
    *   **Forward Path:** Se pasa $x$ por una convolución 1D y una activación SiLU. Luego se proyectan dinámicamente las matrices $B_{forward}$, $C_{forward}$ y el tamaño de paso $\Delta_{forward}$.
    *   **Backward Path:** Para simular la lectura de derecha a izquierda, se **invierte** el tensor $x$ en la dimensión de secuencia $M$ (`torch.flip(x, dims=[1])`). Se realiza el mismo procesamiento (convolución, SiLU, proyecciones) y, tras completar el escaneo del SSM, **se vuelve a invertir el tensor resultante** para alinearlo con el flujo principal.
3.  **Discretización (Líneas 12-14):**
    *   La matriz continua $A$ (parámetro learnable de tamaño $(E, N)$) se discretiza usando el tamaño de paso $\Delta$:
        $$\overline{A} = \exp(\Delta \otimes A)$$
    *   La matriz $B$ se discretiza de forma lineal:
        $$\overline{B} = \Delta \otimes B$$
    *   *Nota técnica:* Para realizar estas multiplicaciones en PyTorch, se expanden las dimensiones utilizando `unsqueeze` para aprovechar el *broadcasting* automático.
4.  **El Bucle de Recurrencia / Scan (Líneas 15-22):**
    *   Se inicializa la memoria interna del sistema (estado oculto $h$) en ceros con dimensiones $(B, E, N)$.
    *   Para cada paso temporal/token $i \in \{0, \dots, M-1\}$:
        *   Se actualiza el estado oculto:
            $$h_i = \overline{A}_i \odot h_{i-1} + \overline{B}_i \odot x'_i$$
        *   Se genera la salida intermedia $y_i$ proyectando la memoria $h_i$ con la matriz $C_i$:
            $$y_i = \sum_{n} h_i[:, e, n] \cdot C_i[:, n]$$
            *(En PyTorch, esto equivale a hacer una reducción con producto de matrices o un `torch.einsum('ben,bn->be', h, C)`)*.
5.  **Unión, Gating y Residual (Líneas 24-29):**
    *   Se aplican las compuertas utilizando la rama $z$:
        $$y'_{dir} = y_{dir} \odot \text{SiLU}(z)$$
    *   Se suman ambos caminos ($y'_{forward} + y'_{backward}$), se proyectan de vuelta a la dimensión $D$ mediante una capa lineal, y se añade la conexión residual original.

---

## 3. ¿Cómo funciona la Vista de Pájaro (BEV) con 8 Cámaras?

El objetivo es fusionar la perspectiva 2D deformada de 8 cámaras en una sola cuadrícula plana tridimensional métrica sobre el suelo, centrada en el carro.

El algoritmo estándar implementado en tu módulo `BEV_perception.py` (usando **Proyección Hacia Atrás / Spatial Grid Querying**) funciona de la siguiente manera:

```
          [ Foto Cámara 1 ] [ Foto Cámara 2 ] ... [ Foto Cámara 8 ]
                  │                 │                     │
                  │                 │                     │
    1. Rejilla virtual 3D en el espacio del Carro (Ego Frame)
       Ej: 400x400 puntos virtuales en el suelo alrededor de (0,0,0)
                  │
                  ▼
    2. Matrices Extrínsecas (R, T):
       Transforman las coordenadas 3D del Carro -> Espacio 3D de cada Cámara
                  │
                  ▼
    3. Matrices Intrínsecas (K):
       Proyectan los puntos 3D de las cámaras -> Coordenadas de píxel 2D (u, v)
                  │
                  ▼
    4. Muestreo (grid_sample):
       Leen las características visuales (features de CNN/Mamba) en (u, v)
                  │
                  ▼
    5. Fusión y Reducción Z:
       Promedian los solapamientos de cámaras y reducen el eje de la altura
                  │
                  ▼
          [ Mapa BEV 2D Final ]
```

---

## 4. ¿Dónde entra Mamba en la Percepción BEV?

Vision Mamba (ViM) se puede integrar en tres componentes clave del sistema:

### Opción A: Como el Extractor de Características 2D (Backbone)
*   **Reemplazo:** Sustituye a las CNN tradicionales (como ResNet) que procesan las fotos crudas de cada cámara.
*   **Funcionamiento:** La foto pasa por `PatchEmbedding2d` y luego por `VisionMambaEncoder` para extraer características de alta calidad con contexto global pero a coste computacional lineal.

### Opción B: Como Procesador del Espacio BEV (Fusion & Spatial Reasoning)
*   **Reemplazo:** Reemplaza a las redes convolucionales o Swin Transformers que procesan el plano de suelo unificado de $400 \times 400$ píxeles.
*   **Funcionamiento:** Se aplana el mapa BEV en una gran secuencia de $160,000$ tokens y se pasa por un codificador Mamba.
*   **Ventaja:** Permite que las celdas del mapa que están muy distantes se comuniquen entre sí (por ejemplo, continuar una carretera bloqueada por oclusiones) con complejidad lineal $O(M)$ en lugar de la cuadrática de los Transformers convencionales.

### Opción C: Como Módulo de Fusión Spatio-Temporal
*   **Funcionamiento:** Mamba actúa como una RNN de alto rendimiento. En lugar de procesar solo el mapa BEV actual, el estado oculto del SSM ($h$) retiene la información física del mapa en los instantes de tiempo pasados, permitiendo al vehículo recordar coches tapados temporalmente y calcular velocidades de manera robusta.
