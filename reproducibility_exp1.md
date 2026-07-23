# Guía de Reproducibilidad — Experimento No. 1 (Helioskrill)

Este documento detalla los pasos exactos para reproducir, entrenar y validar el **Experimento No. 1** del modelo Helioskrill utilizando la arquitectura espacio-temporal basada en **Vision Mamba (ViM)** y **Temporal Mamba**.

---

## 1. Objetivos del Experimento
* Validar la integración del bloque `TemporalMamba` para la memoria espacio-temporal y seguimiento de objetos a lo largo del tiempo.
* Entrenar de extremo a extremo el modelo de planificación pura usando **Huber Loss (Smooth L1)** para estimar los 10 waypoints futuros en coordenadas locales del vehículo (`rel_x`, `rel_y`, `rel_z`, `rel_yaw`).
* Prevenir la **fuga de datos temporal (Data Leakage)** mediante la división episódica de datos (Episodic Split).
* Monitorear métricas de trayectoria (ADE, FDE), cinemática (Velocidad, Aceleración, Yaw) y errores temporales por paso (Horizon ADE).

---

## 2. Requisitos de Hardware y Librerías
* **GPU:** NVIDIA con arquitectura Ampere/Ada/Blackwell (ej. RTX 5070 Ti, mínimo 12GB VRAM).
* **Librerías principales:**
  ```bash
  conda activate helioskrill
  
  pip install torch torchvision numpy pandas opencv-python matplotlib tqdm tensorboard
  pip install causal-conv1d>=1.4.0
  pip install mamba-ssm
  ```

---

## 3. Pre-procesamiento de Datos (Opcional para acelerar I/O)

Para evitar cuellos de botella en la lectura y redimensionamiento en tiempo real durante el entrenamiento:
```bash
python3 scripts/preprocess_dataset.py
```
* **Efecto:** Redimensiona en paralelo con 32 núcleos las 67,176 imágenes a $400 \times 300$ ($0.5\times$) y las guarda en `data/Perception_resized`, conservando intactos los datos originales de `data/Perception`.

---

## 4. Configuración del Pipeline de Tensores (8 Pasos)

El flujo del experimento estructura los tensores de la siguiente manera:
1. **Entrada:** `camera_imgs` de forma `[B, S=5, N=8, C=3, H=300, W=400]` ($B=1$, $S=5$ frames pasados, $N=8$ cámaras RGB).
2. **Lote de Imágenes:** Aplanado a `[B * S * N, 3, 304, 400]` (con padding dinámico en altura $300 \rightarrow 304$ divisible por 16).
3. **Mamba Espacial 2D:** Extracción de características visuales mediante `VisionMambaEncoder` $\rightarrow$ `[B * S * N, 64, 19, 25]`.
4. **Desaplanado a Vistas:** `[B * S, N, 64, 19, 25]`.
5. **Proyección BEV:** Proyección IPM con matrices $K$ y extrínsecas a plano Top-Down $\rightarrow$ `[B * S, 64, 400, 400]`.
6. **Desaplanado Temporal:** `[B, S, 64, 400, 400]`.
7. **Mamba Temporal 1D:** Recurrencia secuencial a lo largo de la dimensión $S$ celda por celda $\rightarrow$ `[B, S, 64, 400, 400]`.
8. **Planning Head:** Regresión del mapa fusionado del último frame $\rightarrow$ `[B, 10, 4]`.

---

## 5. División de Datos (Episodic Split)

Para garantizar que el conjunto de validación evalúe la generalización en rutas inéditas:
* **Entrenamiento (85%):** Episodios `episode_0000` a `episode_0011`.
* **Validación (15%):** Episodios `episode_0012` a `episode_0013`.

---

## 6. Comandos para Ejecutar el Entrenamiento

### A. Lanzar Servidor TensorBoard (Terminal 1)
```bash
tensorboard --logdir checkpoints/experimento_1/tensorboard --port 6006
```

### B. Lanzar Entrenamiento Optimizado con Reanudación (Terminal 2)
```bash
python3 scripts/train_vim.py \
    --data_dir ./data/ \
    --epochs 20 \
    --batch_size 1 \
    --seq_len 5 \
    --stride 5 \
    --resize_factor 0.5 \
    --accumulation_steps 8 \
    --num_workers 2 \
    --lr 1e-4 \
    --resume
```

---

## 7. Métricas Registradas y Checkpoints

### Métricas en CSV (`checkpoints/experimento_1/metrics.csv`) y TensorBoard:
* `Loss/train` y `Loss/val`: Huber Loss ($\delta=1.0$).
* `Metrics/ADE` y `Metrics/FDE`: Average / Final Displacement Error en metros.
* `Temporal/vel_error_mps`: Error de velocidad en $\text{m/s}$.
* `Temporal/accel_error_mps2`: Error de aceleración en $\text{m/s}^2$.
* `Temporal/yaw_error_deg`: Error angular en grados ($^\circ$).
* `Horizon_ADE/step_1_m` a `step_10_m`: Error de posición por paso temporal a futuro.

### Estructura del Checkpoint (`last_model.pth` y `best_model.pth`):
Cada checkpoint guarda el diccionario completo de control:
```python
{
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'scaler_state': scaler.state_dict(),
    'best_val_loss': best_val_loss,
    'early_stopping_counter': early_stopping.counter
}
```
