# Helioskrill — Experimento No. 4: Guía de Reproducibilidad Técnica

## 1. Visión General del Experimento
El **Experimento No. 4** introduce el condicionamiento por comando navegacional, la arquitectura Multi-Head con $Yaw$ trigonométrico, aprendizaje con tasas de variación diferenciadas (Differential LR) y balance de clases.

---

## 2. Parámetros y Configuración de Entrenamiento

* **Script de Entrenamiento:** `scripts/train_exp4.py`
* **Directorio de Checkpoints:** `./checkpoints/experimento_4/`
* **Épocas:** 15
* **Batch Size por GPU:** 1 (Acumulación de gradiente = 8)
* **Stride Temporal:** 5 (~1,438 iteraciones por época)
* **Learning Rate Backbone (DINOv2 + LoRA):** `5e-5`
* **Learning Rate Head (Mamba & Multi-Head):** `3e-4`
* **DataLoader num_workers:** `0` (WSL Fast Direct Loading)

---

## 3. Comando de Ejecución de Entrenamiento

```bash
python3 scripts/train_exp4.py \
    --data_dir ./data/ \
    --epochs 15 \
    --batch_size 1 \
    --seq_len 5 \
    --stride 5 \
    --accumulation_steps 8 \
    --lr_backbone 5e-5 \
    --lr_head 3e-4 \
    --num_workers 0
```

---

## 4. Comandos de Visualización y Diagnóstico

```bash
# Generar curvas de entrenamiento (Loss, ADE, FDE, Yaw, Velocidad)
python3 visualizations/visualize.py --mode metrics

# Visualizar mapa PCA de características DINOv2
python3 visualizations/visualize.py --mode pca

# Inferencia Autónoma en Tiempo Real en CARLA Simulator (Con Escudo LiDAR + Reversa)
python3 scripts/run_carla_closed_loop.py \
    --checkpoint ./checkpoints/experimento_4/best_model.pth \
    --command 1 \
    --lookahead 6.0 \
    --max_speed 8.33
```
