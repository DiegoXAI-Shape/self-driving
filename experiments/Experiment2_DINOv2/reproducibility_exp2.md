# Guía de Reproducibilidad — Experimento No. 2 (DINOv2 + LoRA + Temporal Mamba)

Este documento detalla los pasos exactos para reproducir, entrenar y validar el **Experimento No. 2** del modelo Helioskrill, utilizando **Meta DINOv2 Small (`dinov2_vits14`)** con adaptación **LoRA ($r=8$)**, estabilidad por **GroupNorm** y recurrencia espacio-temporal mediante **Temporal Mamba**.

---

## 1. Objetivos del Experimento
* Solucionar la falencia de representación visual 2D del Experimento 1 (entrenamiento desde cero) mediante el uso del backbone autosupervisado pre-entrenado **DINOv2 Small**.
* Entrenar de forma eficiente congelando el 93%+ del modelo base e inyectando adaptadores de bajo rango (**LoRA tradicional**) en las proyecciones `qkv` de atención visual.
* Eliminar la inestabilidad de `BatchNorm2d` mediante **`GroupNorm(num_groups=16)`** en el cuello de fusión BEV.
* Entrenar el modelo de planificación para estimar los 10 waypoints futuros en coordenadas locales del vehículo (`rel_x`, `rel_y`, `rel_z`, `rel_yaw`) guardando checkpoints en `checkpoints/experimento_2/`.

---

## 2. Requisitos de Hardware y Librerías
* **GPU:** NVIDIA con mínimo 8 GB de VRAM (ej. RTX 5070 Ti / RTX 3070).
* **Librerías principales:**
  ```bash
  conda activate helioskrill
  
  pip install torch torchvision numpy pandas opencv-python matplotlib tqdm tensorboard peft
  pip install causal-conv1d>=1.4.0
  pip install mamba-ssm
  ```

---

## 3. Resumen de Parámetros del Modelo

* **Backbone Visual:** `dinov2_vits14` (Patch Size: $14 \times 14$, Embed Dim: $384$).
* **Configuración LoRA:** $r=8$, $\alpha=16$, Dropout: $0.05$, Módulos objetivo: `qkv`.
* **Parámetros Totales:** $23,700,744$.
* **Parámetros Entrenables:** $1,644,168$ ($\sim 6.94\%$ del total).
* **Consumo de VRAM:** $\sim 3.5 \text{ GB}$ (para $B=1, S=5, N=8$).

---

## 4. Comandos para Ejecutar el Entrenamiento

### A. Lanzar Servidor TensorBoard (Terminal 1)
```bash
tensorboard --logdir checkpoints/experimento_2/tensorboard --port 6006
```

### B. Lanzar Entrenamiento Experimento 2 (Terminal 2)
```bash
python3 scripts/train_dinov2.py \
    --data_dir ./data/ \
    --epochs 20 \
    --batch_size 1 \
    --seq_len 5 \
    --stride 5 \
    --resize_factor 0.5 \
    --accumulation_steps 8 \
    --lora_r 8 \
    --lr 1e-4 \
    --resume
```

---

## 5. Evaluación Visual y Comparativa

Para evaluar el modelo entrenado del Experimento 2 utilizando el script de validación cruzada:

```bash
python3 scripts/evaluate_visualization.py \
    --checkpoint ./checkpoints/experimento_2/best_model.pth \
    --data_dir ./data/ \
    --num_samples 10 \
    --output_dir ./eval_results_exp2/
```
