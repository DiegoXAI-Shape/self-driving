# Helioskrill — Experimento No. 3: Parametrización Polinomial de 5to Grado & Pérdida de Suavizado Cinemático

## 1. Visión General del Experimento
El **Experimento No. 3** soluciona de raíz la coherencia geométrica y la inestabilidad en la rotación de la trayectoria mediante:
1. **Parametrización Polinomial de 5to Grado (`PolynomialBEVPlanningHead`):** La red predice 12 coeficientes continuos ($a_0..a_5$ para $X(t)$ y $b_0..b_5$ para $Y(t)$), garantizando una trayectoria $C^\infty$ matemáticamente suave.
2. **Derivación Analítica del Ángulo de Yaw:** El ángulo de la trayectoria se calcula como la tangente exacta de la velocidad $\text{atan2}(\dot{Y}(t), \dot{X}(t))$, bloqueando el ángulo de la cabina a la curvatura de la trayectoria por diseño.
3. **Pérdida de Suavizado Cinemático (`PolynomialSmoothnessLoss`):** Penaliza los términos de aceleración, tirón (*jerk*) y curvaturas de alta frecuencia.

---

## 2. Parámetros y Configuración de Entrenamiento

* **Script de Entrenamiento:** `scripts/train_exp3.py`
* **Directorio de Checkpoints:** `./checkpoints/experimento_3/`
* **Épocas:** 20
* **Batch Size por GPU:** 1 (Acumulación de gradiente = 8)
* **Stride Temporal:** 10 (~720 iteraciones por época)
* **Optimización de Carga:** `prefetch_factor=2`, `persistent_workers=True`, `pin_memory=True`

---

## 3. Comando de Ejecución

```bash
python3 scripts/train_exp3.py \
    --data_dir ./data/ \
    --epochs 20 \
    --batch_size 1 \
    --seq_len 5 \
    --stride 10 \
    --resize_factor 0.5 \
    --accumulation_steps 8 \
    --lora_r 8 \
    --lr 1e-4 \
    --num_workers 4
```
