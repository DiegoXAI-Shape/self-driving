# Helioskrill — Experimento No. 3: Quintic Polynomial Planning Head + Kinematic Smoothness Loss (L_smooth)

## 1. Resumen Ejecutivo
El **Experimento 3** introduce una reforma matemática fundamental a la arquitectura de planificación de Helioskrill: se reemplaza la regresión lineal de puntos libres no acoplados por una **Cabeza de Parametrización Polinomial de 5to Grado (`PolynomialBEVPlanningHead`)** combinada con una **Función de Pérdida de Suavizado Cinemático ($\mathcal{L}_{\text{smooth}}$)**.

### Métricas Clave Obtenidas (Validación Cruzada en Rutas Desconocidas)
* **Mean ADE (Error Medio de Trayectoria):** **0.49 metros** (Sub-métrico: 49 cm en 5.0 segundos).
* **Mean FDE (Error Final a 5.0s):** **0.86 metros** (Consistencia continua al final del horizonte).
* **Mean Yaw Error (Orientación):** **34.4 grados** (Reducción drástica desde los $52.8^\circ$ del Experimento 2).
* **Mean Velocity Error:** **0.21 m/s**.
* **Mean Acceleration Error:** **1.35 m/s²**.
* **Forma de la Trayectoria:** Curvatura $C^2$ continua diferenciable sin oscilaciones en diente de sierra.
* **Parámetros Entrenables:** **1,644,168** ($6.94\%$ del total de $23.7\text{M}$).

---

## 2. Arquitectura Modificada (`PolynomialBEVPlanningHead`)

1. **Parametrización Polinomial de 5to Grado:**
   En lugar de predecir 40 valores independientes ($10 \times 4$), la red predice **12 coeficientes polinomiales**:
   $$X(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$
   $$Y(t) = b_0 + b_1 t + b_2 t^2 + b_3 t^3 + b_4 t^4 + b_5 t^5$$
   Los waypoints se evalúan en tiempo real mediante multiplicación matricial fija de Vandermonde $T \cdot C$.

2. **Orientación Tangencial Analítica ($Yaw$ Derivado):**
   El ángulo de orientación se deriva directamente de las velocidades analíticas del polinomio:
   $$\hat{\theta}(t) = \text{atan2}(\dot{Y}(t), \dot{X}(t))$$
   Esto garantiza matemáticamente que el morro del vehículo esté 100% alineado con la dirección del movimiento.

3. **Función de Pérdida Cinemática ($\mathcal{L}_{\text{smooth}}$):**
   Se penalizan las derivadas de 2do orden (Aceleración $a_2^2 + b_2^2$), 3er orden (Jerk $a_3^2 + b_3^2$) y altas frecuencias de curvatura ($a_4, a_5, b_4, b_5$), suprimiendo cualquier wiggle o sacudida en las predicciones.

---

## 3. Diagnóstico Técnico y Aprendizajes (Post-Mortem)

### Lo que funcionó extraordinariamente bien:
* **Reducción del Error de Yaw ($52.8^\circ \to 34.4^\circ$):** El acoplamiento tangencial obligó a la dirección a seguir la curva de velocidad, eliminando cambios bruscos de orientación.
* **Continuidad Cinemática:** Las trayectorias generadas en la grilla BEV son totalmente suaves y ejecutables por controladores vehiculares (Pure Pursuit / Stanley / MPC).

### Limitaciones Estructurales Identificadas en Bucle Cerrado (CARLA Live):
1. **Dilema Cuerda vs Tangente Real:** En giros de $90^\circ$, la recta de desplazamiento desde el origen $(0,0)$ a la esquina forma una diagonal de $45^\circ$ (la cuerda). El polinomio calcula la velocidad del vector de desplazamiento ($45^\circ$), en lugar de rotar completamente la trompa a $90^\circ$.
2. **Necesidad de Condicionamiento por Comando (Command Conditioning):** Sin una señal de comando de alto nivel (GPS Target / `TURN_LEFT`, `TURN_RIGHT`), el modelo enfrenta ambigüedad multimodal en intersecciones de 4 vías.
3. **Desequilibrio de Escala en Loss ($Yaw$ en Grados vs Metros):** La `HuberLoss` suma directamente metros ($0.49\text{m}$) y grados sexagesimales ($34^\circ$), haciendo que el $95\%$ de la pérdida de validación ($\sim 43.0$) sea dominada por los grados sexagesimales.

---

## 4. Próximos Pasos (Experimento No. 4)

1. **Integrar Encoders de Comando Navegacional GPS:** Pasar la señal de intención (`LANE_FOLLOW`, `TURN_LEFT`, `TURN_RIGHT`) a la red.
2. **Cabeza Trigonométrica para Yaw:** Predecir la rotación con pares trigonométricos `(sin(θ), cos(θ))` acotados $[-1, 1]$ o radianes.
3. **Loss Multitarea Ponderado:** `L_total = L_pos + 0.05 * L_yaw` para equilibrar la escala entre metros y ángulos.
