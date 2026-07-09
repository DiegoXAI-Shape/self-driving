"""
visualize_carla_data.py
=======================
Script interactivo para inspeccionar y visualizar los datos recolectados de CARLA.

CARACTERÍSTICAS
---------------
  - Muestra una cuadrícula con las 8 cámaras sincronizadas.
  - Muestra el grid BEV del LiDAR (canal de altura o densidad).
  - Superpone en el BEV la posición relativa de coches y peatones cercanos (Prediction).
  - Superpone los waypoints futuros del coche (Planning) convertidos a coordenadas locales.
  - Muestra telemetría de control ( throttle, brake, steer) y localización (velocidad, IMU).
  - Control de reproducción:
      * Flecha Derecha: Siguiente frame
      * Flecha Izquierda: Frame anterior
      * Barra Espaciadora: Play / Pause
      * Tecla 'q': Salir

REQUISITOS
----------
  pip install numpy matplotlib pandas opencv-python

USO
---
  python src/scripts/visualize_carla_data.py --episode 0
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Mapeo de índices de cámara a sus nombres y posiciones relativas
CAM_NAMES = [
    "Front Main", "Front Wide", "Front Narrow",
    "Left B-Pillar", "Right B-Pillar",
    "Left Repeater", "Right Repeater", "Rear"
]

def global_to_local(ego_x, ego_y, ego_yaw_deg, points_x, points_y):
    """
    Convierte puntos de coordenadas globales (mapa) a locales (ego-vehículo).
    En CARLA:
      - X local apunta hacia adelante.
      - Y local apunta hacia la derecha.
    """
    # Convertir a radianes y alinear con orientación de CARLA (yaw)
    yaw = np.radians(ego_yaw_deg)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    # Traslación
    dx = points_x - ego_x
    dy = points_y - ego_y

    # Rotación inversa
    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y

    return local_x, local_y

class CARLADataVisualizer:
    def __init__(self, data_root, episode_id):
        self.data_root = data_root
        self.episode_id = episode_id
        self.ep_str = f"episode_{episode_id:04d}"
        print(self.ep_str)

        # Validar existencia de carpetas
        self.ep_path_perception = os.path.join(data_root, "Perception", "CARLA", self.ep_str)
        self.ep_path_location = os.path.join(data_root, "Location", self.ep_str)
        self.ep_path_planning = os.path.join(data_root, "Planning", self.ep_str)
        self.ep_path_control = os.path.join(data_root, "Control", self.ep_str)
        self.ep_path_prediction = os.path.join(data_root, "Prediction", self.ep_str)

        if not os.path.exists(self.ep_path_perception):
            raise FileNotFoundError(f"No se encontró el episodio {self.ep_str} en {data_root}.")

        print("[Cargando] Leyendo metadatos tabulares de los CSV...")
        
        # Cargar CSVs
        self.df_location = pd.read_csv(os.path.join(self.ep_path_location, "location.csv"))
        self.df_planning = pd.read_csv(os.path.join(self.ep_path_planning, "waypoints.csv"))
        self.df_control = pd.read_csv(os.path.join(self.ep_path_control, "control.csv"))
        
        # Prediction puede estar vacío si no hubo actores cerca
        pred_file = os.path.join(self.ep_path_prediction, "actors.csv")
        if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
            self.df_prediction = pd.read_csv(pred_file)
        else:
            self.df_prediction = pd.DataFrame()

        # Determinar número total de frames
        self.num_frames = len(self.df_location)
        self.current_frame = 0
        self.playing = False

        print(f"[OK] Cargados {self.num_frames} frames para visualizar.")

        # Configuración de Matplotlib
        self.setup_layout()

    def setup_layout(self):
        # Crear figura grande de 3 columnas x 4 filas
        # Fila 1-3: Cámaras (3 columnas)
        # Columna Derecha (abarca varias filas): Grid BEV
        plt.rcParams['toolbar'] = 'None' # Quitar barra de herramientas fea de matplotlib
        self.fig = plt.figure(figsize=(18, 10), facecolor='#0f172a')
        self.fig.suptitle(f"Helioskrill — Inspector del Episodio {self.episode_id:04d}", 
                          color='white', fontsize=18, fontweight='bold', y=0.97)

        # Definir la rejilla de subplots (GridSpec)
        # 3 filas x 4 columnas
        # Las cámaras van en las columnas 0, 1, 2. El BEV ocupa la columna 3 (toda la altura)
        gs = plt.GridSpec(3, 4, figure=self.fig, left=0.03, right=0.97, bottom=0.05, top=0.92, wspace=0.15, hspace=0.25)

        # Ejes para las 8 cámaras (3x3 grid, quitando el del centro o rear dependiente de acomodo)
        # Diseño de Cámaras:
        # [Cam 0: Front Main]   [Cam 1: Front Wide]     [Cam 2: Front Narrow]
        # [Cam 3: Left B-Pill]  [Cam 7: Rear]           [Cam 4: Right B-Pill]
        # [Cam 5: Left Repeat]  [Info/Telemetría]       [Cam 6: Right Repeat]
        self.cam_axes = []
        cam_positions = [
            (0, 1), # Front Main
            (0, 0), # Front Wide
            (0, 2), # Front Narrow
            (1, 0), # Left B-Pillar
            (1, 2), # Right B-Pillar
            (2, 0), # Left Repeater
            (2, 2), # Right Repeater
            (1, 1), # Rear
        ]

        for pos in cam_positions:
            ax = self.fig.add_subplot(gs[pos[0], pos[1]])
            ax.axis('off')
            self.cam_axes.append(ax)

        # Subplot de Telemetría e Info (Fila 2, Columna 1)
        self.info_ax = self.fig.add_subplot(gs[2, 1])
        self.info_ax.axis('off')

        # Subplot para el Bird's Eye View (BEV)
        # Ocupa toda la columna de la derecha (Filas 0 a 2, Columna 3)
        self.bev_ax = self.fig.add_subplot(gs[0:3, 3])
        self.bev_ax.set_facecolor('#090d16')
        self.bev_ax.set_title("LiDAR BEV & Prediction Map", color='white', fontsize=14, fontweight='bold')
        
        # Conectar eventos de teclado
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def draw_frame(self):
        frame_idx = self.current_frame
        
        # Obtener datos de la fila actual en los DataFrames
        row_loc = self.df_location.iloc[frame_idx]
        row_ctrl = self.df_control.iloc[frame_idx]
        row_plan = self.df_planning.iloc[frame_idx]

        # ── 1. Dibujar las 8 imágenes de cámaras ──────────────────────────────────
        for i in range(8):
            ax = self.cam_axes[i]
            ax.clear()
            ax.axis('off')
            
            img_path = os.path.join(self.ep_path_perception, "cameras", f"cam_{i}", f"frame_{frame_idx:06d}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                # Mostrar cuadro gris de aviso
                ax.text(0.5, 0.5, "SIN SEÑAL", color='#ef4444', ha='center', va='center', fontsize=12, fontweight='bold')
                ax.set_facecolor('#1e293b')
            
            ax.set_title(f"{CAM_NAMES[i]} (cam_{i})", color='#94a3b8', fontsize=10, pad=4)

        # ── 2. Dibujar Telemetría (Información del coche) ──────────────────────────
        self.info_ax.clear()
        self.info_ax.axis('off')
        
        speed_kmh = row_loc['speed_mps'] * 3.6
        throttle = row_ctrl['throttle']
        brake = row_ctrl['brake']
        steer = row_ctrl['steer']

        info_text = (
            f"Frame: {frame_idx:06d} / {self.num_frames - 1}\n\n"
            f"Velocidad: {speed_kmh:.1f} km/h ({row_loc['speed_mps']:.1f} m/s)\n"
            f"Ubicación: X: {row_loc['ego_x']:.2f}, Y: {row_loc['ego_y']:.2f}\n"
            f"Orientación (Yaw): {row_loc['ego_yaw']:.1f}°\n\n"
            f"Acciones de Control:\n"
            f" ├─ Throttle: {throttle * 100:.1f}%\n"
            f" ├─ Freno:    {brake * 100:.1f}%\n"
            f" └─ Volante:  {steer:+.4f}\n\n"
            f"IMU G-Forces:\n"
            f" ├─ Accel X:  {row_loc['imu_accel_x']:+.2f} m/s²\n"
            f" └─ Accel Y:  {row_loc['imu_accel_y']:+.2f} m/s²"
        )
        self.info_ax.text(0.05, 0.95, info_text, color='white', family='monospace', 
                          fontsize=11, ha='left', va='top', transform=self.info_ax.transAxes)

        # ── 3. Dibujar el LiDAR BEV ───────────────────────────────────────────────
        self.bev_ax.clear()
        self.bev_ax.set_facecolor('#090d16')
        
        # Cargar archivo de cuadrícula numpy (.npy)
        npy_path = os.path.join(self.ep_path_perception, "lidar", f"frame_{frame_idx:06d}.npy")
        if os.path.exists(npy_path):
            # El grid tiene canales (5, 400, 400).
            # Canal 3: Densidad de puntos LiDAR
            bev_grid = np.load(npy_path)
            density_channel = bev_grid[3]  # [400, 400]
            
            # Dibujar la densidad de fondo (escala de grises azulados/verdes)
            self.bev_ax.imshow(density_channel, cmap='inferno', extent=[-50, 50, -50, 50], origin='upper')
        else:
            self.bev_ax.text(0, 0, "No se encontró el archivo LiDAR (.npy)", color='gray', ha='center', va='center')

        # Dibujar límites de alcance visuales en el BEV
        circles = [10, 20, 30, 40, 50]
        for r in circles:
            circle = plt.Circle((0, 0), r, color='#334155', fill=False, linestyle='--', alpha=0.5)
            self.bev_ax.add_patch(circle)

        # Coche Ego (en el centro del grid 0,0 apuntando al Norte en coordenadas locales)
        # CARLA Local: X=adelante, Y=derecha. Ploteamos con X=vertical (adelante), Y=horizontal (izquierda/derecha).
        # En matplotlib invertimos: Y_local va en el eje horizontal y X_local en el eje vertical
        ego_marker = plt.Rectangle((-1.0, -2.4), 2.0, 4.8, color='#10b981', fill=True, label="Ego (Tesla)")
        self.bev_ax.add_patch(ego_marker)
        # Flecha de dirección del coche ego
        self.bev_ax.arrow(0, 0, 0, 6, head_width=2.5, head_length=2.5, fc='#059669', ec='#059669')

        # ── 4. Dibujar los Waypoints de Planning (Futuro del coche) ────────────────
        # Leer waypoints del CSV de este frame
        wp_x = []
        wp_y = []
        for w_i in range(10):
            wp_x.append(row_plan[f"wp_{w_i}_x"])
            wp_y.append(row_plan[f"wp_{w_i}_y"])
            
        # Convertir a coordenadas locales relativas al Ego
        local_wp_x, local_wp_y = global_to_local(
            row_loc['ego_x'], row_loc['ego_y'], row_loc['ego_yaw'],
            np.array(wp_x), np.array(wp_y)
        )
        
        # En el gráfico, local_y (derecha/izquierda) se asocia al eje X
        # local_x (adelante) se asocia al eje Y del gráfico
        # Invertimos el signo de local_y para que la derecha quede a la derecha en la pantalla
        self.bev_ax.plot(-local_wp_y, local_wp_x, color='#a78bfa', marker='o', markersize=6, 
                         linewidth=2, linestyle='-', label="Waypoints Planificados")

        # ── 5. Dibujar Actores del Entorno (Predicción / Tráfico) ──────────────────
        if not self.df_prediction.empty:
            # Filtrar actores correspondientes a este frame
            frame_actors = self.df_prediction[self.df_prediction['frame'] == frame_idx]
            
            vehicles_plotted = False
            walkers_plotted = False
            
            for _, actor in frame_actors.iterrows():
                # En actores ya tenemos rel_x (distancia adelante/atrás) y rel_y (distancia izq/der)
                # rel_x es local X, rel_y es local Y.
                # Eje horizontal del plot = -rel_y, Eje vertical del plot = rel_x
                act_x_plot = -actor['rel_y']
                act_y_plot = actor['rel_x']
                
                # Clasificar tipo de objeto
                if actor['actor_type'] == "vehicle":
                    self.bev_ax.plot(act_x_plot, act_y_plot, color='#3b82f6', marker='s', markersize=8, 
                                     linestyle='None', markeredgecolor='white', markeredgewidth=1)
                    vehicles_plotted = True
                else:
                    self.bev_ax.plot(act_x_plot, act_y_plot, color='#ef4444', marker='o', markersize=6, 
                                     linestyle='None', markeredgecolor='white', markeredgewidth=1)
                    walkers_plotted = True

            # Añadir leyendas fantasmas para el gráfico
            if vehicles_plotted:
                self.bev_ax.plot([], [], color='#3b82f6', marker='s', label="Vehículo Detectado", linestyle='None')
            if walkers_plotted:
                self.bev_ax.plot([], [], color='#ef4444', marker='o', label="Peatón Detectado", linestyle='None')

        # Configurar límites y rejilla
        self.bev_ax.set_xlim([-50, 50])
        self.bev_ax.set_ylim([-50, 50])
        self.bev_ax.grid(color='#1e293b', linestyle=':', alpha=0.6)
        self.bev_ax.tick_params(colors='#64748b', labelsize=9)
        self.bev_ax.legend(loc='lower right', facecolor='#0f172a', edgecolor='#1e293b', labelcolor='white', fontsize=8)

        # Forzar redibujado de la interfaz
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """Manejador de atajos de teclado."""
        if event.key == 'right':
            self.playing = False
            self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
            self.draw_frame()
        elif event.key == 'left':
            self.playing = False
            self.current_frame = max(self.current_frame - 1, 0)
            self.draw_frame()
        elif event.key == ' ':
            self.playing = not self.playing
            if self.playing:
                self.run_player()
        elif event.key == 'q':
            plt.close()

    def run_player(self):
        """Loop de reproducción automática."""
        while self.playing and self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.draw_frame()
            plt.pause(0.05) # Pausa de ~20 FPS de dibujo
        
        self.playing = False

    def show(self):
        self.draw_frame()
        print("\n" + "═"*60)
        print("  INSTRUCCIONES DE USO:")
        print("  ├─ Flecha Derecha  : Siguiente fotograma")
        print("  ├─ Flecha Izquierda: Fotograma anterior")
        print("  ├─ Barra Espaciadora: Play / Pausa")
        print("  └─ Tecla 'q'       : Salir de la herramienta")
        print("═"*60 + "\n")
        plt.show()


if __name__ == "__main__":

    CONFIG = {
        "output_root": os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data")
        )
    }
    
    parser = argparse.ArgumentParser(description="Visualizador de Datos de Simulación de Helioskrill")
    parser.add_argument("--data_dir", default=CONFIG["output_root"], help="Ruta al directorio src/data/")
    parser.add_argument("--episode", type=int, default=0, help="ID del episodio a visualizar (ej: 0)")
    args = parser.parse_args()

    # Resolver ruta absoluta por seguridad
    abs_data_dir = os.path.abspath(args.data_dir)

    try:
        visualizer = CARLADataVisualizer(abs_data_dir, args.episode)
        visualizer.show()
    except Exception as e:
        print(f"\n[ERROR] No se pudo inicializar la visualización: {e}")
        print("Asegúrate de haber recolectado datos primero ejecutando:")
        print("  python src/models/utils/carla_data_collector.py")
