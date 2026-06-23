import streamlit as st
import statsapi as mlb
import pandas as pd
import plotly.express as px
from pybaseball import statcast_pitcher
from datetime import datetime, timedelta
import mediapipe as mp
import numpy as np
import cv2
import tempfile
import requests

# IMPORTACIÓN SEGURA DE MEDIAPIPE
from mediapipe.python.solutions import pose as mp_solutions_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Andrés Muñoz Analysis", layout="wide", page_icon="⚾")

# 2. FUNCIONES DE APOYO
@st.cache_data
def get_statcast_data(start, end, player_id):
    try:
        data = statcast_pitcher(start, end, player_id)
        if not data.empty:
            data['V_Break_in'] = data['pfx_z'] * 12
            data['H_Break_in'] = data['pfx_x'] * 12
            # Aseguramos que el game_pk sea un entero
            data['game_pk'] = data['game_pk'].astype(int)
        return data
    except:
        return pd.DataFrame()

def buscar_video_oficial(game_pk):
    """Busca el video directo (.mp4) usando la ruta oficial de la API de MLB"""
    try:
        # Fíjate bien en las barras '/' y que diga statsapi
        url = f"https://mlb.com{game_pk}/content"
        
        # Esto te servirá para verificar en Chrome que la ruta es correcta
        st.write(f"Consultando API en: {url}") 
        
        response = requests.get(url).json()
        
        # Buscamos en la lista de highlights
        highlights = response.get('highlights', {}).get('highlights', {}).get('items', [])
        
        for item in highlights:
            title = item.get('title', '')
            # Filtramos para asegurar que es un pitcheo o highlight de Muñoz
            if "Andres Munoz" in title or "Munoz" in title:
                playbacks = item.get('playbacks', [])
                for pb in playbacks:
                    # Buscamos específicamente el archivo de video reproducible
                    video_url = pb.get('url', '')
                    if video_url.endswith('.mp4'):
                        return video_url
        return None
    except Exception as e:
        st.error(f"Error de conexión con MLB: {e}")
        return None


# 3. CARGA DE DATOS
MUÑOZ_ID = 662253
today = datetime.now().strftime('%Y-%m-%d')
last_weeks = (datetime.now() - timedelta(days=21)).strftime('%Y-%m-%d')
df = get_statcast_data(last_weeks, today, MUÑOZ_ID)

st.title("🚀 Dashboard de Análisis: Andrés Muñoz")

# 4. ESTADÍSTICAS
st.header("📊 Estadísticas de Temporada")
c1, c2, c3 = st.columns(3)
c1.metric("ERA (Est.)", "1.73")
c2.metric("Strikeouts", "20")
c3.metric("Saves", "6")
st.divider()

# 5. MAPA INTERACTIVO
if not df.empty:
    st.header("🎯 Localización Interactiva")
    tipos = ['Todos'] + list(df['pitch_name'].unique())
    filtro = st.selectbox("Filtrar por tipo:", tipos, key="mapa_filtro")
    df_mapa = df if filtro == 'Todos' else df[df['pitch_name'] == filtro]

    fig = px.scatter(
        df_mapa, x='plate_x', y='plate_z', color='release_speed',
        custom_data=['release_speed', 'pitch_name', 'V_Break_in', 'H_Break_in', 'release_spin_rate', 'game_pk'],
        range_x=[-2, 2], range_y=[0, 5], color_continuous_scale='RdYlBu_r'
    )
    
    # Zona de Strike
    x_l, x_r, y_b, y_t = -0.85, 0.85, 1.5, 3.5
    fig.add_shape(type="rect", x0=x_l, y0=y_b, x1=x_r, y1=y_t, line=dict(color="Black", width=3))
    for x in [x_l + (x_r-x_l)/3, x_l + 2*(x_r-x_l)/3]:
        fig.add_shape(type="line", x0=x, y0=y_b, x1=x, y1=y_t, line=dict(color="Gray", dash="dash"))
    for y in [y_b + (y_t-y_b)/3, y_b + 2*(y_t-y_b)/3]:
        fig.add_shape(type="line", x0=x_l, y0=y, x1=x_r, y1=y, line=dict(color="Gray", dash="dash"))

    fig.update_layout(width=700, height=600)
    evento_click = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="mapa_principal")

    if evento_click and "selection" in evento_click and len(evento_click["selection"]["points"]) > 0:
        punto = evento_click["selection"]["points"][0]
        st.session_state['video_url'] = None # Reset
        
        game_id = int(punto["customdata"][5])
        st.session_state['video_url'] = buscar_video_oficial(game_id)
        
        st.success(f"📍 Seleccionado: {punto['customdata'][1]} | Juego ID: {game_id}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Velocidad", f"{punto['customdata'][0]:.1f} mph")
        m2.metric("V. Break", f"{punto['customdata'][2]:.1f} in")
        m3.metric("H. Break", f"{punto['customdata'][3]:.1f} in")
        m4.metric("Spin Rate", f"{int(punto['customdata'][4])} rpm")

st.divider()

# 6. ANÁLISIS DE MECÁNICA INTERACTIVO
# --- SECCIÓN 6: ANÁLISIS DE MECÁNICA OPTIMIZADO ---
st.header("🎥 Análisis de Mecánica con IA")

if 'fotos_mecanica' not in st.session_state:
    st.session_state['fotos_mecanica'] = {}

video_source = st.session_state.get('video_url')

uploaded_file = st.file_uploader("Sube un video manual", type=['mp4', 'mov'], key="uploader_mecanica")

if uploaded_file:
    video_source = uploaded_file

if video_source:
    if st.button("🚀 Iniciar Detección y Procesamiento"):
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_source.read())
            cap = cv2.VideoCapture(tfile.name)
        
        angles_list = []
        st.session_state['fotos_mecanica'] = {}
        frame_window = st.empty()
        
        # Reducimos confianza para mejorar detección en videos lejanos
        with mp_solutions_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # MEJORA 1: Aumentar brillo y contraste para ayudar a la IA
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                
                if results.pose_landmarks:
                    try:
                        landmarks = results.pose_landmarks.landmark
                        s = [landmarks[mp_solutions_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_solutions_pose.PoseLandmark.RIGHT_SHOULDER].y]
                        e = [landmarks[mp_solutions_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_solutions_pose.PoseLandmark.RIGHT_ELBOW].y]
                        w = [landmarks[mp_solutions_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_solutions_pose.PoseLandmark.RIGHT_WRIST].y]
                        
                        # Cálculo de ángulo (asegúrate de tener definida la función calculate_angle arriba)
                        def internal_calc_angle(a, b, c):
                            a, b, c = np.array(a), np.array(b), np.array(c)
                            ba, bc = a - b, c - b
                            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                        ang = internal_calc_angle(s, e, w)
                        angles_list.append(ang)
                        
                        img_draw = image.copy()
                        mp_drawing.draw_landmarks(img_draw, results.pose_landmarks, mp_solutions_pose.POSE_CONNECTIONS)
                        st.session_state['fotos_mecanica'][idx] = img_draw
                    except:
                        angles_list.append(None)
                
                idx += 1
                frame_window.image(image, channels="RGB", width=400)
            
            cap.release()
            st.session_state['df_cinetica'] = pd.DataFrame({'Frame': range(len(angles_list)), 'Angulo': angles_list}).dropna()

# --- MOSTRAR RESULTADOS E INTERACTIVIDAD ---
if 'df_cinetica' in st.session_state:
    st.divider()
    col_plot, col_img = st.columns([2, 1])
    
    with col_plot:
        st.subheader("📈 Trayectoria del Brazo")
        fig_ev = px.line(st.session_state['df_cinetica'], x='Frame', y='Angulo', markers=True)
        # Capturamos selección para mostrar la foto
        selected_data = st.plotly_chart(fig_ev, use_container_width=True, on_select="rerun", key="chart_mecanica")

    with col_img:
        st.subheader("📸 Frame Seleccionado")
# --- REEMPLAZA ESAS LÍNEAS POR ESTAS ---
    if selected_data and "selection" in selected_data and len(selected_data["selection"]["points"]) > 0:
        # Agregamos [0] para entrar al primer punto de la lista seleccionada
        punto = selected_data["selection"]["points"][0] 
        
        frame_idx = punto["x"]
        angulo_sel = punto["y"]
        
        if frame_idx in st.session_state['fotos_mecanica']:
            img_with_text = st.session_state['fotos_mecanica'][frame_idx].copy()
            
            # Texto del ángulo
            text = f"ANGULO: {int(angulo_sel)} deg"
            pos = (20, 50)
            
            # Dibujar texto (blanco con borde negro para que se vea bien)
            cv2.putText(img_with_text, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img_with_text, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        st.image(img_with_text, caption=f"Análisis del Frame {frame_idx}")
