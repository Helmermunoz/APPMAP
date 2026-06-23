# 🚀 Biomechanical Analysis & Statcast Dashboard: Andrés Muñoz

This interactive web application is developed using **Python** and **Streamlit**. It is engineered to help baseball analysts, scouts, and coaches evaluate pitching performance and track the kinetic throwing mechanics of MLB pitcher Andrés Muñoz.

---

## 🎯 Key Features

*   **Live Season Statistics:** Direct dashboard display of critical performance metrics including estimated ERA, Strikeouts (SO), and Saves (SV).
*   **Interactive Pitch Location Map:** A dynamic scatter plot mapping pitch coordinates over the official strike zone using real-time *Statcast* data streams.
*   **Automated MLB Video Finder:** Dynamically queries and extracts play-by-play video highlights (`.mp4`) directly from the official MLB Stats API based on pitch selection.
*   **AI Biomechanical Analysis:** A computer vision module that processes pitching clips (via manual upload or API) to track arm joint landmarks (shoulder, elbow, wrist) and chart kinetic angles using **MediaPipe** and **OpenCV**.

---

## 🛠️ Tech Stack & Dependencies

The core infrastructure relies on the following data science and computer vision libraries:

*   **Streamlit:** Web interface framework and deployment dashboard.
*   **Pybaseball / Statsapi:** Data pipeline for historical data extraction and live MLB API sourcing.
*   **Plotly Express:** Dynamic rendering engine for the interactive strike-zone grid.
*   **MediaPipe (Pose Landmarker):** Core AI vision framework for body tracking and skeletal mapping.
*   **OpenCV:** Video stream manipulation, resolution handling, and brightness/contrast optimizations.
*   **Pandas & NumPy:** Core vector calculations, data modeling, and matrix structuring.

---

## 💻 Local Setup & Execution Instructions

Follow these steps to run this analytics platform locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com
cd APPMAP
```

### 2. Install Required Packages
Install all necessary development dependencies at once by executing:
```bash
pip install streamlit pybaseball plotly mediapipe opencv-python pandas numpy requests
```

### 3. Launch the Application
Run the local web server hosting the Streamlit app:
```bash
streamlit run AMapp.py
```
*Your system terminal will output a local network URL. A browser window will automatically open at `http://localhost:8501` displaying the dashboard.*

---

## 📊 Quick User Guide for Team Members

1.  **Pitch Filtering:** Use the dropdown menu above the **Interactive Location** map to filter by specific pitch types (e.g., *Four-Seam Fastball*, *Slider*).
2.  **Statcast Inspection:** Click directly on any data point inside the strike-zone scatter plot. The app will immediately pull up its specific velocity (mph), vertical/horizontal break values, and spin rate (rpm).
3.  **Computer Vision Pipeline:** Scroll down to the **AI Biomechanics** section. Drag and drop any `.mp4` or `.mov` pitching clip of Andrés Muñoz, and click **"Iniciar Detección y Procesamiento"**. The AI engine will overlay the skeleton tracking and generate a line graph plotting the kinetic arm trajectory frame by frame.
