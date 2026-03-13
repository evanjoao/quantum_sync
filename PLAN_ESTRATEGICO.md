# Plan Estratégico y Arquitectura Computacional (Implementado)

## 1) Pipeline general
1. **Setup de entorno Blender embebido**  
   Script: `setup_blender_deps.py`  
   Objetivo: localizar Blender + Python embebido e instalar `librosa`, `soundfile`, `numpy`, `scipy`, `pyopenvdb`.

2. **Extracción DSP (audio → eventos)**  
   Script: `audio_quantum_analyzer.py`  
   Clase principal: `AudioQuantumAnalyzer`.
   - `librosa.load` (mono)
   - `librosa.effects.hpss`
   - `librosa.onset.onset_detect`
   - `librosa.frames_to_time`
   - `librosa.feature.chroma_cqt`
   - Exporta `secuencia_estados.json`.

3. **Mapeo cuántico (pitch → n,l,m)**  
   Script: `mapeo_cuantico.py`  
   Clase principal: `MapeoCuantico`.
   - Transición física por energía fotónica (`E=hν`) y diferencias `ΔE` entre niveles de hidrógeno
   - Restricción física: `0 <= l < n` y `-l <= m <= l`
   - Enriquecimiento de JSON con `n,l,m`.

4. **Motor físico de función de onda**  
   Scripts: `hydrogen_physics.py` y `generar_frames_cuanticos.py`
   - Cálculo de `R_nl(r)` con `scipy.special.genlaguerre`
   - Cálculo angular `Y_l^m` con `scipy.special.sph_harm`
   - Muestreo de nube por `|ψ_nlm|²` y signo de fase real
   - Exportación `npz` compatible con `blender.py`: `coords` y `signs`

5. **Integración Blender procedural (bpy)**  
   Script: `blender_quantum_sync.py`.
   - Objeto contenedor volumétrico `Volumen_Atomo`
   - Material nodal `Quantum_Cloud` con `Principled Volume`
   - Custom properties: `quantum_n`, `quantum_l`, `quantum_m`
   - Keyframes por `timestamp * fps`
   - Interpolación forzada: `CONSTANT`
   - VSE: inserta audio y activa `scene.sync_mode = 'AUDIO_SYNC'`.

6. **Ruta alterna OpenVDB**  
   Script: `blender_openvdb_sync.py`.
   - Importa secuencia `.vdb`
   - Configura `sequence_mode`, `frame_start`, `frame_offset`
   - Keyframes de `frame_offset` sincronizados por timestamps
   - Interpolación `CONSTANT`.

---

## 2) Esquema de datos JSON
Formato de salida intermedio/final:

```json
[
  {
    "event_id": 1,
    "timestamp": 3.42,
    "pitch_dominant_id": 6,
    "n": 2,
    "l": 1,
    "m": 0
  }
]
```

---

## 3) Ejecución recomendada

### A. Instalar dependencias en Python de Blender
```bash
python quantum_sync/setup_blender_deps.py --verbose
```

### B. Extraer eventos DSP de `Sparks.wav`
```bash
python quantum_sync/audio_quantum_analyzer.py \
  --audio /ruta/absoluta/Sparks.wav \
  --output /home/evanj/Latex/quantum_sync/secuencia_estados.json
```

### C. Enriquecer JSON con estados cuánticos
```bash
python quantum_sync/mapeo_cuantico.py \
  --input /home/evanj/Latex/quantum_sync/secuencia_estados.json \
  --output /home/evanj/Latex/quantum_sync/secuencia_estados_enriquecida.json \
  --seed 42
```

### C2. Generar `atomo_frames_music.npz` con `|ψ|²` por evento
```bash
python quantum_sync/generar_frames_cuanticos.py \
   --events /home/evanj/Latex/quantum_sync/secuencia_estados_enriquecida.json \
   --output /home/evanj/Latex/atomo_frames_music.npz \
   --points 50000
```

### C3. Orquestación en un solo comando
```bash
python -m quantum_sync.run_full_pipeline \
   --audio /ruta/absoluta/Sparks.wav \
   --points 50000 \
   --seed 42
```

### D. Ejecutar en Blender (Scripting)
```python
from quantum_sync.blender_quantum_sync import run_pipeline

run_pipeline(
    events_json_path="/home/evanj/Latex/quantum_sync/secuencia_estados_enriquecida.json",
    audio_path="/ruta/absoluta/Sparks.wav",
)
```

### E. Alternativa VDB (en Blender)
```python
from quantum_sync.blender_openvdb_sync import run_openvdb_pipeline
import json

events = json.load(open("/home/evanj/Latex/quantum_sync/secuencia_estados_enriquecida.json", "r", encoding="utf-8"))
timestamps = [float(e["timestamp"]) for e in events]

run_openvdb_pipeline(
    vdb_first_file="/ruta/absoluta/frame_0001.vdb",
    timestamps_seconds=timestamps,
)
```

---

## 4) Observaciones técnicas
- El modelado de salto cuántico discreto queda garantizado por interpolación `CONSTANT` en F-Curves.
- El mapeo tonal→energético es heurístico y reproducible por semilla.
- El mapeo tonal→energético ahora es físico (transiciones por `ΔE` vs `E=hν`) y reproducible por semilla para `m`.
- `blender.py` y `atom.py` priorizan `/home/evanj/Latex/atomo_frames_music.npz` y usan fallback automático a `atomo_frames.npz`.
