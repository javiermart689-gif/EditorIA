import os
import re
import shutil
import uuid
import ffmpeg
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

load_dotenv()

# Inicializamos el cliente de Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="AI Video Editor API")

# Permitir conexiones de la página web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp_videos", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# Base de datos temporal en memoria para tareas en segundo plano
jobs = {}

# Clases para la exportación y datos
class Clip(BaseModel):
    start: float
    end: float

class ExportRequest(BaseModel):
    filename: str
    clips: List[Clip]


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Falta el archivo index.html</h1>"


# ==========================================
# TAREA EN SEGUNDO PLANO (Background Task)
# ==========================================
def process_video_background(job_id: str, video_path: str, audio_path: str):
    jobs[job_id] = {"status": "processing", "message": "Extrayendo audio con FFmpeg..."}
    
    try:
        # 1. Extraer audio
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        
        jobs[job_id]["message"] = "Transcribiendo con Inteligencia Artificial (Groq)..."
        
        # 2. Transcribir
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )
        
        jobs[job_id]["message"] = "Analizando silencios y muletillas..."
        
        # 3. Formatear
        timeline_segments = []
        silences = []
        SILENCE_THRESHOLD = 0.8 
        last_end_time = 0.0
        filler_words = [r"\beh\b", r"\beste\b", r"\bmmm+\b", r"\bah\b", r"\bpues\b"]
        filler_pattern = re.compile("|".join(filler_words), re.IGNORECASE)

        if hasattr(transcription, 'segments'):
            for segment in transcription.segments:
                start_time = segment.get("start") if isinstance(segment, dict) else segment.start
                end_time = segment.get("end") if isinstance(segment, dict) else segment.end
                text = segment.get("text") if isinstance(segment, dict) else segment.text
                
                gap = start_time - last_end_time
                if gap >= SILENCE_THRESHOLD:
                    silences.append({
                        "type": "silence_cut",
                        "start": round(last_end_time, 2),
                        "end": round(start_time, 2),
                        "duration": round(gap, 2)
                    })

                has_filler = bool(filler_pattern.search(text))
                
                timeline_segments.append({
                    "type": "spoken_clip",
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "text": text.strip(),
                    "flag_filler": has_filler
                })
                last_end_time = end_time

        # 4. Guardar resultado final listo para el frontend
        jobs[job_id] = {
            "status": "success", 
            "message": "Análisis completo listo para el editor",
            "timeline": {
                "spoken_segments": timeline_segments,
                "cuts_to_make": silences
            }
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": f"Fallo interno: {str(e)}"}


@app.post("/upload-and-extract/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Formato de video no soportado")

    job_id = str(uuid.uuid4())
    video_path = f"temp_videos/{file.filename}"
    audio_path = f"temp_audio/{file.filename.rsplit('.', 1)[0]}.wav"

    # Guardar video localmente de forma rápida
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Enviar la tarea pesada al fondo para no bloquear el servidor
    background_tasks.add_task(process_video_background, job_id, video_path, audio_path)

    return {"job_id": job_id, "status": "processing", "message": "Video en cola"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Turno no encontrado")
    return jobs[job_id]


@app.post("/export/")
async def export_video(request: ExportRequest):
    input_path = f"temp_videos/{request.filename}"
    output_path = f"temp_videos/final_{request.filename}"

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="El video original ya no está en el servidor.")

    try:
        input_video = ffmpeg.input(input_path)
        streams = []
        
        for clip in request.clips:
            v = input_video.video.trim(start=clip.start, end=clip.end).setpts('PTS-STARTPTS')
            a = input_video.audio.filter('atrim', start=clip.start, end=clip.end).filter('asetpts', 'PTS-STARTPTS')
            streams.extend([v, a])
        
        joined = ffmpeg.concat(*streams, v=1, a=1).node
        
        (
            ffmpeg
            .output(joined[0], joined[1], output_path)
            .overwrite_output()
            .run(quiet=True)
        )
        
        return FileResponse(output_path, media_type="video/mp4", filename=f"Editado_{request.filename}")
        
    except ffmpeg.Error as e:
        error_detail = e.stderr.decode('utf-8') if e.stderr else "Error desconocido"
        raise HTTPException(status_code=500, detail=f"Error cortando el video: {error_detail}")
