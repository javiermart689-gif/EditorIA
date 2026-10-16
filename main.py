from fastapi import FastAPI, UploadFile, File, HTTPException
import ffmpeg
import os
import shutil
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
# Inicializamos el cliente de Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="AI Video Editor API")

os.makedirs("temp_videos", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

@app.post("/upload-and-extract/")
async def process_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Formato de video no soportado")

    video_path = f"temp_videos/{file.filename}"
    audio_path = f"temp_audio/{file.filename.rsplit('.', 1)[0]}.wav"

    # 1. Guardar el video
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

   # 2. Extraer audio con FFmpeg
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        # Aquí capturamos el error interno de FFmpeg
        error_detail = e.stderr.decode('utf-8') if e.stderr else "Error desconocido"
        raise HTTPException(status_code=500, detail=f"FFmpeg falló por: {error_detail}")

    # 3. Transcribir con el modelo Whisper alojado en Groq
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en IA de Groq: {str(e)}")

    # 4. Formatear la línea de tiempo
    timeline_segments = []
    
    # Groq devuelve los segmentos directamente si pedimos verbose_json
    if hasattr(transcription, 'segments'):
        for segment in transcription.segments:
            # Los segmentos en Groq pueden venir como diccionarios o como objetos
            start_time = segment.get("start") if isinstance(segment, dict) else segment.start
            end_time = segment.get("end") if isinstance(segment, dict) else segment.end
            text = segment.get("text") if isinstance(segment, dict) else segment.text
            
            timeline_segments.append({
                "start": start_time,
                "end": end_time,
                "text": text.strip()
            })

    return {
        "status": "success", 
        "message": "Audio transcrito con Groq gratuitamente",
        "transcription": timeline_segments
    }
