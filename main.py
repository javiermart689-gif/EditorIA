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

   # 4. Formatear la línea de tiempo y detectar silencios
    timeline_segments = []
    silences = []
    
    # Umbral: ¿Cuántos segundos de silencio consideramos aburridos para cortar? (ej. 0.8 segundos)
    SILENCE_THRESHOLD = 0.8 
    last_end_time = 0.0

    if hasattr(transcription, 'segments'):
        for segment in transcription.segments:
            start_time = segment.get("start") if isinstance(segment, dict) else segment.start
            end_time = segment.get("end") if isinstance(segment, dict) else segment.end
            text = segment.get("text") if isinstance(segment, dict) else segment.text
            
            # Detectar si hubo un silencio largo antes de esta frase
            gap = start_time - last_end_time
            if gap >= SILENCE_THRESHOLD:
                silences.append({
                    "type": "silence_cut",
                    "start": round(last_end_time, 2),
                    "end": round(start_time, 2),
                    "duration": round(gap, 2)
                })

            # Guardar la frase hablada
            timeline_segments.append({
                "type": "spoken_clip",
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": text.strip()
            })
            
            last_end_time = end_time

    return {
        "status": "success", 
        "message": "Análisis completo listo para el editor",
        "timeline": {
            "spoken_segments": timeline_segments,
            "cuts_to_make": silences
        }
    }
