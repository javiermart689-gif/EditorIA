from fastapi import FastAPI, UploadFile, File, HTTPException
import ffmpeg
import os
import shutil

app = FastAPI(title="AI Video Editor API")

# Crear carpetas temporales
os.makedirs("temp_videos", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

@app.post("/upload-and-extract/")
async def process_video(file: UploadFile = File(...)):
    # Validar formato
    if not file.filename.lower().endswith(('.mp4', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Formato de video no soportado")

    video_path = f"temp_videos/{file.filename}"
    # Cambiar la extensión a .wav para el audio
    audio_path = f"temp_audio/{file.filename.rsplit('.', 1)[0]}.wav"

    # Guardar el video temporalmente
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extraer el audio usando FFmpeg
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise HTTPException(status_code=500, detail="Error extrayendo el audio")

    return {
        "status": "success", 
        "message": "Audio extraído y listo",
        "video_path": video_path,
        "audio_path": audio_path
    }