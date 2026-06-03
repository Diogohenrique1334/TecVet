import io

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from predictor import load_model, run_prediction

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10

app = FastAPI(title="TechVet API", description="Detecção automática de neutrófilos em imagens microscópicas")

modelo = load_model()


@app.post("/predict", summary="Detectar neutrófilos em uma imagem")
async def predict(file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Tipo de arquivo não suportado: {file.content_type}. Use JPEG, PNG ou WebP.")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Arquivo muito grande. Limite: {MAX_FILE_SIZE_MB}MB.")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prediction = run_prediction(modelo, image)

    annotated_bytes = io.BytesIO()
    prediction["annotated_image"].save(annotated_bytes, format="JPEG")
    annotated_bytes.seek(0)

    return StreamingResponse(
        annotated_bytes,
        media_type="image/jpeg",
        headers={"X-Neutrophil-Count": str(prediction["count"])},
    )
