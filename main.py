from fastapi import FastAPI, File, UploadFile
from typing import List
import shutil
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

app = FastAPI()

# Load the model
model = tf.keras.models.load_model("src/keras_model_v2.h5", compile=False)
class_names = open("src\labels.txt", "r", encoding="utf-8").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return class_names[index][2:], round((prediction[0][index] * 100), 2)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result_name, confidence_score = predict(file.filename)
        return {"result": result_name, "confidence": confidence_score}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)