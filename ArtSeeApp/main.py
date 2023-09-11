from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for your FastAPI app
origins = ["*"]  # You can specify specific origins or leave it as "*" to allow all
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# Store the uploaded image
stored_image = None

@app.post('/upload_image/')
async def upload_image(file: UploadFile):
    global stored_image
    if file.content_type.startswith("image/"):
        # Read the uploaded image data and store it
        image_data = await file.read()
        stored_image = image_data
        return {"message": "Image uploaded successfully"}
    else:
        return {"message": "Invalid file type. Please upload an image."}

@app.get('/get_image/')
async def get_image():
    if stored_image:
        return {"message": "Image retrieved successfully", "image": stored_image}
    else:
        return {"message": "No image uploaded."}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
