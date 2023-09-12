from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for your FastAPI app
origins = ["*"]  # You can specify specific origins or leave it as "*" to allow all
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

stored_url = ""

@app.post('/store_url')
async def store_url(data: dict):
    global stored_url
    stored_url = data['url']
    return {"message": "URL stored successfully"}

@app.get('/get_url')
async def get_url():
    return {"url": stored_url}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
