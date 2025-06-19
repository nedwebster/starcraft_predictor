import base64
import io

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from starcraft_predictor import load_scorer

app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the scorer
replay_scorer = load_scorer()

@app.get("/api/health")
async def health_check() -> dict:
    return {"status": "healthy"}

@app.post("/api/score-replay")
async def score_replay(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    
    # Create a file-like object from the contents
    file_obj = io.BytesIO(contents)
    file_obj.name = file.filename
    
    # Score the replay
    figure = replay_scorer.score_replay(file_obj)

    # Convert the figure to base64
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    
    return {"image": img_str}
