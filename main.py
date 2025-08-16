import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

# Import the main processing function from your other file
from ai_pipeline import process_video_pipeline

app = FastAPI(title="AI Dubbing Service")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Dubbing API!"}


@app.post("/dub-video/")
async def dub_video_endpoint(video: UploadFile = File(...)):
    """
    This endpoint accepts an uploaded video, processes it through the full AI pipeline,
    and returns the final dubbed video file.
    """
    # Create a temporary directory to store the uploaded file
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define the path for the temporary input file
    input_video_path = os.path.join(temp_dir, video.filename)
    
    # Save the uploaded video to the temporary path
    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    print(f"File '{video.filename}' saved to '{input_video_path}'")

    # --- Call the AI Pipeline ---
    # We pass the path of the saved video to our processing function
    # The function will return the path of the final output video
    final_video_path = process_video_pipeline(video_path=input_video_path)

    # Return the processed video file as a download
    return FileResponse(path=final_video_path, media_type='video/mp4', filename="dubbed_video.mp4")