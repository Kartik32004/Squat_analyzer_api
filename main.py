import os
import cv2
import numpy as np
import tempfile
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your custom modules (make sure these files are in the same folder)
from utils import get_mediapipe_pose
from thresholds import get_thresholds_beginner, get_thresholds_pro
from process_frame import ProcessFrame

app = FastAPI(title="AI Fitness Trainer API")

# Allow CORS for local development/testing (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the AI Fitness Trainer API"}

@app.post("/upload_video")
async def upload_video(
    mode: str = Form("Beginner"),
    file: UploadFile = File(...)
):
    # Validate file type
    if file.content_type not in ["video/mp4", "video/quicktime", "video/x-msvideo"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Save uploaded video to a temporary file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_filename = tmp.name
    
    # Select thresholds based on mode
    if mode.lower() == "beginner":
        thresholds = get_thresholds_beginner()
    elif mode.lower() == "pro":
        thresholds = get_thresholds_pro()
    else:
        os.remove(tmp_filename)
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'Beginner' or 'Pro'.")

    # Initialize processing pipeline components
    process_frame = ProcessFrame(thresholds=thresholds)
    pose = get_mediapipe_pose()

    # Open the input video file
    cap = cv2.VideoCapture(tmp_filename)
    if not cap.isOpened():
        os.remove(tmp_filename)
        raise HTTPException(status_code=500, detail="Could not open video file")
    
    # Prepare an output video file (using a temporary file)
    output_filename = tempfile.mktemp(suffix=".mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Process the video frame-by-frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame from BGR to RGB as processing expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, _ = process_frame.process(frame_rgb, pose)
        # Write processed frame back in BGR format
        out_writer.write(processed_frame[..., ::-1])
    
    # Clean up and remove temporary input file
    cap.release()
    out_writer.release()
    os.remove(tmp_filename)

    # Return the processed video file as a downloadable response
    return FileResponse(output_filename, filename="processed_video.mp4", media_type="video/mp4")


# -----------------------
# WebSocket for Live Streaming
# -----------------------
@app.websocket("/ws/live_stream")
async def websocket_live_stream(websocket: WebSocket):
    await websocket.accept()
    
    # Create processing objects for live streaming.
    # (For live streaming you might want to tweak settings.)
    thresholds = get_thresholds_beginner()
    process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
    pose = get_mediapipe_pose()

    try:
        while True:
            # Expect the client to send a binary JPEG frame
            data = await websocket.receive_bytes()
            
            # Decode the frame from JPEG bytes
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Convert from BGR to RGB before processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame, _ = process_frame.process(frame_rgb, pose)
            
            # Encode the processed frame to JPEG to send back
            ret, buffer = cv2.imencode('.jpg', processed_frame[..., ::-1])
            if not ret:
                continue
            await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        print("WebSocket disconnected")
