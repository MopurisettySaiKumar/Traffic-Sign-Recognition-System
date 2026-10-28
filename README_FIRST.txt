TRAFFIC SIGN RECOGNITION SYSTEM — REAL AI MINI PROJECT

WINDOWS SETUP
1. Install Python 3.11 (64-bit) from python.org and check "Add Python to PATH".
2. Extract this ZIP into a normal folder (e.g. Desktop\TSR_AI_Expo).
3. Connect laptop to internet for first setup (packages and pretrained model).
4. Double-click START_WINDOWS.bat. First setup may take several minutes.
5. Open http://127.0.0.1:5000 in Chrome or Edge.
6. Test image upload, live camera, and uploaded video.

VS CODE MANUAL COMMANDS (terminal opened in project folder)
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python download_model.py
python app.py
Then open http://127.0.0.1:5000

MODEL: https://huggingface.co/ankitjha07/Traffic-Sign-Detection-YOLOv8
This project does NOT fake predictions. Weights are downloaded on first run, not included in ZIP.
No accuracy or Indian-sign coverage is guaranteed. Model is third-party and has not been validated on this laptop.
Only model-trained sign categories can be recognized. This is an educational demo, not a safety product.
Video output is MP4 (mp4v codec), which may need VLC on Windows.
Video limited to first 1800 frames. Processing is synchronous and can be slow on CPU.
If model download fails, manually download best.pt from the model URL above and put it inside models\best.pt.

MODEL LICENSE AND ATTRIBUTION
Model is by ankitjha07 on Hugging Face. Check the model page for licensing and training details before redistribution.
Ultralytics package has licensing terms; check https://www.ultralytics.com/license .
