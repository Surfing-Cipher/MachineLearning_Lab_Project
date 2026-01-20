========================================================================
             FACE MASK DETECTION FLASK APP - RUNNING INSTRUCTIONS
========================================================================

1.  SETUP ENVIRONMENT
    -----------------

    a) Activate the Virtual Environment (RECOMMENDED):
       - Windows (PowerShell):
           .venv\Scripts\Activate
       - Windows (Command Prompt):
           .venv\Scripts\activate.bat
       - Mac/Linux:
           source .venv/bin/activate

    b) Install Dependencies:
       pip install -r requirements.txt

    This will install Flask, TensorFlow, OpenCV, and other necessary libraries.

2.  RUN THE APP
    -----------
    Run the following command:

        python app.py

    You should see output indicating the server is starting, like:
    "Running on http://127.0.0.1:5000"

3.  USE THE APP
    -----------
    Open your web browser (Chrome/Edge recommended) and go to:
    
        http://127.0.0.1:5000/

    - Click "Real-Time Demo" to start your webcam.
      (Allow browser permission if asked).
    
    - Click "Image Upload" to choose a file (JPG/PNG) and see the result.

========================================================================
                               TROUBLESHOOTING
========================================================================
- ERROR: "Cannot open camera"
  -> Make sure no other app (Zoom, Teams) is using the webcam.
  
- ERROR: "Module not found"
  -> Re-run `pip install -r requirements.txt`.

- ERROR: "Model not found" or "Cascade not found"
  -> Ensure `face_mask_detector_final.h5` and `haarcascade_frontalface_default.xml`
     are in the SAME folder as `app.py`.

- SLOW PERFORMANCE?
  -> Deep Learning on CPU can be slow. It's normal for the video to lag 
     slightly depending on your computer's speed.
