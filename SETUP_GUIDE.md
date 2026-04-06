# Face Recognition Attendance System - Live Registration Guide

## Localhost Web Version

If you want to run the project in your browser on your own PC, use `web_app.py`.

## Step 1: Install Required Libraries

Open PowerShell in your project folder and run:

```bash
pip install flask opencv-contrib-python numpy
```

If you already installed `opencv-python`, remove it first:

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## Step 2: Run the Application

```bash
python web_app.py
```

Then open your browser and go to:

```text
http://127.0.0.1:5000
```

## Step 3: Register Faces Live (No Image Upload)

1. Stand in front of webcam.
2. In the browser, type the person name in the register box.
3. Click Register Face.
4. App captures multiple live samples and trains automatically.

No folder upload is required.

## Step 4: Automatic Recognition and Attendance

After registration:
- Known faces are recognized live.
- Attendance is stored in attendance.csv.
- Unknown faces are labeled as UNKNOWN.

Each person is marked only once per day.

## Keyboard Controls

- Use the browser form to register faces
- Stop the app by pressing Ctrl+C in the terminal

## Files Created Automatically

- dataset/: stores captured training face images
- labels.json: maps numeric labels to person names
- trainer.yml: trained OpenCV LBPH model
- attendance.csv: stores daily attendance records

The browser page also has a Download CSV button.

## Project Structure

```text
face recognition/
|- face_recognition_app.py
|- web_app.py
|- SETUP_GUIDE.md
|- templates/
|  |- index.html
|- dataset/
|- labels.json
|- trainer.yml
|- attendance.csv
```

## Troubleshooting

1. Webcam not opening:
	- Close other camera apps and try again.

2. Face not registering:
	- Ensure one clear face is visible and room lighting is good.

3. Face recognized as UNKNOWN:
	- Register again in better light.

4. Name not saved:
	- Enter non-empty name when prompted in terminal.

5. Error about cv2.face missing:
	- Run: pip uninstall opencv-python
	- Then run: pip install opencv-contrib-python

6. Browser page does not load:
	- Make sure `web_app.py` is running and open `http://127.0.0.1:5000`
