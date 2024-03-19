from flask import Flask, render_template, request, redirect, url_for
import requests
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from pytube import YouTube


app = Flask(__name__)

model_path = 'models/deepfake_detector.h5'
model = load_model(model_path)

UPLOAD_FOLDER = 'instance/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

FRAMES_OUTPUT_FOLDER = 'instance/framesoutput'
app.config['FRAMES_OUTPUT_FOLDER'] = FRAMES_OUTPUT_FOLDER

ALLOWED_EXTENSIONS = {'mp4'}
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

API_KEY = 'AIzaSyD3Ffci03uxbcMGzHBEeRjoL2z7ljFxmW4'
BASE_URL = 'https://www.googleapis.com/youtube/v3'


def get_most_viewed_shorts():
    url = f'{BASE_URL}/videos'
    params = {
        'part': 'snippet,contentDetails,statistics',
        'chart': 'mostPopular',
        'regionCode': 'FR',  
        'videoCategoryId': '24',  
        'maxResults': 60, 
        'key': API_KEY,
    }

    response = requests.get(url, params=params)
    data = response.json()

    videos = []
    for item in data.get('items', []):
        video_id = item['id']
        title = item['snippet']['title']
        thumbnail_url = item['snippet']['thumbnails']['medium']['url']
        videos.append({'id': video_id, 'title': title, 'thumbnail_url': thumbnail_url})

    return videos

@app.route('/')
def accueil():
    return render_template('accueil.html', page='accueil')

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        pass
    return render_template('video.html', page='video')


@app.route('/youtube')
def youtube():
    video_list = get_most_viewed_shorts()
    return render_template('youtube.html', video_list=video_list)

@app.route('/equipe')
def equipe():
    return render_template('equipe.html', page='equipe')

@app.route('/contact')
def contact():
    return render_template('contact.html', page='contact')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path, output_folder, frame_rate=20):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_skip = int(fps / frame_rate)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_to_skip == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count // frames_to_skip:04d}.png")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extraction des frames terminée. {frame_count} frames extraites dans {output_folder}.")

def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0
    return img

def predict_frame(model, frame):
    img = preprocess_image(frame)
    prediction = model.predict(np.expand_dims(img, axis=0))
    return prediction[0][0]

def predict_frames(model, frame_folder):
    predictions = []
    for filename in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, filename)
        img = cv2.imread(frame_path)
        prediction = predict_frame(model, img)
        predictions.append(prediction)
    return np.array(predictions)

def analyze_video(video_path):
    frames_output_folder = os.path.join(app.config['FRAMES_OUTPUT_FOLDER'], secure_filename(video_path))
    extract_frames(video_path, frames_output_folder)
    predictions = predict_frames(model, frames_output_folder)
    fake_percentage = np.mean(predictions > 0.5) * 100
    print(fake_percentage)

    if fake_percentage >= 60:
        result = "La vidéo est un DeepFake"
    else:
        result = "La vidéo est Réelle"

    return result

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result = analyze_video(file_path)
        return render_template('result.html', result=result)

    return redirect(url_for('accueil'))  # Correction : Redirection vers la page d'accueil en cas d'erreur

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_filename(url):
    video_id = YouTube(url).video_id
    clean_video_id = ''.join(c for c in video_id if c.isalnum() or c in ['-', '_'])
    return clean_video_id + '.mp4'

@app.route('/process_youtube_url', methods=['POST'])
def process_youtube_url():
    url = request.form['youtube_url']
    if not url:
        return redirect(request.url)

    try:
        # Téléchargement de la vidéo depuis l'URL YouTube
        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').first()
        filename = secure_filename(clean_filename(url))
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.download(output_path=app.config['UPLOAD_FOLDER'], filename=filename)

        # Analyse de la vidéo
        result = analyze_video(video_path)
        
        return render_template('result.html', result=result)
    except Exception as e:
        print("Error:", e)
        return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
