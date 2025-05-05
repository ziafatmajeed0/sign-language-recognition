from flask import Flask, render_template, Response, request, jsonify
import cv2, numpy as np, time, threading, json
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
from googletrans import Translator

app = Flask(__name__)

# Load model and labels
labels = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','space',
    'T','U','V','W','X','Y','Z','Nothing'
]
model = load_model('model/optimized_asl_model.keras')

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Translator for English→Urdu
translator = Translator()

# Shared state
text_stream = {
    "alphabet": "", "word": "", "sentence": "",
    "translated_word": "", "translated_sentence": ""
}
char_history = []
last_displayed = ""
last_time = 0
delay = 1.0
camera_active = True
cap = None
prediction_paused = False

def speak_thread(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 130)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/toggle_prediction', methods=['POST'])
def toggle_prediction():
    global prediction_paused
    prediction_paused = not prediction_paused
    return jsonify(status="paused" if prediction_paused else "playing")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text_stream')
def text_stream_route():
    def event_stream():
        while True:
            yield f"data: {json.dumps(text_stream)}\n\n"
            time.sleep(0.2)
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/speak_word', methods=['POST'])
def speak_word():
    word = text_stream["word"].strip()
    if word:
        speak_thread(word)
    return ('', 204)

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    sent = text_stream["sentence"].strip()
    if sent:
        speak_thread(sent)
    return ('', 204)

@app.route('/translate_word', methods=['POST'])
def translate_word():
    word = text_stream["word"].strip()
    if word:
        translated = translator.translate(word, src='en', dest='ur').text
        text_stream["translated_word"] = translated
    return ('', 204)

@app.route('/translate_sentence', methods=['POST'])
def translate_sentence():
    sentence = text_stream["sentence"].strip()
    if sentence:
        translated = translator.translate(sentence, src='en', dest='ur').text
        text_stream["translated_sentence"] = translated
    return ('', 204)

@app.route('/reset', methods=['POST'])
def reset():
    global char_history, last_displayed
    char_history.clear()
    last_displayed = ""
    text_stream.update(
        alphabet="", word="", sentence="",
        translated_word="", translated_sentence=""
    )
    return ('', 204)

def generate_frames():
    global cap, char_history, last_displayed, last_time, prediction_paused

    # Ensure the camera is initialized correctly and once
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            return

    while True:
        if not camera_active:
            time.sleep(0.1)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            continue

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        stable_char = ""
        now = time.time()

        # Draw all detected hands
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Recognition only if not paused
        if results.multi_hand_landmarks and not prediction_paused:
            for handLms in results.multi_hand_landmarks:
                xs = [lm.x for lm in handLms.landmark]
                ys = [lm.y for lm in handLms.landmark]
                data = []
                for lm in handLms.landmark:
                    data.extend([lm.x - min(xs), lm.y - min(ys)])
                if len(data) < 42:
                    data += [0] * (42 - len(data))

                pred = model.predict(np.array(data).reshape(1,42,1), verbose=0)
                ch = labels[np.argmax(pred[0])]

                char_history.append(ch)
                if len(char_history) > 20:
                    char_history.pop(0)

                if char_history.count(ch) > 15 and ch != "Nothing":
                    stable_char = "" if ch == "space" else ch

                    if stable_char and now - last_time > delay:
                        last_time = now
                        text_stream["word"] += stable_char
                        text_stream["alphabet"] = stable_char

                    if ch == "space" and text_stream["word"].strip():
                        text_stream["sentence"] += text_stream["word"] + " "
                        text_stream["word"] = ""
        else:
            char_history.clear()

        # Update UI
        if stable_char:
            last_displayed = stable_char
        text_stream["alphabet"] = last_displayed

        if stable_char:
            cv2.putText(frame, stable_char, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 3)

        ret, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
