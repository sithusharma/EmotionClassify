from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # This will display the home page

@app.route('/start-classification', methods=['POST'])
def start_classification():
    # Here, you call the Python script that starts the camera and does emotion classification
    result = subprocess.run(['python3', 'EmotionClassify.py'], capture_output=True, text=True)
    # You can modify the above line to handle your script's specifics and how you want to capture its output

    return jsonify({"message": "Classification started", "output": result.stdout})

if __name__ == '__main__':
    app.run(debug=True)
