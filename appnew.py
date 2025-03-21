import glob
import json
import os
import re
import cv2
import mysql.connector
import numpy as np
import openai
import requests
from flask import Flask, session, request, jsonify, render_template
from paddleocr import PaddleOCR
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

from mysql.connector import Error
import random
import shutil

# Flask App Configuration
app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\coolw\PycharmProjects\calliope\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#Upload folder
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB Limit
app.secret_key = "your_secret_key"

# OpenAI API Key
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)

ocr = PaddleOCR(use_angle_cls=True, lang="en", ocr_version="PP-OCRv4")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image file: {image_path}")

    return image_path


def extract_text_and_boxes(image_path):
    preprocessed_path = preprocess_image(image_path)
    result = ocr.ocr(preprocessed_path, cls=True)

    extracted_text = ""
    bounding_boxes = []

    for line in result:
        if line:
            for word_info in line:
                text, confidence = word_info[1]
                box = word_info[0]
                extracted_text += text + " "
                bounding_boxes.append((box, text, confidence))

    extracted_text = extracted_text.strip()
    character_count = len(extracted_text.replace(" ", ""))
    word_count = len(extracted_text.split())

    return extracted_text, character_count, word_count, bounding_boxes


def draw_text_boxes(image_path, bounding_boxes, output_path):
    img = cv2.imread(image_path)

    for box, text, confidence in bounding_boxes:
        points = np.array(box, dtype=np.int32)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = int(points[0][0]), int(points[0][1])
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imwrite(output_path, img)

@app.route("/paddle_ocr_analysis", methods=["POST"])
def paddle_ocr_analysis():
    if "user_file" not in request.files:
        return jsonify({"error": "User file is required."}), 400

    user_file = request.files["user_file"]
    if not allowed_file(user_file.filename):
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg."}), 400

    filename = secure_filename(user_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    user_file.save(file_path)

    try:
        extracted_text, char_count, word_count, bounding_boxes = extract_text_and_boxes(file_path)

        if not extracted_text:
            return jsonify({"error": "No text detected in the uploaded image."}), 400

 #       # Generate highlighted image
        highlighted_path = os.path.splitext(file_path)[0] + "_highlighted.jpg"
        draw_text_boxes(file_path, bounding_boxes, highlighted_path)

        # Prepare JSON result
        analysis_result = {
            "extracted_text": extracted_text,
            "character_count": char_count,
            "word_count": word_count,
            "bounding_boxes": bounding_boxes,
            "highlighted_image": highlighted_path
        }

        # Save JSON to a .txt file
        analysis_filename = os.path.splitext(filename)[0] + "_analysis_result.txt"
        analysis_file_path = os.path.join(app.config["UPLOAD_FOLDER"], analysis_filename)

        with open(analysis_file_path, "w") as txt_file:
            json.dump(analysis_result, txt_file, indent=4)

        return jsonify({
            "message": "PaddleOCR analysis completed.",
            "analysis_file": analysis_file_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import openai
import base64
import json
import re

def analyze_handwriting(image_path):
    try:
        # Convert image to Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # GPT-4o Vision API Call - Prompt to Openai
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a handwriting analysis expert. Evaluate the handwriting for clarity, stroke accuracy, and consistency."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Analyze the handwriting in this image. "
                            "Provide letter-wise scores (0-100) based on clarity and consistency. "
                            "List the top 5 and worst 5 letters detected in the text"
                            "Ensure that you only use the letters present in the text and score accordingly"
                            "Return **only valid JSON** in this exact format:\n"
                            "{\n"
                            "  \"extracted text\": \"just the extracted text from the image\"\n"
                            "  \"scores\": {\"A\": 85, \"B\": 90, ...},\n"
                            "  \"top_5\": [\"A\", \"B\", \"C\", \"D\", \"E\"],\n"
                            "  \"bottom_5\": [\"X\", \"Y\", \"Z\", \"W\", \"V\"],\n"
                            "  \"feedback\": \"Detailed handwriting feedback\"\n"
                            "}"
                        )},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=700
        )
        #Corrected Response Parsing
        raw_response = response.choices[0].message.content.strip()
        cleaned_response = re.sub(r"```json\n(.*?)\n```", r"\1", raw_response, flags=re.DOTALL)
        print("GPT-4o Cleaned Response:", cleaned_response)
        try:
            analysis_result = json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse GPT-4o response. Response was not valid JSON.",
                    "raw_response": cleaned_response}

        return analysis_result

    except openai.OpenAIError as e:  #Exception Handling
        return {"error": f"OpenAI API call failed: {str(e)}"}


#Route calling analyze text function above
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    print("Received request for /analyze_text")  # Debug log

    if "user_file" not in request.files:
        print("Error: No file uploaded")  # Debug log
        return jsonify({"error": "User file is required."}), 400

    user_file = request.files["user_file"]
    if not allowed_file(user_file.filename):
        print(f"Invalid file type: {user_file.filename}")  # Debug log
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg."}), 400

    filename = secure_filename(user_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # Ensure UPLOAD_FOLDER exists
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    user_file.save(file_path)
    print(f"File saved to: {file_path}")  # Debug log

    try:
        analysis_result = analyze_handwriting(file_path)
        print(f"Analysis result: {analysis_result}")  # Debug log

        # Save analysis as a JSON-formatted .txt file
        analysis_filename = os.path.splitext(filename)[0] + "_analysis_text.txt"
        analysis_file_path = os.path.join(app.config["UPLOAD_FOLDER"], analysis_filename)

        with open(analysis_file_path, "w") as json_file:
            json.dump(analysis_result, json_file, indent=4)

        return jsonify(analysis_result)

    except Exception as e:
        print(f"Error: {e}")  # Debug log
        return jsonify({"error": str(e)}), 500

# Done Text extraction using ML (PaddleOCR) and then analysis using OpenAI prompt engineering for returning of
# extracted text, scores for letters present, top5 and bottom5 and feedback
#This result is stored in the text file in format file_name_analysis_text.txt
#I will return this back to user

# Flask endpoint to send saved analysis JSON to frontend
@app.route("/send_to_frontend", methods=["GET"])
def send_to_frontend():
    filename = request.args.get("filename")  # Get filename from query params
    if not filename:
        return jsonify({"error": "Filename is required."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found."}), 404

    try:
        with open(file_path, "r") as file:
            return jsonify(json.load(file)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#Check if user is present in the database
#If not return an error
@app.route('/check_user', methods=['POST'])
def check_if_user_exists():
    if request.method == 'POST':
        # Get JSON data from request
        data = request.get_json()
        mobile_number = data.get('mobile_number')

        # Validate mobile number
        if not mobile_number:
            return jsonify({"status": "error", "message": "Mobile number is required"}), 400

        # Establish a database connection
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(buffered=True)  # Use buffered cursor to fetch all results immediately
            try:
                # Query to check if the mobile number exists
                cursor.execute("SELECT * FROM user_registration WHERE mobile_number = %s", (mobile_number,))
                user = cursor.fetchone()  # Fetch the result

                # If user exists, return an "exists" message
                if user:
                    return jsonify({"status": "exists", "message": "Mobile number already registered"}), 200
                else:
                    return jsonify({"status": "not_found", "message": "Mobile number not found"}), 404

            except Error as e:
                print(f"Database error: {e}")
                return jsonify({"status": "error", "message": "Database error"}), 500
            finally:
                cursor.close()
                conn.close()
        else:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500

#Now for user login and registration + mobile number + otp
def send_otp(mobile_number):
    url = f"https://cpaas.messagecentral.com/verification/v3/send?countryCode=91&customerId=C-E414ED19519F4B5&flowType=SMS&mobileNumber={mobile_number}"

    headers = {
        'authToken': 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJDLUU0MTRFRDE5NTE5RjRCNSIsImlhdCI6MTczOTEwMzk0MSwiZXhwIjoxODk2NzgzOTQxfQ.SSkAmumOLsaCJrM9yuEy6D6rvy7t3vVErz--dZvUB0K41046nUqzhNraibvhHib5Wzfq-nks-UMawX80TL6CcA'
    }

    response = requests.post(url, headers=headers)

    print(f"Sending OTP to {mobile_number}: {response.status_code}, Response: {response.text}")

    if response.status_code == 200:
        try:
            response_data = response.json()

            # The 'verificationId' is inside the 'data' object
            data = response_data.get('data')
            if data:
                verification_id = data.get('verificationId')
                if verification_id:
                    return verification_id
                else:
                    print("Verification ID not found in 'data'.")
                    return None
            else:
                print("No 'data' object in the response.")
                return None
        except ValueError:
            print("Failed to parse JSON response.")
            return None
    else:
        print(f"Failed to send OTP, status code: {response.status_code}")
        return None

# Route to handle mobile number submission and OTP sending
@app.route('/submit_mobile', methods=['POST'])
def submit_mobile():
    data = request.get_json()  # Get the JSON data from the request
    if not data or 'mobile_number' not in data:
        return jsonify({"status": "error", "message": "Mobile number is required."}), 400

    mobile_number = data['mobile_number']  # Extract the mobile number
    print(f"Received mobile number: {mobile_number}")

    # Validate mobile number
    if not mobile_number or len(mobile_number) != 10 or not mobile_number.isdigit():
        return jsonify({"status": "error", "message": "Invalid mobile number."}), 400

    # Send the OTP and get verification ID
    verification_id = send_otp(mobile_number)

    if not verification_id:
        return jsonify({"status": "error", "message": "Failed to send OTP."}), 500

    # Store verification ID in the session
    session['verification_id'] = verification_id
    session['mobile_number'] = mobile_number  # Store mobile number as well

    # Return a success response in JSON format
    return jsonify({"status": "success", "message": "OTP sent successfully."}), 200


# Route to handle OTP verification (verification_id is handled in the backend)
@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    mobile_number = session.get('mobile_number')  # Retrieve mobile number from session
    verification_id = session.get('verification_id')  # Retrieve verification ID from session

    if not mobile_number or not verification_id:
        return jsonify({"status": "error", "message": "No mobile number or verification ID found in session."}), 400

    # Get JSON data from the request (OTP sent by frontend)
    data = request.get_json()
    if not data or 'otp' not in data:
        return jsonify({"status": "error", "message": "OTP is required."}), 400

    otp = data['otp']  # Extract OTP from the request

    # Check if mobile number exists in the database
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT mobile_number FROM user_registration WHERE mobile_number = %s", (mobile_number,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({"status": "error", "message": "Mobile number already registered."}), 400  # Mobile number exists in DB

    cursor.close()
    conn.close()

    # Validate the OTP using stored verification ID
    otp_validation_response = validate_otp(mobile_number, verification_id, otp)

    if otp_validation_response['status'] == "success":
        return jsonify(otp_validation_response), 200
    else:
        return jsonify({"status": "error", "message": otp_validation_response.get("message", "Failed to verify OTP.")}), 400


# Function to validate the OTP
def validate_otp(mobile_number, verification_id, code):
    url = f"https://cpaas.messagecentral.com/verification/v3/validateOtp?countryCode=91&mobileNumber={mobile_number}&verificationId={verification_id}&customerId=C-E414ED19519F4B5&code={code}"

    headers = {
        'authToken': 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJDLUU0MTRFRDE5NTE5RjRCNSIsImlhdCI6MTczOTEwMzk0MSwiZXhwIjoxODk2NzgzOTQxfQ.SSkAmumOLsaCJrM9yuEy6D6rvy7t3vVErz--dZvUB0K41046nUqzhNraibvhHib5Wzfq-nks-UMawX80TL6CcA'
    }

    response = requests.get(url, headers=headers)

    print(f"Validating OTP for {mobile_number}: {response.status_code}, Response: {response.text}")

    if response.status_code == 200:
        return {"status": "success", "message": "OTP verified successfully!"}
    else:
        return {"status": "error", "message": response.json()}

#For user registration and insertion into handwriting analysis table


# User registration is based on only Mobile number and password
#Since we have mobile number as primary key we will return the scores associated with that number back to the user
#That means only 1 account per user based on mobile
# MySQL connection configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'mridul1608',
    'database': 'CALLIOPE'
}

# Establish MySQL Connection
def get_db_connection():
    return mysql.connector.connect(**db_config)

@app.route("/register_user", methods=["POST"])
def register_user():
    try:
        data = request.json
        mobile_number = data.get("mobile_number")
        password = data.get("password")

        if not mobile_number or not password:
            return jsonify({"error": "Mobile number and password are required."}), 400

        hashed_password = generate_password_hash(password)  # Hash the password

        conn = get_db_connection()
        cursor = conn.cursor()

        query = "INSERT INTO user_registration (mobile_number, password) VALUES (%s, %s)"
        cursor.execute(query, (mobile_number, hashed_password))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "User registered successfully."}), 201

    except mysql.connector.IntegrityError:
        return jsonify({"error": "Mobile number already exists."}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/store_handwriting_analysis", methods=["POST"])
def store_handwriting_analysis():
    try:
        data = request.json
        user_mobile = data.get("user_mobile")
        extracted_text = data.get("extracted_text")
        scores = data.get("scores")
        top_5 = data.get("top_5")
        bottom_5 = data.get("bottom_5")
        character_count = data.get("character_count")
        word_count = data.get("word_count")
        feedback = data.get("feedback")

        if not all([user_mobile, extracted_text, scores, top_5, bottom_5, character_count, word_count, feedback]):
            return jsonify({"error": "All fields are required."}), 400

        # Convert JSON objects to strings for database storage
        scores_json = json.dumps(scores)
        top_5_json = json.dumps(top_5)
        bottom_5_json = json.dumps(bottom_5)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT mobile_number FROM user_registration WHERE mobile_number = %s", (user_mobile,))
        if cursor.fetchone() is None:
            return jsonify({"error": "User does not exist."}), 404

        # Insert analysis data
        query = """
        INSERT INTO handwriting_analysis (user_mobile, extracted_text, scores, top_5, bottom_5, character_count, word_count, feedback)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (user_mobile, extracted_text, scores_json, top_5_json, bottom_5_json, character_count, word_count, feedback))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "Handwriting analysis stored successfully."}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#Create assessment routes

# Strip Markdown Code Blocks
def strip_code_blocks(response_text):
    if response_text.startswith("```json") and response_text.endswith("```"):
        return response_text[7:-3].strip()
    return response_text


# Generate Questions using OpenAI
def generate_questions(language, segment_name):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": (
                    f"You are an expert language tutor creating multiple-choice questions for {language} learners. "
                    f"Generate exactly 20 multiple-choice questions for the segment '{segment_name}'. "
                    f"The question_text must be in ENGLISH only"
                    "Each question must be structured as follows:\n"
                    "- 'question_text': A meaningful language learning question.\n"
                    "- 'correct_answer': The correct answer.\n"
                    "- 'wrong_answers': A list of exactly 3 unique incorrect but plausible answers.\n"
                    "Return ONLY a valid JSON array with 20 questions, nothing else."
                )}
            ],
            max_tokens=2000
        )

        response_content = response.choices[0].message.content.strip()
        response_content = strip_code_blocks(response_content)

        questions = json.loads(response_content)

        if not isinstance(questions, list) or len(questions) != 20:
            raise ValueError("OpenAI did not return exactly 20 questions.")

        for q in questions:
            if not q.get("question_text") or not q.get("correct_answer") or not q.get("wrong_answers"):
                raise ValueError("One or more questions have missing fields.")
            if not isinstance(q["wrong_answers"], list) or len(q["wrong_answers"]) != 3:
                raise ValueError("Each question must have exactly 3 wrong answers.")

        return questions
    except openai.OpenAIError as e:
        return {"error": f"OpenAI API error: {str(e)}"}
    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# API to Generate & Store Questions
@app.route("/generate_questions", methods=["POST"])
def generate_questions_for_chapter():
    data = request.get_json()
    language = data.get("language")
    chapter_number = data.get("chapter_number")
    segment_name = data.get("segment_name")

    if language not in ["French", "German", "Tamil", "Malayalam"]:
        return jsonify({"error": "Invalid language."}), 400

    #Segments will not change irrespective of language
    segment_mapping = {
        1: ["Introduction", "Numbers and Days", "Basic Sentences"],
        2: ["Colors, Shapes & Sizes", "Personality & Emotions", "Talking About Places"],
        3: ["In a Restaurant & Café", "At the Airport & Hotel", "Public Transport & Directions"],
        4: ["Agreeing & Disagreeing", "Giving Reasons & Justifications", "Talking About Hobbies & Interests"]
    }

    if chapter_number not in segment_mapping or segment_name not in segment_mapping[chapter_number]:
        return jsonify({"error": f"Invalid segment for Chapter {chapter_number}."}), 400

    segment_number = segment_mapping[chapter_number].index(segment_name) + 1

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        questions = generate_questions(language, segment_name)
        if "error" in questions:
            return jsonify({"error": questions["error"]}), 500

        for q in questions:
            cursor.execute(
                f"SELECT COUNT(*) FROM {language} WHERE chapter_number = %s AND segment_number = %s AND question_text = %s",
                (chapter_number, segment_number, q["question_text"])
            )
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    f"INSERT INTO {language} (chapter_number, segment_number, segment_name, question_text, correct_answer, wrong_answers) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (chapter_number, segment_number, segment_name, q["question_text"], q["correct_answer"],
                     json.dumps(q["wrong_answers"]))
                )
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify(
            {"message": f"Questions stored for {language} - {segment_name} (Chapter {chapter_number})."}), 201

    except mysql.connector.Error as db_err:
        conn.rollback()
        return jsonify({"error": f"Database error: {str(db_err)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API to Retrieve 10 Random Questions
@app.route("/get_questions", methods=["POST"])
def get_questions():
    try:
        data = request.get_json()
        language = data.get("language")
        chapter_number = data.get("chapter_number")
        segment_name = data.get("segment_name")

        if language not in ["French", "German", "Spanish", "Tamil", "Malayalam"]:
            return jsonify({"error": "Invalid language."}), 400

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            f"SELECT id, question_text, correct_answer, wrong_answers FROM {language} "
            "WHERE chapter_number = %s AND segment_name = %s",
            (chapter_number, segment_name)
        )

        questions = cursor.fetchall()
        cursor.close()
        conn.close()

        for q in questions:
            try:
                q["wrong_answers"] = json.loads(q["wrong_answers"])
            except json.JSONDecodeError:
                q["wrong_answers"] = []

        random.shuffle(questions)
        return jsonify({"questions": questions[:10]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#To do - Translate, Writing, Speaking, Dictionary
from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    translations = {}

    target_languages = {
        "French": "fr",
        "Italian": "it",
        "Spanish": "es",
        "Portuguese": "pt",
        "Romanian": "ro",
        "German": "de",
        "Dutch": "nl",
        "Swedish": "sv",
        "Norwegian": "no",
        "Danish": "da",
        "Finnish": "fi",
        "Polish": "pl",
        "Hungarian": "hu",
        "Czech": "cs",
        "Slovak": "sk",
        "Tamil" : "ta",
        "Telugu": "te",
        "Malayalam": "ml",
        "Kannada": "kn",
        "Hindi": "hi",
        "Bengali": "bn"
    }

    for lang_name, lang_code in target_languages.items():
        try:
            translated_text = GoogleTranslator(source="en", target=lang_code).translate(text)
            translations[lang_name] = translated_text
        except Exception as e:
            translations[lang_name] = f"Error: {str(e)}"

    return jsonify({
        "original_text": text,
        "translations": translations
    })

#Translation for latin based languages fixed and completed
#Now, will start speech recognition using Speech Recognition Library
from flask import Flask, jsonify, request
import speech_recognition as sr
import openai
import json
import time
import os

# Store the last transcribed text globally
last_transcription = {"text": None}


def record_audio():
    recognizer = sr.Recognizer()
    filename = f"recorded_{int(time.time())}.wav"  # Generate unique filename

    with sr.Microphone() as source:
        print("Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5)
            print("Processing...")

            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())

            return filename

        except sr.WaitTimeoutError:
            return None


@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    print("Received request to /analyze_speech")

    # Check if the 'file' is in the request
    file = request.files.get('file')
    if not file:
        print("No file found in request.")
        return jsonify({"error": "No file uploaded."}), 400

    if file.filename == '':
        print("Empty file received.")
        return jsonify({"error": "No selected file."}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    print(f"File saved: {file_path}")

    recognizer = sr.Recognizer()

    try:
        # Load and transcribe audio
        with sr.AudioFile(file_path) as source:
            print("Loading file for transcription...")
            audio = recognizer.record(source)

        transcribed_text = recognizer.recognize_google(audio)
        print(f"Transcribed Text: {transcribed_text}")

        # OpenAI GPT-4o Analysis
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are a language learning assistant. Analyze the spoken sentence for pronunciation, "
                    "grammar, and fluency. Return feedback in JSON format."
                )},
                {"role": "user", "content": (
                    f"Analyze this spoken sentence: \"{transcribed_text}\".\n"
                    "Provide structured feedback in JSON format:\n"
                    "{\n"
                    "  'pronunciation': 'string',\n"
                    "  'grammar': 'string',\n"
                    "  'fluency': 'string'\n"
                    "}"
                )}
            ],
            max_tokens=500
        )

        response_content = response.choices[0].message.content.strip()

        # Extract JSON content safely
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if not json_match:
            raise ValueError("Invalid JSON response from OpenAI.")

        feedback = json.loads(json_match.group(0))

        return jsonify({"success": True, "text": transcribed_text, "analysis": feedback})

    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio.")
        return jsonify({"error": "Could not understand the audio."}), 400
    except sr.RequestError:
        print("Speech recognition API request failed.")
        return jsonify({"error": "Speech recognition request failed. Check internet connection."}), 500
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {str(e)}")
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

#To do - Generate route for dictionary, I use gpt 4 here coz gpt 4o is having some issues
#gpt 4o would still work unfortunately the request takes a lot of time for this
#Dictionary should include words, meanings, synonyms and antonyms
def generate_dictionary(language):
    try:
        # Call OpenAI's chat completions API to generate dictionary entries
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a language learning expert. Always return valid JSON with no extra text."},
                {"role": "user", "content": (
                    f"Generate exactly 5 words to be included in a dictionary for the specified {language}. "
                    "For every word the key should be 'word'"
                    "Every word must have :\n"
                    "- A non-empty 'meaning' which includes the meaning of the word (string) with key 'meaning'\n"
                    "- A non-empty list of 'synonym' which includes 3 synonyms for the word with key 'synonyms', if no synonym "
                    "exists "
                    "return an empty array.\n"
                    "- A non-empty list of 'antonyms' which includes 3 antonyms for the word with key 'antonyms' , if no antonyms exists "
                    "return an empty array.\n"
                    "Return ONLY a valid JSON array with the above format:\n"
                )}
            ],
            max_tokens=700
        )

        # Correctly access the response content
        response_content = response.choices[0].message.content
        print(response_content)

        # Handle potential wrapping in backticks (```json ... ```)
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[7:-3].strip()

        # Parse the JSON content
        words = json.loads(str(response_content))

        # Validate the response structure
        if not isinstance(words, list) or len(words) != 5:
            raise ValueError("OpenAI did not return exactly 5 words")

        return words

    except openai.OpenAIError as e:
        print("OpenAI API error:", str(e))
        return {"error": f"OpenAI API error: {str(e)}"}
    except ValueError as ve:
        print("ValueError:", str(ve))
        return {"error": str(ve)}
    except Exception as e:
        print("Unexpected error:", str(e))
        return {"error": f"Unexpected error: {str(e)}"}


@app.route('/generate_and_store_dictionary', methods=['POST'])
def generate_and_store_dictionary():
    data = request.get_json()

    # Check if language is provided
    if not data.get('language'):
        return jsonify({"error": "Language is required"}), 400

    language = data['language']

    try:
        print("working")
        # Call OpenAI to generate dictionary words
        words = generate_dictionary(language)
        print("working2")

        # Check for errors from the generate function
        if 'error' in words:
            return jsonify(words), 500

        # Debugging: Log the raw response
        print("OpenAI Response:", words)

        # Store the dictionary words in the database
        connection = get_db_connection()
        cursor = connection.cursor()

        for word in words:
            print(word)
            cursor.execute('''
                INSERT INTO dictionary_words (language, word, meaning, synonyms, antonyms)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                language,
                word.get('word', ''),  # assuming the word itself is included in the response or adjusting as needed
                word.get('meaning', ''),
                json.dumps(word.get('synonyms', [])),
                json.dumps(word.get('antonyms', []))
            ))

        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({"message": "Dictionary words successfully generated and stored!"}), 200

    except Exception as e:
        # Log the exception to help debug
        print("Error:", str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

#To do send the questions back to frontend
@app.route('/get_dictionary', methods=['POST'])
def get_dictionary():
    data = request.get_json()

    # Check if language is provided
    if not data or 'language' not in data:
        return jsonify({"error": "Language is required"}), 400

    language = data['language']

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute('''
            SELECT word, meaning, synonyms, antonyms FROM dictionary_words
            WHERE language = %s
        ''', (language,))

        words = cursor.fetchall()
        cursor.close()
        connection.close()

        # Convert database results into JSON format
        dictionary_list = [
            {
                "word": word[0],
                "meaning": word[1],
                "synonyms": json.loads(word[2]) if word[2] else [],
                "antonyms": json.loads(word[3]) if word[3] else []
            }
            for word in words
        ]

        return jsonify({"language": language, "dictionary": dictionary_list}), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



#Dictionary finishes here


#To do -> include regional languages in the app - Tamil and Malayalam tables and implementation done
#More languages can still be added though
#Focus - Gamification (3 hearts)- done in frontend, save user progress - table implemented
#Visualization of mistakes, letters might take some time

#Visualization of Mistakes begins here
#I will make a line graph like in Knowledge Pro

# Root endpoint
# Graph endpoint
UPLOAD_FOLDER = r"C:\Users\coolw\PycharmProjects\calliope\uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER  # Ensure UPLOAD_FOLDER is set

@app.route("/graph")
def graphpage():
    analysis_files = sorted(
        glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*_analysis_text.txt")),
        key=os.path.getmtime,
        reverse=True
    )

    if not analysis_files:
        return jsonify({"error": "No analysis data available"}), 404  # If no files, return error

    latest_analysis_file = analysis_files[0]  # Pick the latest file

    try:
        with open(latest_analysis_file, "r") as text_file:
            analysis_result = eval(text_file.read())  # Convert back to dictionary

        scores = analysis_result.get("scores", {})
        labels = list(scores.keys())
        data = list(scores.values())

        return render_template("chartjs-example.html", labels=labels, data=data)

    except Exception as e:
        return jsonify({"error": f"Failed to read analysis file: {str(e)}"}), 500
@app.route("/save_progress", methods=["POST"])
def save_progress():        
    data = request.json
    mobile_number = data.get("mobile_number")
    language = data.get("language")
    chapter_number = data.get("chapter_number")
    segment_name = data.get("segment_name")
    scores = data.get("scores")  # Received from frontend

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if progress already exists for the mobile_number
    cursor.execute(
        "SELECT * FROM user_progress WHERE mobile_number = %s AND language = %s AND chapter_number = %s AND "
        "segment_name = %s",
        (mobile_number, language, chapter_number, segment_name),
    )
    existing = cursor.fetchone()

    if existing:
        # Update existing progress
        cursor.execute(
            "UPDATE user_progress SET scores = %s WHERE mobile_number = %s AND language = %s AND chapter_number = %s "
            "AND segment_name = %s",
            (scores, mobile_number, language, chapter_number, segment_name),
        )
    else:
        # Insert new progress entry
        cursor.execute(
            "INSERT INTO user_progress (mobile_number, language, chapter_number, segment_name, scores) VALUES (%s, %s, "
            "%s, %s, %s)",
            (mobile_number, language, chapter_number, segment_name, scores),
        )

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Progress saved successfully!"})

@app.route('/get_user_progress', methods=['POST'])
def get_user_progress():
    data = request.get_json()
    mobile_number = data.get('mobile_number')

    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)

            # Check if user exists in user_registration
            cursor.execute("SELECT * FROM user_registration WHERE mobile_number = %s", (mobile_number,))
            user = cursor.fetchone()

            if not user:
                return jsonify({"status": "error", "message": "User not found."}), 404

            # Fetch progress details based on mobile_number
            cursor.execute("SELECT * FROM user_progress WHERE mobile_number = %s", (mobile_number,))
            progress_details = cursor.fetchall()

            return jsonify({
                "status": "success",
                "user_progress": progress_details  # Sends empty list if no progress
            }), 200

        except mysql.connector.Error as e:
            print(f"Database error: {e}")  # Print the full database error for debugging
            return jsonify({"status": "error", "message": f"Database error: {str(e)}"}), 500

    return jsonify({"status": "error", "message": "Database connection failed"}), 500

# Flask Endpoints

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message" : "Hello from calliope"}),200
@app.route("/test", methods=["GET"])
def home_test():
    return jsonify({"message": "Flask app is running!"}), 200

if __name__ == '__main__':
    app.run(debug=True)