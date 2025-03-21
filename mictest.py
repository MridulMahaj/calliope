import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)  # Reduces background noise

        try:
            audio = recognizer.listen(source, timeout=5)  # Listen for 5 seconds max
            print("Processing...")

            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text

        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError:
            print("Could not request results, check your internet connection.")
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period.")

if __name__ == "__main__":
    recognize_speech()
