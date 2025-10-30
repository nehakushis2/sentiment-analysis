#!/usr/bin/env python
# coding: utf-8

# In[23]:


import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Use your own audio file (e.g., speech.wav)
audio_file = "C:/Users/Neha/Downloads/WhatsApp Ptt 2025-10-26 at 20.27.00.wav"

# Load the audio file
with sr.AudioFile(audio_file) as source:
    audio = recognizer.record(source)

# Convert speech to text
try:
    text = recognizer.recognize_google(audio)
    print("Transcribed Text:", text)
except sr.UnknownValueError:
    print("Speech could not be understood")
except sr.RequestError:
    print("Error with the Speech Recognition service")


# In[24]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

sid = SentimentIntensityAnalyzer()
scores = sid.polarity_scores(text)
print(scores)


# In[25]:


from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

result = sentiment_pipeline(text)
print(result)


# In[26]:


import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def speech_sentiment(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣️ Transcribed Text: {text}")
    except Exception as e:
        print("Speech Recognition Error:", e)
        return
    
    # Sentiment
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    
    sentiment = (
        "Positive" if scores['compound'] > 0.05 else
        "Negative" if scores['compound'] < -0.05 else
        "Neutral"
    )
    
    print(f"Sentiment: {sentiment}")
    print(f"Scores: {scores}")

# Example
speech_sentiment(audio_file)


# In[ ]:




