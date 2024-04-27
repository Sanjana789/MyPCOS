from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import joblib

from neuralintents.assistants import BasicAssistant
from flask import Flask, request, jsonify, render_template
import joblib

def chatbot_intents():
    intents = {
    "intents": [
        {
        "tag": "symptom_inquiry",
        "patterns": [
            "What are the symptoms of PCOS?",
            "How do I know if I have PCOS?",
            "What are common signs of PCOS?"
        ],
        "responses": [
            "Common symptoms of PCOS include irregular periods, excess hair growth, and acne."
        ],
        "context_set": ""
        },
        {
        "tag": "diagnosis_assistance",
        "patterns": [
            "How is PCOS diagnosed?",
            "What tests are used to diagnose PCOS?",
            "Do I need to see a doctor for a PCOS diagnosis?"
        ],
        "responses": [
            "PCOS is typically diagnosed through blood tests, pelvic exams, and ultrasound."
        ],
        "context_set": ""
        },
        {
        "tag": "treatment_options",
        "patterns": [
            "What treatments are available for PCOS?",
            "How can I manage PCOS symptoms?",
            "Are there medications for PCOS?"
        ],
        "responses": [
            "Treatment options for PCOS include lifestyle changes, medications, and fertility treatments."
        ],
        "context_set": ""
        },
        {
        "tag": "nutritional_advice",
        "patterns": [
            "What diet is best for PCOS?",
            "How can I improve my diet with PCOS?",
            "Are there specific foods to avoid with PCOS?"
        ],
        "responses": [
            "A low-glycemic index diet and balanced meals are recommended for managing PCOS symptoms."
        ],
        "context_set": ""
        },
        {
        "tag": "exercise_recommendations",
        "patterns": [
            "What exercise is recommended for PCOS?",
            "How often should I exercise with PCOS?",
            "Are there specific types of exercise for PCOS?"
        ],
        "responses": [
            "Regular physical activity, such as cardio and strength training, is beneficial for managing PCOS."
        ],
        "context_set": ""
        },
        {
        "tag": "emotional_support",
        "patterns": [
            "How can I cope with the emotional effects of PCOS?",
            "Where can I find support for PCOS?",
            "Are there online communities for PCOS?"
        ],
        "responses": [
            "Strategies for coping with the emotional challenges of PCOS include seeking support from others and joining online communities."
        ],
        "context_set": ""
        },
        {
        "tag": "fertility_concerns",
        "patterns": [
            "Does PCOS affect fertility?",
            "What are my chances of getting pregnant with PCOS?",
            "Are there fertility treatments for PCOS?"
        ],
        "responses": [
            "PCOS can affect fertility, but there are fertility treatments available for individuals with PCOS."
        ],
        "context_set": ""
        },
        {
        "tag": "long_term_health_risks",
        "patterns": [
            "What are the long-term health risks of PCOS?",
            "How does PCOS affect my health in the long run?",
            "Am I at risk for other conditions because of PCOS?"
        ],
        "responses": [
            "Long-term health risks associated with PCOS include type 2 diabetes, heart disease, and endometrial cancer."
        ],
        "context_set": ""
        }
    ]
    }


    assistant = BasicAssistant(intents)

    assistant.fit_model(epochs=50)
    assistant.save_model()

    done = False
    return assistant

# while not done:
#     message = input("Enter a message: ")
#     if message == "STOP":
#         done = True
#     else:
#         print(assistant.process_input(message))

def pcoschatbotpredict(assistant,message):
  return assistant.process_input(message)