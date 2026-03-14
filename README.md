# neural-storyteller
Neural Storyteller
AI Image Captioning System

Neural Storyteller is an AI-based image captioning application that generates natural language descriptions for images using deep learning. The system combines a powerful convolutional neural network for image understanding with a recurrent neural network for language generation.

Users can upload an image through a modern web interface and instantly receive a caption describing the visual content.

Project Demo

Upload an image and the model will generate a caption such as:

A cat sitting on a white background

The system analyzes the visual features of the image and converts them into a human-readable sentence.

Features

Automatic image caption generation

Deep learning–based vision–language model

Interactive web interface

Image upload support

Real-time caption generation

Clean and responsive UI

Model Architecture

The model follows an Encoder–Decoder architecture commonly used in image captioning systems.

Encoder (Vision Model)

A pretrained ResNet-50 model extracts high-level visual features from images.

The encoder:

processes the image

extracts a feature vector

passes the features to the language decoder

Decoder (Language Model)

The decoder is a Long Short-Term Memory (LSTM) network that generates captions word-by-word.

The decoder:

receives image features

predicts the next word in the sentence

continues until the <end> token is generated

Dataset

The model was trained using the Flickr30k dataset.

Dataset details:

31,000 images

5 captions per image

natural language descriptions of scenes

Example caption:

A man riding a bicycle down a street
Technologies Used

This project uses several modern machine learning and web technologies:

PyTorch — deep learning framework

TorchVision — pretrained CNN models

Streamlit — web application framework

Hugging Face Hub — model hosting

Python — core programming language

Project Structure
neural-storyteller
│
├── app.py
├── requirements.txt
├── README.md
└── assets (optional)

Model artifacts stored on Hugging Face:

config.json
model_weights.pt
vocab.pkl
Installation

Clone the repository:

git clone https://github.com/yourusername/neural-storyteller.git

Navigate to the project folder:

cd neural-storyteller

Install dependencies:

pip install -r requirements.txt
Running the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser at:

http://localhost:8501
How It Works

User uploads an image

The image is processed using ResNet50

Extracted features are passed to the LSTM decoder

The decoder generates a caption word-by-word

The caption is displayed in the interface

Deployment

The application can be deployed easily using Streamlit Community Cloud.

Deployment steps:

Push code to GitHub

Connect repository to Streamlit Cloud

Deploy the application

After deployment, users can access the app through a public URL.

Future Improvements

Possible future enhancements include:

Beam search caption generation

Attention-based captioning model

Transformer-based architecture

Multi-language caption generation

Caption confidence scores

Caption editing suggestions

Applications

Image captioning systems have many real-world applications:

Assistive technology for visually impaired users

Automatic image indexing

Content moderation

Image search systems

Social media accessibility features

Author

Developed by:

Usman Anwar

Software Engineering Student
AI and Machine Learning Enthusiast

License

This project is released under the MIT License.
