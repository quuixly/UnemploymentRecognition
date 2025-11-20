## Unemployment Recognition

This project was created to develop a computer vision model that predicts whether a person is unemployed or employed based on their facial expression. If the person appears happy, they are classified as employed. After training the model, we deployed it on a Raspberry Pi with a connected camera to demonstrate the device’s capabilities.

## About the Model

The model uses a custom CNN architecture containing only 160k parameters. Due to the characteristics of the training dataset, you should position your face similarly to the one shown in this image: 

![Training example](training_example.png).

Move your head and test which position the model starts recognizing you (raising your eyebrows increases the model's confidence when smiling).

## Dataset
https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions

## Metrics

* Validation precision: 96%
* Validation recall: 96%
* Validation accuracy: 97%

## Installation
Make sure you have Python 3.13.7 installed. Then run the following commands in your terminal:
```bash
git clone https://github.com/quuixly/UnemploymentRecognition.git
cd UnemploymentRecognition
python -m venv .venv
source .venv/bin/activate
pip install opencv-python
pip install tensorflow
pip install numpy
python live.py
```
The application will run automatically.

## Future work
* add more training examples of the face from different angles and distances.
