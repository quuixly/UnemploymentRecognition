## Unemployment Recognition

This project was created to develop a computer vision model that predicts whether a person is unemployed or employed based on their facial expression. If the person appears happy, they are classified as employed. After training the model, we deployed it on a Raspberry Pi with a connected camera to demonstrate the device’s capabilities.

## About the Model

The model uses a custom CNN architecture containing only 160k parameters. Due to the characteristics of the training dataset, you should position your face similarly to the one shown in this image: 

![Training example](training_example.png).

(Make sure both ears are visible and the face is close enough (the forehead is visible, but not the hair))

)
## Dataset
https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions

## Metrics

* Validation precision: 96%
* Validation recall: 96%
* Validation accuracy: 97%

---
