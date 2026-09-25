# SCT_ML_4 – Hand Gesture Recognition (CNN)
A Convolutional Neural Network that classifies hand gesture images into 10 categories, built as Task 4 of my Machine Learning Internship at SkillCraft Technology.

🎯 Task
Build a model to classify different hand gestures from image data, using the LeapGestRecog dataset.

⚙️ How it works
Grayscale hand gesture images are resized and fed into a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The model learns to recognize 10 distinct gestures directly from the raw images.

✋ Gestures Classified
Palm · L · Fist · Fist Moved · Thumb · Index · OK · Palm Moved · C · Down
📊 Results
Test Accuracy: 100%

🚀 Try it yourself
bash
python predict_gesture.py your_image.png

Give it any gesture image from the dataset — it'll tell you which gesture it predicts with a confidence score.

🛠️ Built with
Python · TensorFlow · Keras · NumPy · Matplotlib · Pillow

📁 Files
File	Description
task04_gesture.py --Trains the CNN model
predict_gesture.py --Test the model on any gesture image
training_curves.png	--Accuracy/loss over training epochs
confusion_matrix.png --Model performance per gesture class
sample_predictions.png --Example correct predictions
