import sys 
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet import preprocess_input

model = tf.keras.models.load_model("brain_tumor_model.h5") 

image_path=sys.argv[1]
img = load_img(image_path, target_size=(224, 224,3))

# Convert the image to a NumPy array
print("dd")
img_array = img_to_array(img)
normalized_img_array = (img_array/255)
input_tensor = tf.expand_dims(normalized_img_array, axis=0)

class_names = ['cancer', 'healthy']  # Replace with your actual class names
predictions = model.predict(input_tensor)
# Get the predicted class index
predicted_class_index = tf.argmax(predictions, axis=1)[0]

# Get the predicted class name
predicted_class_name = class_names[predicted_class_index]

# Display the predicted class name
print(f"Predicted class: {predicted_class_name}")