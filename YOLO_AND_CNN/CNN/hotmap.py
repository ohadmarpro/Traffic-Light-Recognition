import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# נתיבים
model_path = r"D:\trafic\aa\bb\cnn\best_cnn_model_high_end.h5"
img_path = r"D:\trafic\fin_vid\part 2\relv\red.jpg"# שנה לפי תמונה שתרצה

# הגדרות
img_size = (224, 224)
class_names = ['green', 'red', 'red_orange', 'yellow']

# טען את המודל
model = load_model(model_path)

# טען את התמונה
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img) / 255.0
img_array_exp = np.expand_dims(img_array, axis=0)

# ניבוי
pred = model.predict(img_array_exp)[0]
pred_class_idx = np.argmax(pred)
pred_class = class_names[pred_class_idx]
print(f"🔍 זיהוי: {pred_class} ({pred[pred_class_idx]*100:.2f}%)")

# קבל את הפלט של השכבה הקונבולוציונית האחרונה
last_conv_layer_name = None
for layer in reversed(model.layers):
    if 'conv' in layer.name:
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name is None:
    raise Exception("❌ לא נמצאה שכבת Conv במודל")

# צור מודל שייתן גם את הפלט של השכבה הקונבולוציונית האחרונה וגם את הפלט הסופי
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

# הפעל GradientTape כדי לחשב את הגרדיאנטים
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array_exp)
    loss = predictions[:, pred_class_idx]

# חישוב גרדיאנט
grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# שילוב משוקלל ליצירת CAM
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

# נרמול
heatmap = np.maximum(heatmap, 0)
heatmap /= tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# הצגת מפת חום על התמונה המקורית
img_orig = cv2.imread(img_path)
img_orig = cv2.resize(img_orig, img_size)
heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_color, 0.4, 0)

# הצג
cv2.imshow(f"CAM for class: {pred_class}", superimposed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
