import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# נתיבים
base_dir = r'D:\trafic\aa\bb\cnn\cropped\dataset_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
model_save_path = r'D:\trafic\aa\bb\cnn\best_cnn_model_high_end.h5'

# הגדרות
img_size = (224, 224)
batch_size = 16
epochs_stage1 = 10
epochs_stage2 = 20
num_classes = 4

# Augmentation חכם
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.7, 1.3],
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# חישוב class weights
class_indices = train_generator.class_indices
labels = train_generator.classes
weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# בניית המודל עם ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
]

# שלב ראשון: רק השכבות העליונות
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_stage1,
    class_weight=class_weights,
    callbacks=callbacks
)

# שלב שני: fine-tune כל המודל
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_stage2,
    class_weight=class_weights,
    callbacks=callbacks
)

print("✅ האימון הסתיים. המודל נשמר כ:", model_save_path)
