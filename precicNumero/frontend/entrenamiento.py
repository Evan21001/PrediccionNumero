import os
import numpy as np
import tensorflow as tf

# Dimensiones de entrada definidas por el estándar MNIST
INPUT_SHAPE    = (28, 28, 1)
NUM_CLASSES    = 10
BATCH_SIZE     = 64
TRAINING_EPOCHS = 10
TFJS_OUTPUT_PATH = "modelo_web"


def load_and_prepare_data():
    """Descarga MNIST y escala los valores de píxel al rango [0, 1]."""
    print("Obteniendo el dataset de dígitos MNIST...")
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.mnist.load_data()

    train_images = train_images[..., np.newaxis].astype("float32") / 255.0
    test_images  = test_images[..., np.newaxis].astype("float32")  / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def build_classifier():
    """
    Construye una CNN de dos bloques convolucionales.
    Los mapas de características van de 32 a 64 filtros con max-pooling
    para reducción espacial; la cabeza densa usa dropout para evitar
    sobreajuste en el conjunto de entrenamiento.
    """
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return cnn


def train_and_export():
    (train_images, train_labels), (test_images, test_labels) = \
        load_and_prepare_data()

    cnn_model = build_classifier()

    print("\nIniciando entrenamiento del clasificador...")
    cnn_model.fit(
        train_images, train_labels,
        epochs=TRAINING_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(test_images, test_labels),
    )

    # Artefacto Keras nativo como respaldo local
    cnn_model.save("modelo_mnist.h5")
    print("\nModelo Keras guardado → 'modelo_mnist.h5'")

    # Conversión al formato LayersModel de TensorFlow.js para inferencia en el navegador
    print("\nConvirtiendo al formato TensorFlow.js...")
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(cnn_model, TFJS_OUTPUT_PATH)

    separador = "=" * 55
    print(f"\n{separador}")
    print("Exportación completada. Fragmentos binarios en:")
    print(f"  {TFJS_OUTPUT_PATH}/")
    print(separador)


if __name__ == "__main__":
    train_and_export()
