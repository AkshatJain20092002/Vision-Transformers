import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

# Load the test dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Custom Patches layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Custom PatchEncoder layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# Load the ViT classifier model and class names
with keras.utils.custom_object_scope({'Patches': Patches, 'PatchEncoder': PatchEncoder}):
    vit_classifier = keras.models.load_model("vit_classifier_model.h5")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to get model prediction
def img_predict(images, model):
    if len(images.shape) == 3:
        # If the input is a single image, convert it to a batch with a single sample
        images = np.expand_dims(images, axis=0)

    out = model.predict(images)
    prediction = np.argmax(out, axis=1)
    img_pred = [class_names[i] for i in prediction]
    return img_pred

# Main Streamlit app
def main():
    st.title("ViT Cifar10 Transformers")
    st.write("Available Classes: ", class_names)
    
    # Input index
    index_label = "<b>Enter the index:</b>"
    st.markdown(index_label, unsafe_allow_html=True)
    index = st.number_input("", min_value=0, max_value=len(x_test)-1, step=1)

    # Display image for the given index
    st.image(x_test[index], use_column_width=True)

     # Get model prediction
    prediction = img_predict(x_test[index], vit_classifier)
    
# Show the model prediction in bold with increased font size
    st.markdown("<p style='font-size:24px;'><b>Model Prediction:</b> " + prediction[0] + "</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()