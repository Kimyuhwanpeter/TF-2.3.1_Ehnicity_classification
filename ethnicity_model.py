import tensorflow as tf

layers = tf.keras.layers

def ethnic_model(input_shape=(224, 224, 3), 
                 include_top=True, 
                 pooling=None, 
                 weights="vggface",
                 weight_path=""):

    h = inputs = tf.keras.Input(input_shape)
    # Block 1
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
            h)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, name='fc6')(x)
        x = layers.Activation('relu', name='fc6/relu')(x)
        x = layers.Dense(4096, name='fc7')(x)
        x = layers.Activation('relu', name='fc7/relu')(x)
        x = layers.Dense(2622, name='fc8')(x)
        x = layers.Activation('softmax', name='fc8/softmax')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(inputs, x, name='vggface_vgg16')  # load weights

    if weights == 'vggface':
        if include_top:
            weights_path = weight_path
            print("==============================================")
            print("Loading the vggface weight include top...")
        else:
            weights_path = weight_path
            print("==============================================")
            print("Loading the vggface weight only conv layers...")

        model.load_weights(weights_path, by_name=True)
        print("Got weights from file!!")
        print("==============================================")

    return model