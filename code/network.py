from keras.models import Sequential,Model,load_model,save_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization,Flatten,Conv2D,MaxPooling2D,Input,LeakyReLU,ReLU,Softmax
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy


def build_model_2(x_shape):
    inputs = Input(shape=x_shape)
    x = Conv2D(2*16, kernel_size=(5, 5), strides=(1, 1), padding='valid')(inputs)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(x)
    x = Conv2D(4*16, kernel_size=(5, 5), strides=(1, 1), padding='valid')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=(1, 2), padding='valid')(x)
    x = Conv2D(8*16, kernel_size=(3, 3), strides=(2, 1), padding='valid')(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dense(32)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(4)(x)
    predictions = Softmax()(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


def build_model_lstm(window_size,num_of_features,dropout,label_length,optimizer):

    # Create a model
    model = Sequential()
    model.add(LSTM(512, return_sequences=False, input_shape=(window_size, num_of_features)))
    model.add(Dropout(dropout))
    model.add(Dense(label_length, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model







def build_model(x_shape,num_classes):
    inputs = Input(shape=x_shape)
    x = Conv2D(16,kernel_size=(5,5),input_shape=(100,40,1),padding='same',use_bias=True)(inputs)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x= Dropout(0.1)(x)
    x= Conv2D(2*16, (5,5), padding='same',use_bias=True)(x)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x= MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x= Dropout(0.1)(x)
    x= Conv2D(4*16, (5,5),padding='same',use_bias=True)(x)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    x= MaxPooling2D(pool_size=(2,2),padding='same')(x)
    x= Flatten()(x)
    x= Dense(64)(x)
    x=ReLU()(x)
    x = Dense(32)(x)
    x = ReLU()(x)
    x = Dense(16)(x)
    x = ReLU()(x)
    x=Dense(num_classes)(x)
    predictions = Softmax()(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model