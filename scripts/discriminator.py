from tensorflow import keras
from tensorflow.keras import layers as lay


#discriminator model
D = keras.Sequential()
depth = 64
dropout = 0.4
# In: 28 x 28 x 1, depth = 1
# Out: 14 x 14 x 1, depth=64
input_shape = (256, 256, 3)
D.add(lay.Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same', activation= lay.LeakyReLU(alpha=0.2)))
D.add(lay.Dropout(dropout))
D.add(lay.Conv2D(depth*2, 5, strides=2, padding='same', activation=lay.LeakyReLU(alpha=0.2)))
D.add(lay.Dropout(dropout))
D.add(lay.Conv2D(depth*4, 5, strides=2, padding='same', activation=lay.LeakyReLU(alpha=0.2)))
D.add(lay.Dropout(dropout))
D.add(lay.Conv2D(depth*8, 5, strides=1, padding='same', activation=lay.LeakyReLU(alpha=0.2)))
D.add(lay.Dropout(dropout))
# Out: 1-dim probability
D.add(lay.Flatten())
D.add(lay.Dense(1))
D.add(lay.Activation('sigmoid'))
D.summary()

#generator model
G = keras.Sequential()
dropout = 0.4
depth = 64+64+64+64
dim = 7
# In: 100
# Out: dim x dim x depth
G.add(lay.Dense(dim*dim*depth, input_dim=100))
G.add(lay.BatchNormalization(momentum=0.9))
G.add(lay.Activation('relu'))
G.add(lay.Reshape((dim, dim, depth)))
G.add(lay.Dropout(dropout))
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
G.add(lay.UpSampling2D())
G.add(lay.Conv2DTranspose(int(depth/2), 5, padding='same'))
G.add(lay.BatchNormalization(momentum=0.9))
G.add(lay.Activation('relu'))
G.add(lay.UpSampling2D())
G.add(lay.Conv2DTranspose(int(depth/4), 5, padding='same'))
G.add(lay.BatchNormalization(momentum=0.9))
G.add(lay.Activation('relu'))
G.add(lay.Conv2DTranspose(int(depth/8), 5, padding='same'))
G.add(lay.BatchNormalization(momentum=0.9))
G.add(lay.Activation('relu'))
# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
G.add(lay.Conv2DTranspose(1, 5, padding='same'))
G.add(lay.Activation('sigmoid'))
G.summary()

optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
DM = keras.Sequential()
DM.add(keras.layers.discriminator())
DM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])