import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

callDataFrame = pd.read_csv("/content/callData/outputData.csv")

val_dataframe = callDataFrame.sample(frac=0.2, random_state=1337)
train_dataframe = callDataFrame.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("Scammer")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
#from tensorflow.keras.layers.experimental.preprocessing import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

# Numerical features
TotalCalls = keras.Input(shape=(1,), name="TotalCalls")
UniqueNums = keras.Input(shape=(1,), name="UniqueNums")
AvgCalls = keras.Input(shape=(1,), name="AvgCalls")

all_inputs = [
    TotalCalls,
    UniqueNums,
    AvgCalls,
]

# Numerical features
TotalCalls_encoded = encode_numerical_feature(TotalCalls, "TotalCalls", train_ds)
UniqueNums_encoded = encode_numerical_feature(UniqueNums, "UniqueNums", train_ds)
AvgCalls_encoded = encode_numerical_feature(AvgCalls, "AvgCalls", train_ds)

all_features = layers.concatenate(
    [
        TotalCalls_encoded,
        UniqueNums_encoded,
        AvgCalls_encoded,
    ]
)

x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# `rankdir='LR'` is to make the graph horizontal.
keras.utils.plot_model(model, show_shapes=True)

model.fit(train_ds, epochs=8, validation_data=val_ds)

TotalCalls = 76
UniqueNums = 75
AvgCalls = TotalCalls / UniqueNums

CallSample= {
    "TotalCalls": TotalCalls,
    "UniqueNums": UniqueNums,
    "AvgCalls": AvgCalls,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in CallSample.items()}
predictions = model.predict(input_dict)

print(
    "Total calls = %i , UniqueNums = %i \n" 
    "This particular call pattern had a %.1f percent probability "
    "of being suspicious" % (TotalCalls, UniqueNums, 100 * predictions[0][0],)
)