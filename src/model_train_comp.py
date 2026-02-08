import os
os.environ['KERAS_BACKEND'] = "torch"
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import keras
import numpy as np

from config import config_parser
from mp_util_legacy import preprocess_keypoints_multiple, get_raw_coords
import model_test

np.random.seed(42)

"""
Model architecture based on the paper:
Bhadouria, A., Bindal, P., Khare, N., Singh, D., & Verma, A. (2024). 
LSTM-Based Recognition of Sign Language. 
IC3-2024: Proceedings of the 2024 Sixteenth International Conference on Contemporary Computing, 508-514. 
https://doi.org/10.1145/3675888.3676105

Structure:
LSTM(64) -> LSTM(128) -> LSTM(64) -> Dense(64) -> Dense(32) -> Dense(Y_shape)

Main differences:
- Trained for 2000 epochs
- Hands only
- They did not use any dropout layers
- They did not use any data augmentation
- No callbacks were used

Changes made for fair comparisons:
- Trained for 128 epochs
- Use data augmentation (so input data for both models are the same)
- With callbacks (because original model is designed to overfit, but we don't)
"""

class KeypointDataGenerator(keras.utils.Sequence):
    def __init__(self, keypoint_path, seq_max_len, batch_size=32, shuffle=False, transform=False):
        self.keypoint_path = keypoint_path
        self.seq_max_len = seq_max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        
        self.file_list = []
        self.labels = []

        self.all_labels = sorted(os.listdir(keypoint_path))
        for label in self.all_labels:
            for file in os.listdir(os.path.join(keypoint_path, label)):
                self.file_list.append(file)
                self.labels.append(label)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))
    
    def preprocess_keypoints(self, keypoints):
        angle = np.random.randint(-20, 20)
        tx = np.random.uniform(-0.4, 0.4)
        ty = np.random.uniform(-0.4, 0.4)
        scale = np.random.uniform(0.6, 1.2)
        return preprocess_keypoints_multiple(keypoints, angle=angle, tx=tx, ty=ty, scale=scale)

    def __data_generation(self, indexes):
        sequences = []
        labels = []

        for index in indexes:
            kp_file_name = self.file_list[index]
            label = self.labels[index]
            sequences.append(np.load(os.path.join(self.keypoint_path, label, kp_file_name), mmap_mode="r"))
            labels.append(label)

        for i in range(len(sequences)):
            local_sequence = sequences[i].copy()
            if self.transform:
                local_sequence = self.preprocess_keypoints(local_sequence)
            local_sequence = local_sequence.reshape(self.seq_max_len, -1)
            result_sequence = []
            # _, _, left_hand_coord, right_hand_coords = get_raw_coords(local_sequence)
            # local_sequence = np.concatenate([left_hand_coord, right_hand_coords], axis=1)
            # print(local_sequence.shape)
            # sequences[i] = local_sequence
            for j in range(self.seq_max_len):
                _, _, left_hand_coords, right_hand_coords = get_raw_coords(local_sequence[j])
                concat = np.concatenate([left_hand_coords, right_hand_coords], axis=0)
                concat = concat.reshape(-1)
                result_sequence.append(concat)
            sequences[i] = np.array(result_sequence)

        X = np.array(sequences)
        Y = np.array([self.all_labels.index(label) for label in labels])
        return X, Y
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = self.__data_generation(indexes)
        return X, Y
    
def main(config_path: str):
    config = config_parser(config_path)

    training_generator = KeypointDataGenerator(
        config.paths.split_train, 
        seq_max_len=config.sequence.frame, 
        batch_size=64, 
        shuffle=True, 
        transform=True
        # transform=True
    )
    validation_generator = KeypointDataGenerator(
        config.paths.split_val, 
        seq_max_len=config.sequence.frame, 
        batch_size=64, 
        shuffle=False, 
        transform=False
    )

    first_batch = training_generator.__getitem__(0)
    X_shape = first_batch[0].shape[1:]
    Y_shape = len(training_generator.all_labels)
    del first_batch

    model = Sequential()
    
    model.add(LSTM(64, input_shape=X_shape, return_sequences=True, use_cudnn=False))
    model.add(LSTM(128, return_sequences=True, use_cudnn=False))
    model.add(LSTM(64, return_sequences=False, use_cudnn=False))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(Y_shape, activation="softmax"))

    model.summary()
    """
    Total params: 204,383 (798.37 KB)
    Trainable params: 204,383 (798.37 KB)
    Non-trainable params: 0 (0.00 B)
    """
    # model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])


    model.fit(
        training_generator,
        epochs=128,
        # epochs=64,
        shuffle=True,
        validation_data=validation_generator,
        callbacks=[
            keras.callbacks.ModelCheckpoint(config.paths.model_checkpoint, monitor="val_loss", save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        ]
    )

    # modify the model path's file name to comp.keras
    model_path = config.paths.model
    model_path = model_path.replace(".keras", "_comp.keras")

    model.save(model_path)
    model_test.main(config_path, model_path)

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml")