import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import keras
from config import config_parser
from mp_util_legacy import get_raw_coords

def main(config_path: str, model_path_override: str = None):
    # config = config_parser("config/config_clip.yaml")
    config = config_parser(config_path)
    model_path = model_path_override if model_path_override else config.paths.model
    model = keras.models.load_model(model_path)

    # validation set
    pred_dict = {}
    dictionary = sorted(os.listdir(config.paths.split_val))
    for label in dictionary:
        preds = []
        label_path = os.path.join(config.paths.split_val, label)
        for file in os.listdir(label_path):
            kp = np.load(os.path.join(label_path, file))
            pred_kp = kp.reshape(1, kp.shape[0], -1)
            if model_path_override:
                new_kp = []
                for seq in range(pred_kp.shape[1]):
                    local_kp = pred_kp[:, seq, :][0]
                    _, _, left_hand_coords, right_hand_coords = get_raw_coords(local_kp)
                    local_kp = np.concatenate([left_hand_coords, right_hand_coords], axis=0)
                    local_kp = local_kp.reshape(-1)
                    new_kp.append(local_kp)
                pred_kp = np.array(new_kp).reshape(1, kp.shape[0], -1)
            pred = model.predict(pred_kp)
            pred = np.argmax(pred)
            pred = dictionary[pred]
            preds.append(pred)
        pred_dict[label] = preds
    print("Using path:", config.paths.split_val)
    print(pred_dict)

    correct = 0
    total = 0
    for key in pred_dict:
        for pred in pred_dict[key]:
            if pred == key:
                correct += 1
            total += 1
    validation_accuracy = correct / total
    # print("Accuracy: ", validation_accuracy)

    # testing set
    pred_dict = {}
    dictionary = sorted(os.listdir(config.paths.split_test))
    for label in dictionary:
        preds = []
        label_path = os.path.join(config.paths.split_test, label)
        for file in os.listdir(label_path):
            kp = np.load(os.path.join(label_path, file))
            pred_kp = kp.reshape(1, kp.shape[0], -1)
            if model_path_override:
                new_kp = []
                for seq in range(pred_kp.shape[1]):
                    local_kp = pred_kp[:, seq, :][0]
                    _, _, left_hand_coords, right_hand_coords = get_raw_coords(local_kp)
                    local_kp = np.concatenate([left_hand_coords, right_hand_coords], axis=0)
                    local_kp = local_kp.reshape(-1)
                    new_kp.append(local_kp)
                pred_kp = np.array(new_kp).reshape(1, kp.shape[0], -1)
            pred = model.predict(pred_kp)
            pred = np.argmax(pred)
            pred = dictionary[pred]
            preds.append(pred)
        pred_dict[label] = preds
    print("Using path:", config.paths.split_test)
    print(pred_dict)

    # compute accuracy
    correct = 0
    total = 0
    for key in pred_dict:
        for pred in pred_dict[key]:
            if pred == key:
                correct += 1
            total += 1
    test_accuracy = correct / total
    # print("Accuracy: ", correct / total)
    print("Validation accuracy: ", validation_accuracy)
    print("Testing accuracy: ", test_accuracy)

if __name__ == "__main__":
    main("config/config_clip.yaml" if os.name == "nt" else "config/config_clip_linux.yaml", "models/final_full_comp.keras")