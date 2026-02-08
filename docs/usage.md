# Usage

**Note:** Remember to activate the python environment first!

### Linux/MacOS
```sh
source .venv/bin/activate
```

### Windows
```sh
.venv\Scripts\Activate
```

## Capturing Dataset

You should change the following settings in your config file first.
- Change `capture.source` to a camera instead of a video.
- Make sure that `capture.resolution` is correct as well.
- Set the list of glosses you want to capture in `dictionary`. If a recorded gloss in `dictionary` is already in the dataset, the capture program will re-record the glosses.
- Set the number of frames per clip and number of clips you want to capture in `sequence`

You should also make sure that the glosses you intend to capture should also appear in sentence samples in `llm_samples`

Once you have set up everything, simply run the following script to start capturing your own dataset.

```sh
python3 src/cam_capture.py
```

Next, you'll need to:
- Perform feature extraction
- Split dataset into training, testing and validation sets
- Re-train model

We provide an all-in-one script for the above steps

```sh
python3 src/!FULL.py --extractsplit
```

## Using the Application
Simply run

```sh
python3 src/app.py
```

### Button usages

- `Start`: Starts the camera for isolated sign language classification.
- `Stop`: Stops the camera for isolated sign language classification.
- `Predict Natural Sentence`: Predicts a sentence based on the sequence of recorded glosses.
- `Clear`: Clear all cached sequence of glosses and predicted sentence.
- `Kill Thread`: Kills the thread for isolated sign language classification. 
- `Restart`: Restarts the thread. This will kill the thread first before restarting.

## Other Scripts

We do NOT recommend that you run other scripts unless you know what you are doing. Here are the usage of other scripts.

### Experiment scripts

Scripts designed for experimenting. Includes:
- `src/experiments_kp_Z_test.py` for verifying Z coordinates scaling equation.
- `src/experiments_model_selection.ipynb` for model selection.
- `src/model_use_sliding.py` for testing with sliding window approach.
- `src/model_train_comp.py` for comparing with existing solution.

### Utilities

Designed to provide functions to other scripts. Includes:
- `src/config.py`
- `src/llm_util.py`
- `src/mp_util_legacy.py`

### Functional Scripts

Contains a `main` function that will be called by `src/!FULL.py`. 

They can also be executed independently as well.

Includes:
- `src/feat_extraction.py` if `--extractsplit` is included.
- `src/dataset_split.py` if `--extractsplit` is included.
- `src/model_train.py`
- `src/model_test.py`
- `src/model_use_isolated.py`

### Other Scripts

Contains a `main` function that is **NOT** called by any other scripts. Includes:
- `src/full_clip_builder.py` that builds `dataset/full_test/final.avi`