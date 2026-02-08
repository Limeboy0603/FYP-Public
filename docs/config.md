# Configurations

This file details the instructions of configuring the software. There are 2 configuration files provided:

- Windows: `config/config_clip.yaml`
- Linux/MacOS: `config/config_clip_linux.yaml`

The main difference of these 2 config files lies in the paths, as Windows uses `\` instead of `/`

Examples of each config are written in code blocks.

## `capture`

Contains settings regarding the video input, which will be used by OpenCV's VideoCapture. 

### `capture.source`

Sets the source of the video input. This can either be:

- The ID of your camera. Check your connected cameras using the command below:
    - Windows: `Get-CimInstance Win32_PnPEntity | ? { $_.service -eq "usbvideo" } | Select-Object -Property PNPDeviceID, Name`
    - Linux: `ls /dev/video*`
    - MacOS: `system_profiler SPCameraDataType`
- A path to a video

```yaml
  # Using camera
  source: 0

  # Using video
  source: "dataset\\full_test\\final.avi"
```

### `capture.resolution`

Sets the resolution of the video input. It should be in the form of `{width}x{height}`

```yaml
  resolution: 1920x1080
```

## `paths`

Contains settings to directories and files necessary for the system. 

By default, if the directory does not exist, it will automatically be created.

### `paths.model`

Sets the path of the model output after training and the model to use for the application.

```yaml
  model: "models/final_full.keras"
```

### `paths.keypoints`

Sets the output directory of the extracted features.

```yaml
  keypoints: "dataset\\keypoints"
```

### `paths.model_checkpoint`

Sets the output file location of the model checkpoint during training.

```yaml
  model_checkpoint: "model_checkpoint/model_checkpoint.keras"
```

### `paths.split`

Sets the output directory of extracted features after performing train-test-val split.

```yaml
  split: "dataset\\split"
```

### `paths.llm`

Sets the directory name to store the language model and it's training result.

```yaml
  llm: "llm"
```

## `dictionary`

A list that defines the glosses to capture using `src/cam_capture.py`

```yaml
dictionary:
  # The glosses for self introduction
  - "#BLANK"
  - Hello
  - Me
  - English
  - Name

  # Letters from the alphabet
  - A
  - B
  - ...
```

## `sequence`

Defines the amount of data in the dataset.

### `sequence.frame`

Defines how long each clip in the dataset is in the unit of frames.

```yaml
  frame: 30
```

### `sequence.count`

Defines how many clips each gloss label contains.

```yaml
  count: 100
```

## `llm_samples`

Contains the samples to train the BART model.

It should be a list of objects, each object containing 2 attributes
- `hksl`: Sentence in HKSL grammar
- `natural`: Natural sentence to be translated from `hksl`

```yaml
llm_samples:
  - hksl: Hello Me English Name A B C D
    natural: Hello, my English name is ABCD.
```
