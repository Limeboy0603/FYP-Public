# System Requirements and Setup Instructions

You can run this on any OS with `python3` installed. Simply install all dependencies.

## Setup

### Linux/MacOS
```sh
# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows
```sh
# Create envirionment
# Note: depending on how you installed python, your system might use the alias `python` instead of `python3`
python3 -m venv .venv
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### Building the models on your own

Alternatively, you can manually build the model on your own machine. We provide an all-in-one python script that will run everything in order.

```sh
python3 src/!FULL.py --config $PATH_TO_CONFIG_FILE 
# By default, if you do not include --config, it will use config/config_clip.yaml for windows and config/config_clip_linux.yaml on linux/MacOS

# Flags
# --capture: Capture glosses to the dataset based on whatever you set in your config file. You should include this flag to capture your own dataset.
# --extractsplit: Perform feature extraction and splitting. Use this everytime you update your dataset.
# --use: Perform practical usage on the model directly after building everything. The video source will depend on what you set in your config file.
```

## Configurations

A yaml config file can be found in `config/config_clip.yaml`. Simply follow the instructions in the yaml comments.

Note that Windows uses backslashes as opposed to forward slash. Thus, you should use `\\` instead of `/` for Windows. Alternatively, a config file for Linux and MacOS systems is provided at `config/config_clip_linux.yaml`

For a detailed documentation of configuration files, please check `config.md`
