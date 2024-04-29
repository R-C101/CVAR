
# CVAR (Computer-Vision Assisted Referee)

The CVAR is a comprehensive application designed to revolutionize the way football (soccer) player performance is analyzed and documented. This project aims to provide a platform that meticulously monitors and tracks various aspects of the game, including player movement, ball possession, referee decisions, goalkeeper actions, and more.
## Features

- Player, referee, goalkeeper and ball tracking
- Ball possession Tracker
- Player Speed and Distance Travelled Estimator
- Camera Movement Tracker
- Ball Interpolation


## Installation

To get Started:



```bash
  git clone https://github.com/R-C101/CVAR.git
```

Get the model from Here:[Model 100 epochs](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view?usp=sharing)

The provided stub files are for the default input video.
The code mentions 
```python
read_from_stub = False
```

If you want to save computation time remove the stubs in the stubs folder and put 
```python
read_from_stub = True
```
 in the main.py file.

After the first run if you want to run the files again, the object tracking for the file will be saved in stubs and you will save a lot fo computation time for the second run.

The model I have used was trained for 100 epochs.

You can train your own model from the training directory. Just run the entire ipynb.

Put the model inside the models directory after the training.

## TODO

1. Get the coordinates automatically instead of having to assign them manually

2. Ensure that the player_id doesnt go above 30


## Screenshots

![Output Screenshot](Ouput.png)

## Usage/Examples

```python
read_from_stub = True
```




## External Links

 - [Tutorial Video](https://youtu.be/neBZ6huolkg?si=Zb_umuk6pOS72Iua)

