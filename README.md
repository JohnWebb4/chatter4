# Chatter4
The goal of this repo is text to speech (TTS) prediction using Tensorflow, and
features one network and one characterization program.

## Characterization
[tts_comparison.py](tts_comparison.py) allows you to compare a given .wav file
with a string of the intended text. The sound file is then analyzed for optimal training quality
This includes:
- Are there large amounts of unaccounted for silence?
- The program assume a person speaks at 20 characters per second. Test if your file meets this.

### Running TTS Comparison
``` batch
python tts_comparision.py
```

The comparison then provides
- Analysis of silence compared to whitespace(s) in text file
- Recomendation for total length of text file and ratio of white space

## Training and Prediction
[chatter_4.py](chatter_4.py) is a Recurrent Neural Network trained under Stoichastic Gradient Descent (SGD) with
cross-entropy and a Root-Mean Square Optimizer.

### Running Chatter4
``` batch
python chatter_4.py
```

After providing a filepath to an appropriate WAV file and ASCII text file. the network will be trained.
After training, it will enter a prediction loop, and generate a WAV sound file based of your input text

## Language
Python 3

## License
[MIT](LICENSE)
