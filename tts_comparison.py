"""
Determine characteristics of text to speech data,
and determine if optimal.
"""

# Imports
import numpy as np
import soundfile as sf

# Definitions
char_per_sec_optimal = 20  # optimal speed
silence_threshold = 0.01  # threshold to be considered silence

sound_filename = input("Enter the (.wav) file of sound: ")  # get sound
sound, samp_rate = sf.read(sound_filename)  # open sound
len_sound = len(sound)  # get length of sound
time_sound = len_sound / samp_rate  # get duration of sound
print("The sound is {0} seconds long.".format(time_sound))

text_filename = input("Enter the (.txt) file of text: ")  # get text
with open(text_filename, "r") as f:
    text = f.read()  # read file
len_text = len(text)  # get text length
print("There are {0} characters in the text file.".format(len_text))

# Characters optimal
char_per_sec = len_text / time_sound  # get character speed
print("The characters per second speed is {0} char/s.".format(char_per_sec))
print("The optimal speed is {0} char/s.".format(char_per_sec_optimal))
pad_char = (char_per_sec_optimal - char_per_sec) * time_sound  # get number of characters to add or remove
if pad_char > 0:  # if add characters
    print("You should add {0} characters to be optimal.".format(pad_char))
elif pad_char < 0:  # if remove characters
    print("You should remove {0} characters to be optimal.".format(-pad_char))

# Silence ratio
try:  # get number of streams
    streams = len(sound[0])
    silence_data = np.amax(np.abs(sound), axis=-1)  # get max value in streams
except Exception:
    silence_data = np.abs(sound)  # only one stream.
len_silence_data = len(silence_data)  # get length

noise_count = 0
silence_count = 0
for i_sound in range(len_silence_data):  # cycle through sound
    if silence_data[i_sound] < silence_threshold:  # if silence
        silence_count += 1  # increment silence count
    else:  # if noise
        noise_count += 1  # increment noise count

percent_noise = noise_count / len_silence_data
percent_silence = silence_count / len_silence_data
print("Sound file contains {0}% of noise above {1} and {2}% silence.".format(percent_noise,
                                                                             silence_threshold,
                                                                             percent_silence))

# Silence ratio (Text)
whitespace_count = 0
character_count = 0
for c in text:  # cycle through characters
    if c.isspace():
        whitespace_count += 1  # increment whitespace count
    else:
        character_count += 1  # increment character count

percent_whitespace = whitespace_count / len_text  # get percent whitespaces
percent_character = character_count / len_text  # get percent characters
print("Text file contains {0}% characters and {1}% whitespaces.".format(percent_character, percent_whitespace))

optimal_len_text = len_text + pad_char
optimal_characters = percent_noise * optimal_len_text  # get optimal characters
optimal_whitespaces = percent_silence * optimal_len_text  # get optimal white spaces

diff_characters = optimal_characters - character_count  # get difference
diff_whitespaces = optimal_whitespaces - whitespace_count  # get difference

# Recommend padding
if diff_characters > 0:
    print("You should add {0} alpha-numeric characters.".format(diff_characters))
elif diff_characters < 0:
    print("You should remove {0} alpha-numeric characters.".format(abs(diff_characters)))
else:
    print("You should not change the number of alpha-numeric characters.")

if diff_whitespaces > 0:
    print("You should add {0} whitespaces.".format(diff_whitespaces))
elif diff_whitespaces < 0:
    print("You should remove {0} whitespaces.".format(abs(diff_whitespaces)))
else:
    print("You should not change the number of whitespaces.")
