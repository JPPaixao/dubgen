# *dubgen*
##Sampler and MIDI Arrangement Generator

*dubgen* creates monophonic samples from given sounds, modifies MIDI compositions, and then choses a sequence of samples to generate an arrangement (new MIDI + samples).

Created samples can also be modulated by a Synthesizer.

How it works:
1. Samples: In the *".../dubgen/user/test_sounds/"* folder, add sounds to create a library of samples. We strongly recommend inserting more than 20 sounds.

2. MIDI: In the *".../dubgen/user/midi/"* folder, insert three MIDI files, one for each instrument.

3. (Optional) You can train your own "Instrument Classification Model". For this, add sounds in each Instrument Category Folder in *".../dubgen/user/train_sounds/"*. We recommend inserting more than 20 sounds per Instrument.

**Install Guide**

This app was only tested on Windows 10, and requires Python3 (Download here: [python.org](https://www.python.org/downloads/) ).

After Installing Python3, clone this repository into your device:

`git clone https://github.com/JPPaixao/dubgen`

Note: To run this command, open a Git Bash Terminal on your desired directory. (Right Click -> Git Bash Here)

After that, import the required Python Packages. Please open the Git Bash Terminal in the *"/Program/"* Folder, or run this command on the already open Terminal:

`cd Program`

Then install the Packages on *"Requirements.txt"*:

`pip3 install -r requirements.txt`

#Run "dubgen"
To run the program, open a Git Bash Terminal on the Program folder, and run this command:

`python3 dubgen.py`
