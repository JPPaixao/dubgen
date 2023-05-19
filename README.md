# *dubgen*
Sampler and MIDI Arrangement Generator

*dubgen* creates monophonic samples from given sounds, modifies MIDI compositions, and then choses a sequence of samples to generate an arrangement (new MIDI + samples).

Created samples can also be modulated by a Synthesizer

How it works:
1. Samples: In the *".../dubgen/user/test_sounds/"* folder, add sounds to create a library of samples. We recommend inserting more than 20 sounds.

2. MIDI: In the *".../dubgen/user/midi/"* folder, insert three MIDI files, one for each instrument.

3. (Optional) You can train your own "Instrument Classification Model". For this, add sounds in each Instrument Category Folder in *".../dubgen/user/train_sounds/"*



`git clone https://github.com/JPPaixao/dubgen`

`pip3 install -r requirements.txt`

`cd Program`

`python3 dubgen.py`
