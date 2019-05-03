# Audio-to-midi
An application of melody extraction.
### Core algoithm 
(MSnet): https://github.com/bill317996/Melody-extraction-with-melodic-segnet

### Dependencies

Requires following packages:

- python 3.6
- pytorch 1.0.0
- numpy
- pysoundfile
- scipy
- math
- pypianoroll

### Usage
Put your audio files in "./input/" folder and run
```
python audio2midi.py
```
or
```
python3 audio2midi.py
```
Results will save in './output/'
#### audio2midi.py
```
usage: audio2midi.py [-h] [-in INPUT_FOLDER] [-out OUTPUT_FOLDER] [-m]

Audio to midi : Update in 20190503

optional arguments:
  -h, --help          show this help message and exit
  -in INPUT_FOLDER    path to input folder (default: ./input/)
  -out OUTPUT_FOLDER  Path to output folder (default: ./output/)
  -m                  Extract main-melody instead vocal-melody (default: False)
```
