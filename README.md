# Audio-to-midi
An application of melody extraction.
https://github.com/bill317996/Melody-extraction-with-melodic-segnet

### Dependencies

Requires following packages:

- python 3.6
- pytorch 1.0.0
- numpy
- pysoundfile
- scipy
- math

### Usage
Put your audio files in "./input/" folder and run
```
python main.py
```
or
```
python3 main.py
```
Results will save in './output/'
#### audio2midi.py
```
usage: audio2midi.py [-h] [-in INPUT_DIR] [-t MODEL_TYPE] [-gpu GPU_INDEX]
                     [-o OUTPUT_DIR] [-m MODE]

optional arguments:
  -h, --help            show this help message and exit
  -in INPUT_DIR, --input_dir INPUT_DIR
                        Path to input folder (default: ./input/)
  -t MODEL_TYPE, --model_type MODEL_TYPE
                        Model type: vocal or melody (default: vocal)
  -gpu GPU_INDEX, --gpu_index GPU_INDEX
                        Assign a gpu index for processing. It will run with
                        cpu if None. (default: None)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to output folder (default: ./output/)
  -m MODE, --mode MODE  The mode of CFP: std and fast (default: std)
```
