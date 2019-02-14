
import torch
import argparse
import numpy as np
from MSnet.MelodyExtraction import MeExt
import os
# import matplotlib.pyplot as plt

import glob, sys
import subprocess as subp
from pypianoroll import Multitrack, Track
from scipy.signal import medfilt
import math
def medianFilter(ext_melody, filter_size=9, unit=32):
    # unit = bpm*midi_resolution/60 #midi_resolution and bpm can be defined by yourself
    # unit = 32
    base_fre = 27.5
    size = 44100/256
    output = np.zeros((round(len(ext_melody)*unit/size),128))
    for j,fre in enumerate(ext_melody):
        if float(fre)>0:
            try:
                time = round(j*unit/size)
                note = float(fre)/base_fre
                note = round(12*math.log(note,2))
                output[time][note+21] = 1
            except: pass
    for j in range(128):
        output[:,j] = medfilt(output[:,j],filter_size) #filter_size: decided by yourself, in this default setting is 7
    return output

def write_midi(filepath, pianorolls, program_nums=[0], is_drums=[False],
               track_names=None, velocity=100, tempo=120.0, beat_resolution=24):
    
    if not os.path.exists(filepath):
       os.makedirs(filepath)

    if not np.issubdtype(pianorolls.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")
    if isinstance(program_nums, int):
        program_nums = [program_nums]
    if isinstance(is_drums, int):
        is_drums = [is_drums]
    if pianorolls.shape[2] != len(program_nums):
        raise ValueError("`pianorolls` and `program_nums` must have the same"
                         "length")
    if pianorolls.shape[2] != len(is_drums):
        raise ValueError("`pianorolls` and `is_drums` must have the same"
                         "length")
    if program_nums is None:
        program_nums = [0] * len(pianorolls)
    if is_drums is None:
        is_drums = [False] * len(pianorolls)

    multitrack = Multitrack(beat_resolution=beat_resolution, tempo=tempo)
    for idx in range(pianorolls.shape[2]):
        #plt.subplot(10,1,idx+1)
        #plt.imshow(pianorolls[..., idx].T,cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
        if track_names is None:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx])
        else:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx], track_names[idx])
        multitrack.append_track(track)
    #plt.savefig(cf.MP3Name)
    multitrack.write(filepath)

def note2midi(rolls, output_path, time_unit=42):
    # print(content[-1][1]+content[-1][2])
    # time_length = (content[-1][1] + content[-1][2])*time_unit +1

    # rolls = np.zeros((int(time_length),128))
    # for i,(note,start,dura) in enumerate(content):
    #     s_time = int(float(start)*time_unit)
    #     d_time = int(float(dura)*time_unit)
    #     rolls[s_time:s_time+d_time,int(float(note))]=1
    write_midi(output_path+'.mid', np.expand_dims(rolls.astype(bool),2), program_nums=[0], is_drums=[False])
    print('Save the result in '+output_path+'.mid')

def main(filepath, model_type, output_dir, gpu_index, mode):
    songname = filepath.split('/')[-1].split('.')[0]
    model_path = './MSnet/pretrain_model/MSnet_'+str(model_type)

    if gpu_index is not None:
        with torch.cuda.device(gpu_index):
            est_arr = MeExt(filepath, model_type=model_type, model_path=model_path, GPU=True, mode=mode)
    else:
        est_arr = MeExt(filepath, model_type=model_type, model_path=model_path, GPU=False, mode=mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print('Save the result in '+output_dir+'/'+songname+'.txt')
    # np.savetxt(output_dir+'/'+songname+'.raw.melody', est_arr[:,1])
    return est_arr[:,1]

    # call(['python', 'Note_tracking.py', output_dir+'/'+songname+'.raw.melody', output_dir])
    
def parser():
    
    p = argparse.ArgumentParser()

    p.add_argument('-in', '--input_dir',
                    help='Path to input folder (default: %(default)s',
                    type=str, default='./input/')
    p.add_argument('-t', '--model_type',
                    help='Model type: vocal or melody (default: %(default)s',
                    type=str, default='vocal')
    p.add_argument('-gpu', '--gpu_index',
                    help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s',
                    default=None)
    p.add_argument('-o', '--output_dir',
                    help='Path to output folder (default: %(default)s',
                    type=str, default='./output/')
    p.add_argument('-m', '--mode', 
                    help='The mode of CFP: std and fast (default: %(default)s',
                    type=str, default='std')
    return p.parse_args()
    
if __name__ == '__main__':
    args = parser()
    for root, dirr, file in os.walk(args.input_dir):
        for filename in file:
            if '.wav' or '.mp3' in filename:
                filepath = os.path.join(root, filename)
                songname = filepath.split('/')[-1].split('.')[0]
                print('=====================')
                ext_melody = main(filepath, args.model_type, args.output_dir, args.gpu_index, args.mode)
                pruned_note =  medianFilter(ext_melody)
                write_midi(args.output_dir+'/'+songname+'.mid', np.expand_dims(pruned_note.astype(bool),2))
                print('Save the result in '+args.output_dir+'/'+songname+'.mid')


