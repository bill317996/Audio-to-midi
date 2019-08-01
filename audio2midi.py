import os, time, sys
import numpy as np
from pypianoroll import Multitrack, Track
import cfp
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

def smoothing(roll):
#     step1  Turn consecutively pitch labels into notes.
    new_map = np.zeros(roll.shape)
    min_note_frames = 3
    last_midinote = 0
    count = 0
    for i in range(len(roll)):
        midinote = np.argmax(roll[i,:])
        if midinote > 0 and midinote == last_midinote:
            count+= 1
        else:
            if count >= min_note_frames:
                new_map[i-count-1:i,last_midinote] = 1
            last_midinote = midinote
            count = 0
    note_map = new_map
    else_map = roll - note_map
#     Step2  Connect the breakpoint near the note.
    new_map = np.zeros(roll.shape)
    for i in range(len(else_map)):
        midinote = np.argmax(else_map[i,:])
        if midinote > 0:
            if note_map[i-1,midinote-1] > 0:
                new_map[i,midinote-1] = 1
                else_map[i,midinote] = 0
            elif note_map[i-1,midinote+1] > 0:
                new_map[i,midinote+1] = 1
                else_map[i,midinote] = 0
            elif (i+1)<len(else_map) and note_map[i+1,midinote-1] > 0:
                new_map[i,midinote-1] = 1
                else_map[i,midinote] = 0
            elif (i+1)<len(else_map) and note_map[i+1,midinote+1] > 0:
                new_map[i,midinote+1] = 1
                else_map[i,midinote] = 0
    note_map = note_map + new_map
#     step3  Turn vibrato pitch labels into notes.
    new_map = np.zeros(roll.shape)
    min_note_frames = 3
    last_midinote = 0
    note_list = []
    count = 0
    for i in range(len(else_map)):
        midinote = np.argmax(else_map[i,:])
        if midinote > 0 and np.abs(midinote - last_midinote) <= 1:
            last_midinote = midinote
            note_list.append(midinote)
            count+= 1
        else:
            if count >= min_note_frames:
                median_note = note_list[int((len(note_list)/2))]
                new_map[i-count-1:i,median_note] = 1
                else_map[i-count-1:i,:] = 0
            last_midinote = midinote
            note_list = []
            count = 0

    note_map = note_map + new_map
#     step4  Connect nearby notes with the same pitch label.
    last_midinote = 0
    for i in range(len(note_map)):
        midinote = np.argmax(note_map[i,:])
        if last_midinote !=0 and midinote == 0:
            if (i+1)<len(note_map) and np.argmax(note_map[i+1,:]) == last_midinote:
                note_map[i,last_midinote] = 1
            elif (i+2)<len(note_map) and np.argmax(note_map[i+2,:]) == last_midinote:
                note_map[i:i+2,last_midinote] = 1
            elif (i+3)<len(note_map) and np.argmax(note_map[i+3,:]) == last_midinote:
                note_map[i:i+3,last_midinote] = 1
            elif (i+4)<len(note_map) and np.argmax(note_map[i+4,:]) == last_midinote:
                note_map[i:i+4,last_midinote] = 1
            elif (i+5)<len(note_map) and np.argmax(note_map[i+5,:]) == last_midinote:
                note_map[i:i+5,last_midinote] = 1
            last_midinote = midinote
        else:
            last_midinote = midinote
    return note_map

class MSnet(nn.Module):
    def __init__(self):
        super(MSnet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding=2),
            nn.SELU()
            )
        self.pool1 = nn.MaxPool2d((3,1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.SELU()
            )
        self.pool2 = nn.MaxPool2d((4,1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.SELU()
            )
        self.pool3 = nn.MaxPool2d((4,1), return_indices=True)

        self.bottom = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, (6,5), padding=(0,2)),
            nn.SELU()
            )

        self.up_pool3 = nn.MaxUnpool2d((4,1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.SELU()
            )

        self.up_pool2 = nn.MaxUnpool2d((4,1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.SELU()
            )

        self.up_pool1 = nn.MaxUnpool2d((3,1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 5, padding=2),
            nn.SELU()
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        c1, ind1 = self.pool1(self.conv1(x))
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))

        bm = self.bottom(c3)

        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))

        out = torch.cat((bm, u1), dim=2)
        out = torch.squeeze(out,1)
        output = self.softmax(out)

        return output

def est(output):
    CenFreq = cfp.get_CenFreq(StartFreq=32.7, StopFreq=2093.0, NumPerOct=48)
    

    song_len = output.shape[-1]
    est_time = np.arange(song_len)*0.02322

    output = output[0,:,:]

    amax_freq = np.argmax(output, axis=0)
    est_freq = np.zeros(song_len)

    for j in range(len(est_freq)):
        if amax_freq[j] > 0.5:
            est_freq[j] = CenFreq[int(amax_freq[j])-1]
    est_arr = np.concatenate((est_time[:,None],est_freq[:,None]),axis=1)

    return est_arr


def feature_ext(fp):
    sr = 22050
    y, sr = cfp.load_audio(fp, sr=sr)
    Z, Time_arr, Freq_arr, tfrL0, tfrLF, tfrLQ = cfp.feature_extraction(y, sr, Hop=512, StartFreq=32.7, StopFreq=2093.0, NumPerOct=48)
    tfrL0 = cfp.norm(cfp.lognorm(tfrL0))[np.newaxis,:,:]
    tfrLF = cfp.norm(cfp.lognorm(tfrLF))[np.newaxis,:,:]
    tfrLQ = cfp.norm(cfp.lognorm(tfrLQ))[np.newaxis,:,:]
    W = np.concatenate((tfrL0,tfrLF,tfrLQ),axis=0)
    return W, Time_arr, Freq_arr


def seq2roll(seq):
    roll = np.zeros((len(seq),128))

    for i, item in enumerate(seq):
        if item > 0:
            midinote = int(round(cfp.hz2midi(item)))
            roll[i, midinote] = 1

    roll = smoothing(roll)
    return roll

def write_midi(filepath, pianorolls, program_nums=None, is_drums=None,
               track_names=None, velocity=100, tempo=170.0, beat_resolution=16):
    
    #if not os.path.exists(filepath):
    #    os.makedirs(filepath)

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

def main(file_path, model_path, output_path):

    Net = MSnet()
    Net.float()
    Net.cpu()
    Net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    Net.eval()

    st = time.time()
    print('     Feature extraction ...', end='\r')

    W, Time_arr, Freq_arr = feature_ext(file_path)
    W = torch.from_numpy(W).float()
    W = W[None,:]

    print('     Feature extraction ... Done. Time:', int(time.time()-st), '(s)')
    
    st = time.time()
    print('     Melody  extraction ...', end='\r')

    pred = Net(W)
    pred = pred.detach().numpy()
    est_arr = est(pred)

    print('     Melody  extraction ... Done. Time:', int(time.time()-st), '(s)')
    
    rolls = seq2roll(est_arr[:,1])
    write_midi(output_path+'.mid', np.expand_dims(rolls.astype(bool),2), program_nums=[0], is_drums=[False])

    print('Save the result in '+output_path+'.mid')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Audio to midi : Update in 20190503',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-in', action='store', dest='input_folder', default='./input/',
                        help='path to input folder')
    parser.add_argument('-out', action='store', dest='output_folder', default='./output/',
                        help='Path to output folder')
    parser.add_argument('-m', action="store_true", dest='main_melody', default=False,
                        help='Extract main-melody instead vocal-melody')

    parameters = vars(parser.parse_args(sys.argv[1:]))

    input_folder = parameters['input_folder']
    output_folder = parameters['output_folder']
    if parameters['main_melody']:
        melody_type = 'Melody'
    else:
        melody_type = 'Vocal'

    print('Audio to midi : Update in 20190503')
    print('------------')
    print('input_folder: '+input_folder)
    print('output_folder: '+output_folder)
    print('melody_type: '+melody_type)
    print('------------')

    for root, dirr, file in os.walk(input_folder):
        for filename in file:
            if '.wav' in filename:
                songname = filename.split('.wav')[0]
                fp = os.path.join(root, filename)
                mp = './model/model_'+melody_type
                op = output_folder + songname
                print ('Songname: '+songname)
                main(fp, mp, op)
                
