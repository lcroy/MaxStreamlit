# Audio Classification of urbansound8k
# Author: Chen Li
# 3 classes: Chen, Jinha, Hahyeon
# Architecture: LeNet5

import os
import torch
import torchaudio # not supported on windows
import pandas as pd
from torch.utils.data import Dataset

class SoundDataset(Dataset):
#obtain model from the data
# Argument List
#   path to the csv file
#   path to the audio files
#   list of folders to use in the data
    def __init__(self, csv_path,file_path, folderList):
        csvData = pd.read_csv(csv_path)
        #initialize lists to hold file name, labels and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        #loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csvData)):
            if csvData.iloc[i,5] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 6])
                self.folders.append(csvData.iloc[i, 5])

        self.file_path = file_path
        #self.mixer = torch.mean()
        #self.mixer = torchaudio.transforms.DownmixMono() # audio uses two
        self.folerList = folderList

    def __getitem__(self, index):
        #format the file path and load the file
        path = os.path.join(self.file_path, "fold" + str(self.folders[index]), self.file_names[index])
        sound = torchaudio.load(path,out=None, normalization=True)
        #load returns a tensor with the sound model and the sampling frequency
        soundData = torch.mean(sound[0],dim=0).unsqueeze(0).permute(1,0)
        #soundData = self.mixer(sound[0])
        # downsample the audio to ~8kHz
        tempData = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]
        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        # take every fifth sample of soundData
        soundFormatted[:32000] = soundData[::5]
        soundFormatted = soundFormatted.permute(1,0)

        return soundFormatted, self.labels[index]

    def __len__(self):

        return len(self.file_names)




