import io, os, json
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from google.cloud import storage
from utils import TextTransform       # Comment this for engine inference

class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, hop_length=160):
        super(LogMelSpec, self).__init__()
        self.transform = transforms.MelSpectrogram(sample_rate=sample_rate, 
                                                   n_mels=n_mels,
                                                   hop_length=hop_length)

    def forward(self, x):
        return self.transform(x)  # mel spectrogram

# MelSpec Feature Extraction for ASR Engine Inference
def get_featurizer(sample_rate=16000, n_mels=128, hop_length=160):
    return LogMelSpec(sample_rate=sample_rate, 
                      n_mels=n_mels,
                      hop_length=hop_length)

# Custom Dataset Class
class CustomAudioDataset(Dataset):
    def __init__(self, json_path, bucket_name, transform=None, log_ex=True, valid=False):
        print(f'Loading json data from {json_path}')
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.bucket_name = bucket_name
        self.client = storage.Client() # uses your Colab auth

        # Initialize TextProcess for text processing    
        self.text_process = TextTransform()                 
        self.log_ex = log_ex

        base_transforms = [transforms.MelSpectrogram(sample_rate=16000, n_mels=128, hop_length=160)]
        if not valid:
            base_transforms += [
                transforms.FrequencyMasking(freq_mask_param=15),
                transforms.TimeMasking(time_mask_param=27)
            ]
        self.audio_transforms = nn.Sequential(*base_transforms)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        fname = os.path.basename(item['key'])  # strip off the C:\…\ prefix
        blob_path = f"converted_clips/clips/{fname}"

        try:
            # dounload bytes
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            audio_bytes = blob.download_as_bytes()
            # read with soundfile from memory
            bio = io.BytesIO(audio_bytes)
            data, sample_rate = sf.read(bio)       
            waveform = torch.from_numpy(data).float().unsqueeze(0)
            #process transcript
            utterance = item['text'].lower()               
            label = self.text_process.text_to_int(utterance)
            # compute featues
            spectrogram = self.audio_transforms(waveform)   # (channel, feature, time)
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)

            if spec_len < label_len:
                raise ValueError(f"spec shorter than label for {fname}")
            if spectrogram.shape[0] > 1:
                raise ValueError(f"dual channel, skipping {fname}")
            if spectrogram.shape[2] > 2650*4:
                raise ValueError(f"spectrogram too big: {spectrogram.shape[2]}")
            if label_len == 0:
                raise ValueError(f"empty label for {fname}")

            return spectrogram, label, spec_len, label_len

        except Exception as e:
            # Print for debugging if letters in sentences have transform issues
            if self.log_ex:
                print(f"Skip {fname} because {e}")
            #fallback to a different index
            if (len(self.data) == 1:
                raise
            new_idx = idx - 1 if idx > 0 else idx + 1
            return self.__getitem__(new_idx)
        
    def describe(self):
        return self.data.describe()
    

# Lightning Data Module
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_json, test_json, num_workers=2):
        super().__init__()
        self.batch_size = batch_size
        self.train_json = train_json
        self.test_json = test_json
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = CustomAudioDataset(self.train_json,
                                                bucket_name="realtime-asr-project",
                                                valid=False)
        self.test_dataset = CustomAudioDataset(self.test_json, 
                                               bucket_name="realtime-asr-project",
                                               valid=True)
        
    def data_processing(self, data):
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for (spectrogram, label, input_length, label_length) in data:
            if spectrogram is None:
                continue

            spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
            labels.append(torch.Tensor(label))
            input_lengths.append(input_length)
            label_lengths.append(label_length)

        # NOTE: https://www.geeksforgeeks.org/how-do-you-handle-sequence-padding-and-packing-in-pytorch-for-rnns/
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          collate_fn=lambda x: self.data_processing(x), 
                          num_workers=self.num_workers, 
                          pin_memory=True)      # Optimizes data-transfer speed

    def val_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          collate_fn=lambda x: self.data_processing(x), 
                          num_workers=self.num_workers, 
                          pin_memory=True)
