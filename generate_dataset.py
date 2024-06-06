from concurrent.futures import ProcessPoolExecutor
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import pyaudio
import matplotlib.pylab as plt
import matplotlib
import torchaudio
import io
import numpy as np
import torch

torchaudio.set_audio_backend("soundfile")
# torch.set_num_threads(5)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


# Define constants
LA19_DATASET_PATH = "/home/hungdx/Datasets/ASVspoof2021_DF_eval"
LA19_TRAIN_PROTOCOL_PATH = "/datad/hungdx/KDW2V-AASISTL/protocols/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt"
SAMPLE_RATE = 16000
CUT_SIZE = 16000
DURATION_CUT = 1
DESTINATION_PATH = f"/datad/hungdx/Rawformer-implementation-anti-spoofing/datasets/ASVspoof2021_DF_eval_{DURATION_CUT}s"
DESTINATION_LA19_TRAIN_PROTOCOL_PATH = f"/datad/hungdx/Rawformer-implementation-anti-spoofing/datasets/protocols/ASVspoof2021.DF.cm.eval.trl_{DURATION_CUT}s.txt"

# Setup destination directories
if os.path.exists(DESTINATION_LA19_TRAIN_PROTOCOL_PATH):
    os.remove(DESTINATION_LA19_TRAIN_PROTOCOL_PATH)
if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH, exist_ok=True)

THRESHOLD = 0.6  # Threshold for VAD to determine speech

# Function to process each line of the protocol file


def process_line(line):
    line = line.strip().split()
    file = line[0]
    file_path = os.path.join(LA19_DATASET_PATH, file + ".flac")
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.squeeze(0)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"File {file}.flac sample rate is not {SAMPLE_RATE}")

    duration = len(waveform) / SAMPLE_RATE
    number_of_cuts = len(waveform) // CUT_SIZE
    residual = len(waveform) % CUT_SIZE

    results = []
    for i in range(number_of_cuts):
        cut_waveform = waveform[i * CUT_SIZE: (i + 1) * CUT_SIZE]
        new_confidence = vad_model(cut_waveform, SAMPLE_RATE).item()
        suffix = "" if new_confidence > THRESHOLD else "_no_speech"
        file_name = f"{file}_{i}{suffix}.flac"
        torchaudio.save(os.path.join(DESTINATION_PATH, file_name),
                        cut_waveform.unsqueeze(0), SAMPLE_RATE)
        results.append(f"{file_name.split('.flac')[0]}\n")

    # Process residual part if necessary
    if residual > 0 and len(waveform[-residual:]) >= 512:
        process_residual_part(waveform, residual,
                              number_of_cuts, line, results)

    return results


def process_residual_part(waveform, residual, count, line, results):
    cut_waveform = waveform[-residual:]
    new_confidence = vad_model(cut_waveform, SAMPLE_RATE).item()
    suffix = "" if new_confidence > THRESHOLD else "_no_speech"
    file_name = f"{line[0]}_{count}_residual{suffix}.flac"
    torchaudio.save(os.path.join(DESTINATION_PATH, file_name),
                    cut_waveform.unsqueeze(0), SAMPLE_RATE)
    results.append(f"{file_name.split('.flac')[0]}\n")


# Reading the lines in advance to prevent file access issues in threads
with open(LA19_TRAIN_PROTOCOL_PATH) as file:
    lines = file.readlines()

# futures = []
# for line in tqdm(lines, total=len(lines)):
#     futures.append(process_line(line))


# with open(DESTINATION_LA19_TRAIN_PROTOCOL_PATH, 'a') as f:
#     for result in futures:
#         f.writelines(result)

# Use a ThreadPoolExecutor or ProcessPoolExecutor based on the task type
# or ThreadPoolExecutor(max_workers=number_of_workers)
executor = ProcessPoolExecutor()
futures = [executor.submit(process_line, line) for line in lines]

# Collect results
with open(DESTINATION_LA19_TRAIN_PROTOCOL_PATH, 'a') as f:
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        f.writelines(result)
