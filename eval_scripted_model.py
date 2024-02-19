import torch
import os
import librosa

model_scripted_ckpt = "/datab/hungdx/KDW2V-AASISTL/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_3_v10_best63.pt"
test_dir = "/datab/hungdx/KDW2V-AASISTL/test_mobile_samples/RandomVideos"
score = "/datab/hungdx/KDW2V-AASISTL/test_mobile_samples/RandomVideos/RandomVideos.txt"
jit_model = torch.jit.load(model_scripted_ckpt)

with open(score, "a+") as out:
    for file in os.listdir(test_dir):
        if file.endswith(".wav"):
            print(f"Processing {file}")
            file = os.path.join(test_dir, file)
            input,_ = librosa.load(file, sr=16000)
            with torch.no_grad():
                intesnor_input = torch.tensor(input).float()
                jit_out = jit_model(intesnor_input)
                out.write(f"{file} - {jit_out}\n")
        