import torch
import os
import librosa
import numpy as np

# Load the model
model_scripted_ckpt = "/datab/hungdx/KDW2V-AASISTL/exports/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_best_checkpoint_11_scaledmobile.pt"
# Get last name from the path
model_name = os.path.basename(model_scripted_ckpt)

jit_model = torch.jit.load(model_scripted_ckpt)

# Define the root directory where subfolders are located
root_dir = "/datab/hungdx/KDW2V-AASISTL/test_samples/test"

# Function to process each subfolder and calculate accuracy


# Function to process each subfolder and calculate accuracy
def process_subfolder(subfolder_path, score_file_path):
    total = 0
    correct = 0
    # Check if the folder is labeled as fake or real
    label = 1 if 'fake' in subfolder_path else 0

    with open(score_file_path, "a+") as out:
        for file in os.listdir(subfolder_path):
            if file.endswith(".wav"):
                total += 1
                print(f"Processing {file}")
                file_path = os.path.join(subfolder_path, file)
                input, _ = librosa.load(file_path, sr=16000)
                # Trim
                # yt, index = librosa.effects.trim(input, top_db=20)
                # input = yt

                with torch.no_grad():
                    tensor_input = torch.tensor(input).float()
                    jit_out = jit_model(tensor_input)
                    prediciton = 1 if jit_out > 0.5 else 0
                    if prediciton == label:
                        correct += 1
                    print(f"{file_path} - {jit_out}")
                    out.write(f"{file_path} - {jit_out}\n")
        accuracy = correct / total if total > 0 else 0
        out.write(
            f"Total: {total}, Correct: {correct}, Accuracy: {accuracy}\n")
    return accuracy


# Calculate the accuracy for each subfolder and then the average
accuracies = []
for subfolder_name in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, subfolder_name)):
        print(f"Processing subfolder: {subfolder_name}")
        subfolder_path = os.path.join(root_dir, subfolder_name)
        score_file_path = os.path.join(
            subfolder_path, f"{model_name}_accuracy_score.txt")
        if os.path.exists(score_file_path):
            print(f"Score file exists: {score_file_path}")
            with open(score_file_path, "r") as f:
                lines = f.readlines()
                print("Last line: ", lines[-1])
                acc = float(lines[-1].split()[-1])
                print(f"Accuracy: {acc}")
                accuracies.append(acc)
                continue
        acc = process_subfolder(subfolder_path, score_file_path)
        accuracies.append(acc)

# Compute the average accuracy
average_accuracy = np.mean(accuracies) if accuracies else 0
print(f"Average Accuracy: {average_accuracy}")


# Calculate the accuracy for each subfolder and then the average
fake_accuracies = []

real_accuracies = []

total_fake = 0
total_real = 0

for subfolder_name in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, subfolder_name)):
        print(f"Processing subfolder: {subfolder_name}")
        subfolder_path = os.path.join(root_dir, subfolder_name)
        score_file_path = os.path.join(
            subfolder_path, f"{model_name}_accuracy_score.txt")

        with open(score_file_path, "r") as f:
            lines = f.readlines()
            print("Last line: ", lines[-1])
            acc = float(lines[-1].split()[-1])
            print(f"Accuracy: {acc}")

        if 'fake' in subfolder_name:
            fake_accuracies.append(acc)
            total_fake += float(lines[-1].split()[1].split(',')[0])

        else:
            real_accuracies.append(acc)
            total_real += float(lines[-1].split()[1].split(',')[0])

# Compute the average accuracy
average_fake_accuracy = np.mean(fake_accuracies) if fake_accuracies else 0
average_real_accuracy = np.mean(real_accuracies) if real_accuracies else 0

print(f"Average Fake Accuracy: {average_fake_accuracy}")
print(f"Total Fake: {total_fake}")
print(f"Average Real Accuracy: {average_real_accuracy}")
print(f"Total Real: {total_real}")
print(
    f"Average Accuracy: {(average_fake_accuracy + average_real_accuracy) / 2}")
