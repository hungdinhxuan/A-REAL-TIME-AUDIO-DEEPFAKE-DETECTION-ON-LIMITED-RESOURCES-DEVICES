import torch
import os
import librosa
import numpy as np
import os
import soundfile as sf
# Load the model
model_scripted_ckpt = "/datab/hungdx/KDW2V-AASISTL/exports/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_best11_mobile.pt"
model_scaled_ckpt = "/datab/hungdx/KDW2V-AASISTL/exports/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_best_checkpoint_11_scaled2mobile.pt"


jit_model = torch.jit.load(model_scripted_ckpt)
jit_model_scaled = torch.jit.load(model_scaled_ckpt)

# audio_path = "/datab/hungdx/KDW2V-AASISTL/segment/segment"

# for file in os.listdir(audio_path):
#     handle_file = os.path.join(audio_path, file)
#     input, sr = librosa.load(handle_file, sr=16000)

#     print(f"Processing file: {handle_file}")
#     with torch.no_grad():
#         tensor_input = torch.tensor(input).float()
#         jit_out = jit_model(tensor_input)
#         jit_out_scaled = jit_model_scaled(tensor_input)
#         prediction = 1 if jit_out > 0.5 else 0  # 1 for fake, 0 for real
#         prediction_scaled = 1 if jit_out_scaled > 0.5 else 0

#         print(
#             f"Prediction: {prediction}, JIT Output: {jit_out}, Scaled Prediction: {prediction_scaled}, Scaled JIT Output: {jit_out_scaled}")


# # Define the audio file path
audio_file_path = "/datab/hungdx/KDW2V-AASISTL/금요일 15 3월 2024 - 014.mp3"
saved_chunk_path = "/datab/hungdx/KDW2V-AASISTL/금요일 15 3월 2024 - 014"

# Assuming label is known; you might need to adjust this logic based on your use case
label = 0  # 0 for real, 1 for fake

# Function to process a single audio file
os.makedirs(saved_chunk_path, exist_ok=True)


def process_audio_file(audio_file_path, sample_rate=16000, chunk_size=64600):
    # Load the audio file
    input, sr = librosa.load(audio_file_path, sr=sample_rate)
    total_chunks = len(input) // chunk_size
    correct_predictions = 0
    correct_predictions_scaled = 0
    processed_chunks = 0

    for i in range(0, len(input), chunk_size):
        end_idx = i + chunk_size if i + chunk_size < len(input) else len(input)
        audio_chunk = input[i:end_idx]
        processed_chunks += 1
        # if len(audio_chunk) < chunk_size:
        #     # Skip the last chunk if it's smaller than chunk_size
        #     # Alternatively, you can pad it to the chunk_size
        #     continue

        # Save the audio chunk to a file
        chunk_file_path = os.path.join(
            saved_chunk_path, f"chunk_{i}.wav")
        if not os.path.exists(chunk_file_path):
            sf.write(chunk_file_path, audio_chunk, sr)

        with torch.no_grad():
            tensor_input = torch.tensor(audio_chunk).float()
            jit_out = jit_model(tensor_input)
            jit_out_scaled = jit_model_scaled(tensor_input)
            prediction = 1 if jit_out > 0.5 else 0  # 1 for fake, 0 for real
            prediction_scaled = 1 if jit_out_scaled > 0.5 else 0

            if prediction == label:
                correct_predictions += 1
            if prediction_scaled == label:
                correct_predictions_scaled += 1

            print(
                f"Chunk {i//chunk_size + 1}/{total_chunks} - Prediction: {prediction}, JIT Output: {jit_out}, Scaled Prediction: {prediction_scaled}, Scaled JIT Output: {jit_out_scaled}, Accuracy: {(correct_predictions / processed_chunks) * 100}%, Scaled Accuracy: {(correct_predictions_scaled / processed_chunks) * 100}%")

    accuracy = correct_predictions / total_chunks if total_chunks > 0 else 0
    print(f"Accuracy for the audio file: {accuracy}")
    accuracy_scaled = correct_predictions_scaled / \
        total_chunks if total_chunks > 0 else 0
    print(f"Scaled Accuracy for the audio file: {accuracy_scaled}")
    return accuracy


# Process the audio file
process_audio_file(audio_file_path)
