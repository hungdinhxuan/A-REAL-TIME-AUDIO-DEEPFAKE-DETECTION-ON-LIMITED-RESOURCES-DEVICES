
import torch



def _add_noise(speech_sig, vad_duration, noise_sig, snr):
    """add noise to the audio.
    :param speech_sig: The input audio signal (Tensor).
    :param vad_duration: The length of the human voice (int).
    :param noise_sig: The input noise signal (Tensor).
    :param snr: the SNR you want to add (int).
    :returns: noisy speech sig with specific snr.
    """
    if vad_duration != 0:
        snr = 10**(snr/10.0)
        speech_power = torch.sum(speech_sig**2)/vad_duration
        noise_power = torch.sum(noise_sig**2)/noise_sig.shape[1]
        noise_update = noise_sig / torch.sqrt(snr * noise_power/speech_power)

        if speech_sig.shape[1] > noise_update.shape[1]:
            # padding
            temp_wav = torch.zeros(1, speech_sig.shape[1])
            temp_wav[0, 0:noise_update.shape[1]] = noise_update
            noise_update = temp_wav
        else:
            # cutting
            noise_update = noise_update[0, 0:speech_sig.shape[1]]

        return noise_update + speech_sig
    
    else:
        return speech_sig