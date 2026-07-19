import torch
import torchaudio
import torch.nn.functional as F


def waveform_to_spectrogram(waveform, sample_rate, target_sample_rate=16000, time=5):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        power=2.0,
    )

    to_db = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=80,
    )

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_sample_rate,
        )
        sample_rate = target_sample_rate

    waveform = waveform[:, : (target_sample_rate * time)]
    if waveform.shape[-1] < (target_sample_rate * time):
        waveform = F.pad(
            waveform,
            (0, (target_sample_rate * time) - waveform.shape[-1]),
        )

    log_mel = to_db(mel_transform(waveform))
    log_mel = ((log_mel + 80.0) / 80.0).clamp(0.0, 1.0)

    return log_mel


def log_mel_to_waveform(log_mel, sample_rate=16000, time=5.0):
    mel_db = log_mel * 80.0 - 80.0

    mel_power = torchaudio.functional.DB_to_amplitude(
        mel_db,
        ref=1.0,
        power=1.0,
    )

    mel_power = torch.where(
        log_mel <= 1e-4,
        torch.zeros_like(mel_power),
        mel_power,
    )

    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=1024 // 2 + 1,
        n_mels=128,
        sample_rate=sample_rate,
    ).to(device=mel_power.device, dtype=mel_power.dtype)

    linear_power = inverse_mel(mel_power)
    linear_power = linear_power.clamp_min(0.0)

    print(linear_power.device)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        power=2.0,
        n_iter=256,
        length=int(sample_rate * time),
    )

    waveform = griffin_lim(linear_power)

    return waveform, sample_rate
