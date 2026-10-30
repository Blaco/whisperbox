"""Optional input speech effects. Defaults bypass processing exactly."""
import numpy as np


def _overlap_transform(audio, sample_rate, envelope_ratio=1., noise_amount=0.):
    """Smooth log-spectrum envelope adjustment and adaptive spectral denoising."""
    from scipy.signal import stft, istft
    n = 2048 if sample_rate < 32000 else 4096
    hop = n // 8
    _, _, spectrum = stft(audio, fs=sample_rate, window='hann', nperseg=n,
                          noverlap=n-hop, boundary='zeros', padded=True)
    magnitude = np.abs(spectrum)
    if noise_amount:
        # Use a conservative time-frequency estimate of steady background noise.
        floor = np.percentile(magnitude, 20, axis=1)
        from scipy.ndimage import maximum_filter1d, gaussian_filter
        floor = gaussian_filter(floor, sigma=4)
        snr = magnitude / np.maximum(floor[:, None], 1.e-9)
        # Stronger settings remove stationary noise, but preserve clearly voiced bins.
        attenuation = 10. ** (-noise_amount * 24. / 20.)
        # Extend intensity above the former 1.0 ceiling. Exponentiation also
        # strengthens attenuation in moderately noisy bins, where raising only
        # the dB floor would otherwise have little audible impact.
        base_gain = np.clip((snr - .85) / (snr + .5), 0., 1.)
        gain = np.maximum(np.power(base_gain, 1.0 + noise_amount), attenuation)
        gain = gaussian_filter(gain, sigma=(1.2, 1.0))
        spectrum = spectrum * gain
        magnitude = np.abs(spectrum)
    if envelope_ratio != 1.:
        # Warp the smooth *cepstral* spectral envelope, leaving fine harmonic
        # spacing and FFT phase in place. This changes resonance rather than F0.
        frames = spectrum.shape[1]
        bins = magnitude.shape[0]
        log_mag = np.log(np.maximum(magnitude, 1.e-8))
        cepstra = np.fft.irfft(log_mag, n=n, axis=0)
        cutoff = max(5, min(n // 8, round(sample_rate / 1000)))
        lifter = np.zeros_like(cepstra)
        lifter[:cutoff + 1] = cepstra[:cutoff + 1]
        lifter[-cutoff:] = cepstra[-cutoff:]
        envelope = np.fft.rfft(lifter, n=n, axis=0).real
        positions = np.arange(bins, dtype=np.float64)
        warped = np.empty((bins, frames), dtype=np.float64)
        for i in range(frames):
            warped[:, i] = np.interp(positions / envelope_ratio, positions,
                                      envelope[:, i], left=envelope[0, i],
                                      right=envelope[-1, i])
        gain = np.exp(np.clip(warped - envelope, -2.0, 2.0))
        spectrum *= gain
    _, output = istft(spectrum, fs=sample_rate, window='hann', nperseg=n,
                      noverlap=n-hop, input_onesided=True, boundary=True)
    if len(output) < len(audio):
        output = np.pad(output, (0, len(audio)-len(output)))
    return np.asarray(output[:len(audio)], dtype=np.float32)


def process(audio, sample_rate, pitch=0., formant=1., cleanup=0.):
    """Adjust mono float speech without changing sample rate or duration."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all() or sample_rate < 8000:
        raise ValueError('Invalid input audio')
    if not -12 <= float(pitch) <= 12 or not .7 <= float(formant) <= 1.3 or not 0 <= float(cleanup) <= 2:
        raise ValueError('Audio effect setting out of range')
    if not samples.size or (pitch == 0 and formant == 1 and cleanup == 0):
        return samples.copy()
    result = samples.copy()
    if pitch:
        # Offline Rubber Band high-quality mode, with formant preservation.
        # Pedalboard includes the native engine in its Windows Python wheel.
        from pedalboard import time_stretch
        result = time_stretch(np.ascontiguousarray(result, dtype=np.float32),
                              float(sample_rate), stretch_factor=1.0,
                              pitch_shift_in_semitones=float(pitch),
                              high_quality=True, preserve_formants=True,
                              use_long_fft_window=True).reshape(-1).astype(np.float32)
        if len(result) < len(samples):
            result = np.pad(result, (0, len(samples) - len(result)))
    if (formant != 1 or cleanup > 0) and len(result) >= 4096:
        result = _overlap_transform(result, int(sample_rate), float(formant), float(cleanup))
    return np.clip(np.nan_to_num(result[:len(samples)], nan=0., posinf=0., neginf=0.), -1., 1.).astype(np.float32)
