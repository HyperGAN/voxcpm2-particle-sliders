"""Matched waveform fidelity and explicit perceptual evidence requirements."""
import math
import torch


def stock_fixed_waveform(model, record, calibration):
    """Unmodified stock generate, with its independent text-length cap disabled.

    Duration is still bounded to at most 64 patches by the calibration protocol.
    """
    if (not calibration["cfg_zero_star"] or calibration["sway"] != 1.
            or type(calibration["frames"]) is not int or not 1 <= calibration["frames"] <= 64
            or type(calibration["steps"]) is not int or not 2 <= calibration["steps"] <= 32):
        raise ValueError("stock sampler requires bounded calibration and default zero-star/sway")
    parameter = next(model.parameters())
    devices = [parameter.device.index or 0] if parameter.is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(int(record["seed"]))
        if devices: torch.cuda.manual_seed(int(record["seed"]))
        audio = model.generate(target_text=record["text"],
            reference_wav_path=record.get("neutral_reference", ""),
            min_len=calibration["frames"] + 1, max_len=calibration["frames"],
            inference_timesteps=calibration["steps"], cfg_value=calibration["cfg_value"],
            retry_badcase=False, retry_badcase_ratio_threshold=1e6)
    expected_samples = calibration["frames"] * model.patch_size * model._decode_chunk_size
    if audio.shape != (1, expected_samples):
        raise ValueError("stock waveform duration differs from fixed calibration")
    return audio


def waveform_metrics(native, distilled):
    if native.shape != distilled.shape or native.ndim != 2:
        raise ValueError("matched mono waveform lengths required; no alignment/truncation")
    if not torch.isfinite(native).all() or not torch.isfinite(distilled).all():
        raise ValueError("finite waveforms required")
    x, y = native.float(), distilled.float()
    error = (x - y).square().mean()
    power = x.square().mean()
    n_fft = min(1024, x.shape[-1])
    if n_fft < 8:
        raise ValueError("at least eight native samples required")
    window = torch.hann_window(n_fft, device=x.device)
    sx = torch.stft(x, n_fft, hop_length=max(1, n_fft // 4), window=window,
                    return_complex=True, center=False).abs()
    sy = torch.stft(y, n_fft, hop_length=max(1, n_fft // 4), window=window,
                    return_complex=True, center=False).abs()
    return dict(samples=x.shape[-1], mse=float(error),
                relative_mse=float(error / power) if power > 0 else None,
                snr_db=float(10 * torch.log10(power / error)) if error > 0 and power > 0 else None,
                log_spectral_distance_db=float(((20 * torch.log10(sx.clamp_min(1e-5))
                    - 20 * torch.log10(sy.clamp_min(1e-5))).square().mean()).sqrt()),
                native_rms=float(power.sqrt()), distilled_rms=float(y.square().mean().sqrt()),
                native_clip_fraction=float((x.abs() >= 1).float().mean()),
                distilled_clip_fraction=float((y.abs() >= 1).float().mean()))


def evidence_template():
    return dict(waveform_fidelity={"status": "requires matched held-out renders"},
                intelligibility={"status": "pending", "required": "reference text and independent ASR/human transcript; WER/CER"},
                identity={"status": "pending", "required": "held-out reference speaker similarity and blind listening"},
                expressivity={"status": "pending", "required": "blind neutral/sleepy/excited/laughing ratings on held-out text"},
                strength={"status": "pending", "required": "matched Off/0.5/1/2; zero bypass, monotonic edit, identity/intelligibility retention"},
                promotion_allowed=False)
