"""Streaming decode, libsoxr conversion, final FFmpeg TPDF quantization."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import soxr
from mutagen.flac import FLAC


def backend(explicit: str | None = None) -> str:
    candidate = explicit or os.environ.get("FLAC_MGR_FFMPEG") or shutil.which("ffmpeg")
    if not candidate:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
    return str(candidate)


def run(exe: str, args: list[str]) -> None:
    result = subprocess.run([exe, "-hide_banner", "-nostdin", "-v", "error", "-xerror", *args],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-4000:])


def version(exe: str) -> str:
    result = subprocess.run([exe, "-version"], capture_output=True, text=True, check=True)
    return result.stdout.splitlines()[0]


def pcm_chunks(path: Path, exe: str):
    """Decode fully; validate frame count and the FLAC STREAMINFO MD5 if present."""
    info = FLAC(path).info
    count = 0
    native_md5 = hashlib.md5()
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(
            [exe, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
             "-i", str(path), "-map", "0:a:0", "-vn", "-c:a", "pcm_s32le", "-f", "s32le", "pipe:1"],
            stdout=subprocess.PIPE, stderr=errors,
        )
        try:
            while data := proc.stdout.read(65536 * info.channels * 4):
                if len(data) % (info.channels * 4):
                    raise ValueError("解码 PCM 数据不完整")
                samples = np.frombuffer(data, dtype="<i4").reshape(-1, info.channels)
                count += len(samples)
                if info.bits_per_sample in (8, 16, 24, 32):
                    native = (samples.reshape(-1) >> (32 - info.bits_per_sample)).astype("<i4")
                    packed = native.view(np.uint8).reshape(-1, 4)[:, :info.bits_per_sample // 8]
                    native_md5.update(packed.tobytes())
                yield samples
            if proc.wait():
                errors.seek(0)
                raise RuntimeError(errors.read().decode("utf-8", "replace")[-4000:])
            if count != info.total_samples:
                raise ValueError(f"解码帧数不符：{count} != {info.total_samples}")
            if info.md5_signature and info.bits_per_sample in (8, 16, 24, 32):
                if int.from_bytes(native_md5.digest(), "big") != info.md5_signature:
                    raise ValueError("FLAC 音频 MD5 校验失败")
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def pcm_digest(path: Path, exe: str) -> str:
    digest = hashlib.sha256()
    for samples in pcm_chunks(path, exe):
        digest.update(samples.tobytes())
    return digest.hexdigest()


def convert(source: Path, dest: Path, raw: Path, rate: int, bits: int,
            compression: int, exe: str) -> dict:
    info = FLAC(source).info
    if bits != 16:
        raise ValueError("首版转换后端只量化到 16-bit；低位深高采样率输入暂不支持")
    resampler = (soxr.ResampleStream(info.sample_rate, rate, info.channels, dtype="float64", quality="VHQ")
                 if info.sample_rate != rate else None)
    peak, frames = 0.0, 0
    with raw.open("wb") as stream:
        def write(block):
            nonlocal peak, frames
            if block.size:
                if not np.isfinite(block).all():
                    raise ValueError("重采样结果包含非有限数值")
                peak = max(peak, float(np.max(np.abs(block))))
                frames += len(block)
                stream.write(np.asarray(block, dtype="<f8").tobytes())
        for samples in pcm_chunks(source, exe):
            block = samples.astype(np.float64) / 2147483648.0
            write(resampler.resample_chunk(block) if resampler else block)
        if resampler:
            write(resampler.resample_chunk(np.empty((0, info.channels), dtype=np.float64), last=True))
    expected = info.total_samples * rate / info.sample_rate
    if abs(frames - expected) > 1:
        raise ValueError(f"重采样时长不符：{frames} vs {expected}")
    # Leave eight 16-bit LSBs for rounding/dither; attenuate risky signals to -0.1 dBFS.
    gain = 10 ** (-0.1 / 20) / peak if peak > 1 - 8 / 32768 else 1.0
    filters = (f"volume={gain:.17g}:precision=double,"
               "aresample=osf=s16:tsf=dblp:dither_method=triangular:output_sample_bits=16")
    run(exe, ["-f", "f64le", "-ar", str(rate), "-ac", str(info.channels), "-i", str(raw),
              "-af", filters, "-c:a", "flac", "-sample_fmt", "s16", "-compression_level", str(compression),
              "-map_metadata", "-1", "-y", str(dest)])
    result = FLAC(dest).info
    if (result.sample_rate, result.bits_per_sample, result.channels, result.total_samples) != (
            rate, bits, info.channels, frames):
        raise ValueError("编码后音频规格或采样帧数不符")
    return {"peak_before_quantization": peak, "gain_db": 20 * math.log10(gain),
            "frames": frames, "resampler": "libsoxr VHQ" if resampler else None,
            "soxr_version": soxr.__version__, "dither": "TPDF", "noise_shaping": False}

