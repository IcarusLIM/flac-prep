from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
import uuid

from mutagen.flac import FLAC, Picture
from PIL import Image

import audio

__version__ = "0.1.0"

READY = {"copy", "metadata", "convert"}
OK = {"success", "already_done"}
OTHER_AUDIO = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wav", ".aiff", ".ape", ".wma"}
LYRIC_KEYS = ("lyrics", "unsyncedlyrics", "unsynced lyrics")


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(f)


def _hash(f):
    h = hashlib.sha256()
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
    return h.hexdigest()


def key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def clean(value: str) -> str:
    value = unicodedata.normalize("NFC", value).strip().replace("[", "［").replace("]", "］")
    value = re.sub(r'[<>:"/\\|?*]', "_", value)
    value = "".join(c for c in value if unicodedata.category(c) != "Cc").rstrip(" .")
    if not value:
        raise ValueError("清理后的文件名字段为空")
    return value


def target_spec(rate: int, bits: int) -> tuple[int, int]:
    if 44100 < rate < 48000:
        raise ValueError("unsupported_rate")
    if rate <= 0 or bits <= 0:
        raise ValueError("无效音频规格")
    return (44100 if rate > 48000 else rate, min(bits, 16))


def config(compression: int) -> dict:
    return {"version": __version__, "policy": "portable-flac-only-v2", "compression": compression,
            "resampler": "soxr-vhq", "dither": "triangular", "noise_shaping": False,
            "peak_threshold_lsb": 8, "attenuated_peak_dbfs": -0.1}


@dataclass
class Entry:
    source: str
    status: str
    target: str | None = None
    reason: str = ""
    source_spec: dict = field(default_factory=dict)
    target_spec: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    lrc: str | None = None
    lrc_text: str | None = None
    embed_lyrics: bool = False
    cover: str | None = None
    fingerprint: str = ""
    processing: dict = field(default_factory=dict)
    removed_tags: list = field(default_factory=list)

    def public(self):
        value = asdict(self)
        value.pop("lrc_text")  # reports must not duplicate lyric text
        return value


def files_under(root: Path, output: Path):
    if root.is_file():
        if root.is_symlink():
            raise ValueError("不处理符号链接输入")
        yield root
        return
    if not root.is_dir() or root.is_symlink():
        raise ValueError("输入必须是普通文件或目录")
    if root == output:
        raise ValueError("输出目录不能与输入目录相同")
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(parent) / d).is_symlink()
                         and (Path(parent) / d).resolve() != output)
        for name in sorted(files):
            path = Path(parent) / name
            if not name.startswith(".") and not path.is_symlink():
                yield path


def matching(path: Path, suffixes: set[str]) -> list[Path]:
    return sorted(p for p in path.parent.iterdir() if p.is_file() and not p.is_symlink()
                  and p.stem == path.stem and p.suffix.lower() in suffixes)


def embedded_lyrics(flac: FLAC) -> str | None:
    for name in LYRIC_KEYS:
        for value in flac.get(name, []):
            if value.strip():
                return value
    return None


def verify_picture(data: bytes) -> tuple[int, int, str]:
    with Image.open(BytesIO(data)) as picture:
        width, height, fmt = picture.width, picture.height, picture.format
        picture.verify()
    if fmt not in {"JPEG", "PNG"}:
        raise ValueError("封面只支持 JPEG/PNG")
    return width, height, "image/jpeg" if fmt == "JPEG" else "image/png"


def state_dir(output: Path) -> Path:
    identity = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
    return output.parent / ".flac-mgr" / identity


def record_path(output: Path, target: str) -> Path:
    name = hashlib.sha256(target.encode()).hexdigest()
    return state_dir(output) / "records" / f"{name}.json"


def artifacts(entry: Entry) -> list[str]:
    return [entry.target]


def read_record(path: Path) -> dict:
    try:
        if path.is_symlink():
            return {}
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def completed(entry: Entry, output: Path) -> bool:
    record = read_record(record_path(output, entry.target))
    hashes = record.get("outputs", {})
    return (record.get("state") == "complete" and record.get("fingerprint") == entry.fingerprint
            and set(hashes) == set(artifacts(entry))
            and all((output / n).is_file() and not (output / n).is_symlink()
                    and sha256(output / n) == h for n, h in hashes.items()))


def inspect(path: Path, cfg: dict) -> Entry:
    entry = Entry(str(path), "error")
    try:
        entry.inputs[str(path)] = sha256(path)
        with path.open("rb") as f:
            if f.read(4) != b"fLaC":
                raise ValueError("不是原生 FLAC 文件")
        flac = FLAC(path)
        info = flac.info
        entry.source_spec = {"rate": info.sample_rate, "bits": info.bits_per_sample,
                             "channels": info.channels, "frames": info.total_samples}
        rate, bits = target_spec(info.sample_rate, info.bits_per_sample)
        entry.target_spec = {"rate": rate, "bits": bits, "channels": info.channels}
        if info.total_samples <= 0:
            raise ValueError("源 FLAC 未包含有效采样帧数")
        if any(not flac.get(k) or not any(v.strip() for v in flac[k]) for k in ("title", "artist", "album")):
            entry.status = "needs_metadata"
            entry.reason = "缺少 TITLE、ARTIST 或 ALBUM"
            return entry
        if len(flac["title"]) != 1 or len(flac["album"]) != 1:
            raise ValueError("TITLE 或 ALBUM 包含多个值，无法确定名称")
        artist = "、".join(v for v in flac["artist"] if v.strip())
        entry.target = f'{clean(flac["title"][0])}-{clean(artist)}[{clean(flac["album"][0])}].flac'
        if len(entry.target.encode("utf-8")) > 255 or len(entry.target.encode("utf-16-le")) // 2 > 255:
            entry.status = "name_too_long"
            return entry
        lyrics = embedded_lyrics(flac)
        sidecars = matching(path, {".lrc"})
        if len(sidecars) > 1:
            raise ValueError("多个同名 LRC，配对不明确")
        if sidecars:
            entry.lrc = str(sidecars[0])
            data = sidecars[0].read_bytes()
            entry.inputs[entry.lrc] = hashlib.sha256(data).hexdigest()
            entry.lrc_text = data.decode("utf-8-sig")
            if not entry.lrc_text.strip():
                entry.warnings.append("外部 LRC 为空，未嵌入")
            elif not lyrics:
                entry.embed_lyrics = True
            elif lyrics.strip() != entry.lrc_text.strip():
                entry.warnings.append("内嵌歌词和源 LRC 不同，保留已有内嵌歌词")
        if not lyrics and not entry.embed_lyrics:
            entry.warnings.append("无可用内嵌歌词")
        for picture in flac.pictures:
            verify_picture(picture.data)
        if not flac.pictures:
            covers = matching(path, {".jpg", ".jpeg", ".png"})
            if len(covers) > 1:
                raise ValueError("多个同名封面，配对不明确")
            if covers:
                entry.cover = str(covers[0])
                data = covers[0].read_bytes()
                verify_picture(data)
                entry.inputs[entry.cover] = hashlib.sha256(data).hexdigest()
            else:
                entry.warnings.append("没有封面")
        if rate < 44100 or bits < 16:
            entry.warnings.append("低规格输入，未自动提升")
        if info.channels > 2:
            entry.warnings.append("多声道输入，保留声道数；设备兼容性需确认")
        entry.status = "convert" if (rate, bits) != (info.sample_rate, info.bits_per_sample) else (
            "metadata" if entry.embed_lyrics or entry.cover else "copy")
        if entry.status == "convert" and bits != 16:
            entry.status, entry.reason = "unsupported_format", "低位深高采样率输入暂不转换"
        payload = {"inputs": entry.inputs, "config": cfg, "target": entry.target,
                   "target_spec": entry.target_spec, "action": entry.status}
        entry.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    except Exception as exc:
        entry.status, entry.reason = "error", str(exc)
    return entry


def plan(source: Path, output: Path, cfg: dict) -> list[Entry]:
    paths = list(files_under(source, output))
    entries = []
    flacs = {(p.parent, p.stem) for p in paths if p.suffix.lower() == ".flac"}
    for path in paths:
        ext = path.suffix.lower()
        if ext == ".flac":
            entries.append(inspect(path, cfg))
        elif ext in OTHER_AUDIO:
            entries.append(Entry(str(path), "unsupported_format", reason="首版仅支持 FLAC"))
        elif ext == ".lrc" and (path.parent, path.stem) not in flacs and not matching(path, {".flac"}):
            entries.append(Entry(str(path), "unpaired_lrc", reason="无同目录同名 FLAC"))
    occupied = {key(p.name) for p in output.iterdir()} if output.is_dir() else set()
    records = defaultdict(list)
    for path in sorted((state_dir(output) / "records").glob("*.json")):
        record = read_record(path)
        if record.get("state") == "complete":
            target = record.get("entry", {}).get("target")
            if isinstance(target, str) and Path(target).name == target:
                records[record.get("fingerprint")].append(target)
    source_paths = {Path(p).resolve() for e in entries for p in e.inputs}
    for entry in entries:
        if entry.status not in READY:
            continue
        base = entry.target
        try:
            # Look up completed suffixed outputs first, even if an earlier name is now free.
            for target in records.get(entry.fingerprint, []):
                entry.target = target
                if completed(entry, output):
                    entry.status = "already_done"
                    break
            if entry.status == "already_done":
                continue
            number = 0
            while True:
                entry.target = base if number == 0 else f"{Path(base).stem} ({number}).flac"
                if len(entry.target.encode("utf-8")) > 255 or len(entry.target.encode("utf-16-le")) // 2 > 255:
                    entry.status, entry.reason = "name_too_long", "添加后缀后文件名过长"
                    break
                if key(entry.target) not in occupied:
                    occupied.add(key(entry.target))
                    break
                number += 1
            if any((output / n).resolve() in source_paths for n in artifacts(entry)):
                entry.status, entry.reason = "source_conflict", "目标指向源文件"
        except Exception as exc:
            entry.status, entry.reason = "error", str(exc)
    return entries


def assert_inputs(entry: Entry):
    for name, digest in entry.inputs.items():
        path = Path(name)
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"处理期间源文件或资源发生变化：{name}")


def update_metadata(entry: Entry, dest: Path, converted: bool):
    source = FLAC(entry.source)
    result = FLAC(dest)
    if converted:
        result.clear()
        for name, values in source.items():
            if name.startswith(("replaygain_", "r128_")) or name in {"peak", "peak_value", "replaygain"}:
                entry.removed_tags.append(name)
            else:
                result[name] = values
        result["encoder"] = [f"flac-mgr {__version__}"]
        result.clear_pictures()
        for pic in source.pictures:
            result.add_picture(pic)
    if entry.embed_lyrics:
        result["lyrics"] = [entry.lrc_text]
    if entry.cover:
        data = Path(entry.cover).read_bytes()
        w, h, mime = verify_picture(data)
        picture = Picture()
        picture.type, picture.mime, picture.width, picture.height = 3, mime, w, h
        picture.data = data
        result.add_picture(picture)
    expected_tags = dict(result)
    expected_pictures = [p.write() for p in result.pictures]
    result.save()
    reread = FLAC(dest)
    if dict(reread) != expected_tags or [p.write() for p in reread.pictures] != expected_pictures:
        raise ValueError("目标标签或封面写入校验失败")


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def output_lock(output: Path):
    # macOS/Linux advisory lock is released by the OS even after an interrupted run.
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    state = state_dir(output)
    if state.parent.is_symlink() or state.is_symlink():
        raise ValueError("状态目录不能是符号链接")
    state.mkdir(parents=True, exist_ok=True)
    for name in ("records", "reports"):
        if (state / name).is_symlink():
            raise ValueError(f"状态子目录不能是符号链接：{name}")
    lock = state / "lock"
    if lock.is_symlink():
        raise ValueError("锁文件不能是符号链接")
    with lock.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("另一个导出任务正在使用该输出目录") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def recover(output: Path):
    """Remove only content proven to belong to an interrupted prepared transaction."""
    folder = state_dir(output) / "records"
    if folder.is_symlink():
        raise ValueError("记录目录不能是符号链接")
    for path in folder.glob("*.json"):
        record = read_record(path)
        if record.get("state") != "prepared":
            continue
        hashes = record.get("outputs", {})
        if not hashes or any(Path(n).name != n or n.startswith(".") for n in hashes):
            continue
        # Refuse to remove anything if any artifact has been changed externally.
        if any((output / n).is_symlink() or
               ((output / n).exists() and (not (output / n).is_file() or sha256(output / n) != h))
               for n, h in hashes.items()):
            continue
        for name in hashes:
            artifact = output / name
            if not artifact.exists():
                continue
            stat = artifact.stat()
            identity = record.get("identities", {}).get(name)
            # The inode is recorded before linking: this also covers a crash between
            # link() and the following journal update, without deleting unrelated files.
            if identity == [stat.st_dev, stat.st_ino] or (
                    identity is None and name in record.get("committed", [])):
                artifact.unlink()
        record["state"] = "interrupted"
        atomic_json(path, record)


def execute(entry: Entry, output: Path, cfg: dict, exe: str, backend_version: str):
    action = entry.status
    record = None
    committed = []
    try:
        assert_inputs(entry)
        with tempfile.TemporaryDirectory(prefix=".flac-mgr-stage-", dir=output) as temp:
            stage = Path(temp)
            dest = stage / entry.target
            if action == "convert":
                entry.processing = audio.convert(Path(entry.source), dest, stage / "audio.f64",
                                                 entry.target_spec["rate"], entry.target_spec["bits"],
                                                 cfg["compression"], exe)
            else:
                source_pcm = audio.pcm_digest(Path(entry.source), exe)
                shutil.copyfile(entry.source, dest)
                entry.processing = {"gain_db": 0, "dither": None, "pcm_modified": False}
            if action != "copy":
                update_metadata(entry, dest, action == "convert")
            output_pcm = audio.pcm_digest(dest, exe)
            if action != "convert" and source_pcm != output_pcm:
                raise ValueError("无需音频转换的路径改变了 PCM")
            if action == "copy" and sha256(dest) != entry.inputs[entry.source]:
                raise ValueError("复制校验失败")
            assert_inputs(entry)
            names = artifacts(entry)
            # Recheck just before committing; os.link below additionally provides no-clobber semantics.
            current = {key(p.name) for p in output.iterdir()}
            if any(key(n) in current for n in names):
                raise FileExistsError("提交前发现同名目标，未覆盖")
            hashes = {n: sha256(stage / n) for n in names}
            identities = {n: [(stage / n).stat().st_dev, (stage / n).stat().st_ino] for n in names}
            for name in names:
                with (stage / name).open("rb") as f:
                    os.fsync(f.fileno())
            record = {"state": "prepared", "fingerprint": entry.fingerprint, "outputs": hashes,
                      "identities": identities,
                      "inputs": entry.inputs, "config": cfg, "backend": backend_version,
                      "action": action, "processing": entry.processing, "committed": committed,
                      "entry": entry.public(), "created_at": datetime.now(timezone.utc).isoformat()}
            rp = record_path(output, entry.target)
            atomic_json(rp, record)
            for name in names:
                os.link(stage / name, output / name)
                committed.append(name)
                atomic_json(rp, record)
            record["state"] = "complete"
            atomic_json(rp, record)
        entry.status = "success"
    except Exception as exc:
        if record:
            for name in committed:
                p = output / name
                if p.is_file() and not p.is_symlink() and sha256(p) == record["outputs"][name]:
                    p.unlink()
            record["state"] = "failed"
            record["error"] = str(exc)
            atomic_json(record_path(output, entry.target), record)
        entry.status, entry.reason = "error", str(exc)

