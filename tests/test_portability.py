import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mutagen.flac import FLAC

import audio
import core


def hold_lock(output, connection):
    with core.output_lock(Path(output)):
        connection.send("locked")
        connection.recv()


class PortabilityTests(unittest.TestCase):
    def test_process_contention_and_termination(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "输出 space"
            ctx = multiprocessing.get_context("spawn")
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=hold_lock, args=(str(output), child))
            proc.start()
            child.close()
            try:
                self.assertTrue(parent.poll(20), "child failed to acquire lock")
                self.assertEqual(parent.recv(), "locked")
                with self.assertRaisesRegex(ValueError, "另一个导出任务"):
                    with core.output_lock(output):
                        self.fail("concurrent lock acquired")
            finally:
                proc.terminate()
                proc.join(10)
                parent.close()
            self.assertFalse(proc.is_alive())
            with core.output_lock(output):
                self.assertTrue((core.state_dir(output) / "lock").exists())

    def test_exception_releases_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "out"
            with self.assertRaisesRegex(RuntimeError, "test"):
                with core.output_lock(output):
                    raise RuntimeError("test")
            with core.output_lock(output):
                pass

    def test_io_failure_is_not_reported_as_contention(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("core.portalocker.lock", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    with core.output_lock(Path(folder) / "out"):
                        self.fail("lock failure ignored")

    @unittest.skipUnless(os.name == "nt", "Windows path semantics")
    def test_windows_case_uses_same_state(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "Output"
            output.mkdir()
            self.assertEqual(core.state_dir(output), core.state_dir(Path(str(output).swapcase())))

    def test_export_conversion_copy_and_repeat(self):
        exe = audio.backend()
        cfg = core.config(8)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "音源 space", root / "输出 space"
            source.mkdir()
            for index, (rate, fmt) in enumerate(((96000, "s32"), (48000, "s16"))):
                path = source / f"{index}.flac"
                audio.run(exe, ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration=0.1",
                                "-c:a", "flac", "-sample_fmt", fmt, str(path)])
                tags = FLAC(path)
                tags.update(title=["歌曲"], artist=["歌手"], album=["专辑"])
                tags.save()
                path.with_suffix(".lrc").write_text("[00:00.00]歌词", encoding="utf-8")
            before = {p.name: core.sha256(p) for p in source.iterdir()}
            with core.output_lock(output):
                entries = core.plan(source, output, cfg)
                self.assertEqual(len(entries), 2)
                for entry in entries:
                    core.execute(entry, output, cfg, exe, audio.version(exe))
                    self.assertEqual(entry.status, "success", entry.reason)
                self.assertEqual({e.status for e in core.plan(source, output, cfg)}, {"already_done"})
            files = sorted(output.glob("*.flac"))
            self.assertEqual(len(list(output.iterdir())), 4)
            self.assertEqual({p.name for p in files}, {"歌曲-歌手[专辑].flac", "歌曲-歌手[专辑] (1).flac"})
            self.assertEqual({FLAC(p).info.sample_rate for p in files}, {44100, 48000})
            for path in files:
                tags = FLAC(path)
                self.assertEqual(tags.info.bits_per_sample, 16)
                self.assertEqual(tags["lyrics"], ["[00:00.00]歌词"])
                self.assertEqual(path.with_suffix(".lrc").read_bytes(), tags["lyrics"][0].encode("utf-8"))
            self.assertEqual(before, {p.name: core.sha256(p) for p in source.iterdir()})

    def test_embedded_lyrics_priority_and_sidecar_conflicts(self):
        exe = audio.backend()
        cfg = core.config(8)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "src", root / "dest"
            source.mkdir()
            output.mkdir()
            path = source / "1.flac"
            audio.run(exe, ["-f", "lavfi", "-i", "sine=duration=0.1:sample_rate=44100",
                            "-c:a", "flac", "-sample_fmt", "s16", str(path)])
            tags = FLAC(path)
            lyrics = "[offset:10]\r\n[00:00.00]内嵌歌词\n"
            tags.update(title=["歌"], artist=["人"], album=["专辑"], lyrics=[lyrics])
            tags.save()
            path.with_suffix(".lrc").write_text("不同的外部歌词", encoding="utf-8")
            occupied = output / "歌-人[专辑].lrc"
            occupied.write_bytes(b"keep")
            with core.output_lock(output):
                entry, = core.plan(source, output, cfg)
                self.assertEqual(entry.target, "歌-人[专辑] (1).flac")
                self.assertEqual(entry.status, "copy")
                core.execute(entry, output, cfg, exe, audio.version(exe))
                self.assertEqual(entry.status, "success", entry.reason)
                sidecar = (output / entry.target).with_suffix(".lrc")
                self.assertEqual(sidecar.read_bytes(), lyrics.encode("utf-8"))
                self.assertEqual(core.sha256(path), core.sha256(output / entry.target))
                self.assertEqual(occupied.read_bytes(), b"keep")
                sidecar.write_bytes(b"edited")
                retry, = core.plan(source, output, cfg)
                self.assertEqual(retry.target, "歌-人[专辑] (2).flac")
                sidecar.unlink()
                retry, = core.plan(source, output, cfg)
                self.assertNotEqual(retry.status, "already_done")
            del tags["lyrics"]
            tags.save()
            path.with_suffix(".lrc").unlink()
            entry = core.inspect(path, cfg)
            self.assertFalse(entry.export_lrc)
            self.assertEqual(core.artifacts(entry), [entry.target])


if __name__ == "__main__":
    unittest.main()
