from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cli
import core


class ProgressTests(unittest.TestCase):
    def test_results_are_visible_before_next_conversion(self):
        for as_json in (False, True):
            with self.subTest(as_json=as_json), tempfile.TemporaryDirectory() as folder:
                stdout, stderr = StringIO(), StringIO()
                entries = [core.Entry("first.flac", "convert", target="first.flac"),
                           core.Entry("second.flac", "convert", target="second.flac")]

                def execute(entry, *args):
                    if entry is entries[1]:
                        stream = stderr if as_json else stdout
                        self.assertIn("[success] first.flac", stream.getvalue())
                        self.assertIn("提示：test warning", stream.getvalue())
                    entry.status = "success"
                    entry.warnings.append("test warning")

                with patch("cli.plan", return_value=entries), patch("cli.execute", side_effect=execute), \
                        patch("cli.audio.backend", return_value="ffmpeg"), \
                        patch("cli.audio.version", return_value="test"), \
                        redirect_stdout(stdout), redirect_stderr(stderr):
                    args = ["export", folder, "-o", str(Path(folder) / "out")]
                    self.assertEqual(cli.main(args + (["--json"] if as_json else [])), 0)
                if as_json:
                    self.assertEqual([e["status"] for e in json.loads(stdout.getvalue())], ["success", "success"])
                else:
                    self.assertEqual(stdout.getvalue().count("[success] first.flac"), 1)
                    self.assertEqual(stdout.getvalue().count("汇总："), 1)

    def test_scan_progress_precedes_file_read(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "one.flac").touch()
            events = []

            def inspect(path, cfg):
                self.assertEqual(events[-1], ("扫描并计算源文件哈希", 1, 1, path))
                return core.Entry(str(path), "error")

            with patch("core.inspect", side_effect=inspect):
                core.plan(source, source / "out", {}, progress=lambda *args: events.append(args))
            self.assertEqual(events[0][0], "枚举文件")
