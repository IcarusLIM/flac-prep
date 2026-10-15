from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

import audio
from core import __version__, OK, READY, atomic_json, config, execute, output_lock, plan, recover, state_dir

PROJECT_ROOT = Path(__file__).resolve().parent

def display(entries, as_json, *, summary=True, stream=None):
    stream = sys.stdout if stream is None else stream
    if as_json:
        print(json.dumps([e.public() for e in entries], ensure_ascii=False, indent=2), file=stream, flush=True)
        return
    for entry in entries:
        old, new = entry.source_spec, entry.target_spec
        spec = f' {old["bits"]}/{old["rate"]} → {new["bits"]}/{new["rate"]}' if old and new else ""
        print(f"[{entry.status}]{spec} {entry.source}", file=stream)
        if entry.target:
            print(f"  → {entry.target}", file=stream)
        if entry.embed_lyrics:
            print(f"  补写内嵌歌词：{entry.lrc}", file=stream)
        if entry.reason:
            print(f"  {entry.reason}", file=stream)
        for warning in entry.warnings:
            print(f"  提示：{warning}", file=stream)
        stream.flush()
    if summary:
        display_summary(entries, stream)


def display_summary(entries, stream):
    print("汇总：" + ", ".join(f"{k}={v}" for k, v in sorted(Counter(e.status for e in entries).items())),
          file=stream, flush=True)


def progress(stage, current, total, path):
    count = f" {current}/{total}" if current is not None else ""
    print(f"[{stage}{count}] {path}", file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="离线整理 FLAC：保留原文件、补写同名 LRC、导出便携副本")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "export"):
        sub = subs.add_parser(command, help="只读预览" if command == "preview" else "导出新文件")
        sub.add_argument("input", type=Path, nargs="?", default=PROJECT_ROOT / "data" / "src",
                         help="输入文件或目录，默认项目下 data/src")
        sub.add_argument("-o", "--output", type=Path, default=PROJECT_ROOT / "data" / "dest",
                         help="输出目录，默认项目下 data/dest")
        sub.add_argument("--compression", type=int, choices=range(0, 13), default=8, metavar="0..12")
        sub.add_argument("--json", action="store_true", help="以 JSON 输出结果")
        if command == "export":
            sub.add_argument("--ffmpeg", help="指定 FFmpeg 可执行文件，默认自动寻找或使用随包二进制")
    args = parser.parse_args(argv)
    try:
        if args.input.is_symlink():
            raise ValueError("输入不能是符号链接")
        source, output = args.input.resolve(), args.output.resolve()
        cfg = config(args.compression)
        if args.command == "preview":
            entries = plan(source, output, cfg, progress=progress)
        else:
            # Check dependencies before creating the output tree.
            exe = audio.backend(args.ffmpeg)
            backend_version = audio.version(exe)
            if source == output or (source.is_file() and source.parent == output):
                raise ValueError("导出必须使用独立于输入的目标目录")
            with output_lock(output):
                recover(output)
                entries = plan(source, output, cfg, progress=progress)
                for i, entry in enumerate(entries, 1):
                    if entry.status in READY:
                        print(f"[{i}/{len(entries)}] {entry.status}: {entry.target}", file=sys.stderr, flush=True)
                        execute(entry, output, cfg, exe, backend_version)
                    display([entry], False, summary=False, stream=sys.stderr if args.json else sys.stdout)
                report = state_dir(output) / "reports" / f"{uuid.uuid4().hex}.json"
                atomic_json(report, {"created_at": datetime.now(timezone.utc).isoformat(),
                                     "config": cfg, "entries": [e.public() for e in entries]})
                print(f"报告：{report}", file=sys.stderr)
        if args.command == "preview" or args.json:
            display(entries, args.json)
        else:
            display_summary(entries, sys.stdout)
        acceptable = OK | READY | {"unpaired_lrc"}
        return 1 if any(e.status not in acceptable for e in entries) else 0
    except KeyboardInterrupt:
        print("已中断；源文件未修改。下次导出会核对未完成记录。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
