#!/usr/bin/env python3
"""Simple Rust repo ingester with test-scope stripping.

Examples:
  python grab.py SauersML/gnomon
  python grab.py SauersML/gnomon src/** --exclude "src/**/benches/**"
  python grab.py /path/to/local/repo src/** --exclude "**/generated/**" -o digest.txt
  python grab.py --self-test
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git/**",
    "target/**",
    "**/target/**",
    "node_modules/**",
    "**/node_modules/**",
    ".venv/**",
    "**/.venv/**",
    "vendor/**",
    "**/vendor/**",
)
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB safety cap


@dataclass(frozen=True)
class SourceInfo:
    kind: str  # "local" or "remote"
    display: str
    path: Path | None = None
    clone_url: str | None = None


def parse_source(source: str) -> SourceInfo:
    p = Path(source).expanduser()
    if p.exists():
        return SourceInfo(kind="local", display=str(p.resolve()), path=p.resolve())

    if source.startswith(("http://", "https://", "git@")):
        clone_url = source
        if source.endswith(".git"):
            display = source[: -len(".git")]
        else:
            display = source
        return SourceInfo(kind="remote", display=display, clone_url=clone_url)

    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source):
        clone_url = f"https://github.com/{source}.git"
        display = f"https://github.com/{source}"
        return SourceInfo(kind="remote", display=display, clone_url=clone_url)

    raise ValueError(
        f"Unsupported source: {source!r}. Use a local path, a Git URL, or owner/repo (GitHub slug)."
    )


def clone_repo_no_checkout(clone_url: str, dest: Path) -> None:
    cmd = ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", clone_url, str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git clone failed: {proc.stderr.strip() or proc.stdout.strip()}")


def _matches_patterns(rel: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    if includes and not any(fnmatch.fnmatch(rel, pat) for pat in includes):
        return False
    if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
        return False
    return True


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    if b"\x00" in data:
        return False
    sample = data[:8192]
    bad = 0
    for b in sample:
        if b in (9, 10, 13):  # tab/newline/carriage return
            continue
        if 32 <= b <= 126:
            continue
        if b >= 128:
            continue
        bad += 1
    return (bad / len(sample)) < 0.10


def _load_text_file(path: Path) -> str | None:
    st = path.stat()
    if st.st_size > MAX_FILE_BYTES:
        return None
    raw = path.read_bytes()
    if not _looks_like_text(raw):
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def find_rust_files(root: Path, includes: Sequence[str], excludes: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for p in root.rglob("*.rs"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if _matches_patterns(rel, includes, excludes):
            found.append(p)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def list_remote_rust_paths(repo_dir: Path, includes: Sequence[str], excludes: Sequence[str]) -> list[str]:
    cmd = ["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", "HEAD"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-tree failed: {proc.stderr.strip() or proc.stdout.strip()}")

    selected: list[str] = []
    for line in proc.stdout.splitlines():
        rel = line.strip()
        if not rel.endswith(".rs"):
            continue
        if _matches_patterns(rel, includes, excludes):
            selected.append(rel)
    return sorted(selected)


def checkout_remote_paths(repo_dir: Path, rel_paths: Sequence[str]) -> None:
    if not rel_paths:
        return
    # Keep argv size safe for very large path lists.
    chunk_size = 200
    for i in range(0, len(rel_paths), chunk_size):
        chunk = list(rel_paths[i : i + chunk_size])
        cmd = ["git", "-C", str(repo_dir), "checkout", "HEAD", "--", *chunk]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"git checkout (selected files) failed: {proc.stderr.strip() or proc.stdout.strip()}")


def _is_attribute_start(line: str) -> bool:
    return bool(re.match(r"^\s*#\[", line))


def _has_inline_code_after_attributes(line: str) -> bool:
    start = line.find("#[")
    if start < 0:
        return False
    depth = 0
    seen = False
    end_idx = -1
    for idx in range(start, len(line)):
        ch = line[idx]
        if ch == "[":
            depth += 1
            seen = True
        elif ch == "]" and seen:
            depth -= 1
            if depth == 0:
                end_idx = idx
    if end_idx < 0:
        return False
    remainder = re.sub(r"//.*", "", line[end_idx + 1 :]).strip()
    return bool(remainder)


def _read_attribute_block(lines: Sequence[str], i: int) -> tuple[list[str], int, bool]:
    attrs: list[str] = []
    n = len(lines)
    inline_code = False
    while i < n and _is_attribute_start(lines[i]):
        bracket_balance = 0
        started = False
        while i < n:
            line = lines[i]
            attrs.append(line)
            for ch in line:
                if ch == "[":
                    bracket_balance += 1
                    started = True
                elif ch == "]" and started:
                    bracket_balance -= 1
            i += 1
            if started and bracket_balance <= 0:
                inline_code = _has_inline_code_after_attributes(line)
                break
    return attrs, i, inline_code


def _is_test_attribute(attr_text: str) -> bool:
    norm = re.sub(r"\s+", "", attr_text.lower())
    if "cfg(test" in norm or "cfg(any(test" in norm or "cfg_attr(test" in norm:
        return True
    if re.search(r"(^|[^a-z0-9_])test([^a-z0-9_]|$)", norm):
        return True
    return False


def _strip_strings_and_comments(line: str) -> str:
    line = re.sub(r'//.*', '', line)
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])'", "''", line)
    return line


def _find_first_brace_or_semicolon(lines: Sequence[str], start: int) -> tuple[int, str] | None:
    for idx in range(start, len(lines)):
        clean = _strip_strings_and_comments(lines[idx])
        for ch in clean:
            if ch == "{":
                return idx, "{" 
            if ch == ";":
                return idx, ";"
    return None


def _skip_brace_block(lines: Sequence[str], start: int) -> int:
    balance = 0
    i = start
    seen_open = False
    while i < len(lines):
        clean = _strip_strings_and_comments(lines[i])
        for ch in clean:
            if ch == "{":
                balance += 1
                seen_open = True
            elif ch == "}" and seen_open:
                balance -= 1
                if balance <= 0:
                    return i + 1
        i += 1
    return len(lines)


def _skip_rust_item(lines: Sequence[str], start: int) -> int:
    pos = _find_first_brace_or_semicolon(lines, start)
    if pos is None:
        return len(lines)
    idx, tok = pos
    if tok == ";":
        return idx + 1
    return _skip_brace_block(lines, idx)


def _is_mod_tests_decl(line: str) -> bool:
    return bool(re.match(r"^\s*(pub\s+)?mod\s+tests\s*(\{|;)", line))


def strip_test_scopes(rust_code: str) -> str:
    lines = rust_code.splitlines(keepends=True)
    out: list[str] = []
    i = 0

    while i < len(lines):
        if _is_attribute_start(lines[i]):
            attrs, j, inline_code = _read_attribute_block(lines, i)
            attr_text = "".join(attrs)
            if _is_test_attribute(attr_text):
                skip_start = i if inline_code else j
                i = _skip_rust_item(lines, skip_start)
                continue
            out.extend(attrs)
            i = j
            continue

        if _is_mod_tests_decl(lines[i]):
            if "{" in lines[i]:
                i = _skip_brace_block(lines, i)
            else:
                i += 1
            continue

        out.append(lines[i])
        i += 1

    return "".join(out)


def render_digest(root: Path, files: Sequence[Path]) -> str:
    header = f"# Rust Digest\n# Source root: {root}\n# Files: {len(files)}\n\n"
    parts = [header]
    rendered = 0
    for f in files:
        rel = f.relative_to(root).as_posix()
        raw = _load_text_file(f)
        if raw is None:
            continue
        cleaned = strip_test_scopes(raw).rstrip()
        if not cleaned:
            continue
        rendered += 1
        parts.append(f"===== {rel} =====\n{cleaned}\n\n")
    parts[0] = f"# Rust Digest\n# Source root: {root}\n# Files: {len(files)}\n# Rendered: {rendered}\n\n"
    return "".join(parts).rstrip() + "\n"


def ingest(source: str, includes: Sequence[str], excludes: Sequence[str]) -> str:
    src = parse_source(source)
    merged_excludes = [*DEFAULT_EXCLUDES, *excludes]
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    try:
        if src.kind == "local":
            root = src.path
            if root is None:
                raise RuntimeError("Local source path was not resolved.")
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="rust_ingest_")
            root = Path(temp_dir.name) / "repo"
            assert src.clone_url is not None
            clone_repo_no_checkout(src.clone_url, root)
            remote_paths = list_remote_rust_paths(root, includes, merged_excludes)
            checkout_remote_paths(root, remote_paths)
            files = [root / p for p in remote_paths]
            return render_digest(root, files)

        files = find_rust_files(root, includes, merged_excludes)
        return render_digest(root, files)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def open_file_with_default_app(path: Path) -> None:
    if sys.platform == "darwin":
        proc = subprocess.run(["open", str(path)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "open failed")
        return

    if os.name == "nt":
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc
        return

    proc = subprocess.run(["xdg-open", str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "xdg-open failed")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest Rust files from a repo/path into one text digest, removing test-scoped code."
    )
    p.add_argument("source", nargs="?", help="Local path, Git URL, or GitHub slug owner/repo")
    p.add_argument("include", nargs="*", help="Glob include patterns (default: **/*.rs)")
    p.add_argument("--exclude", action="append", default=[], help="Glob pattern to exclude (repeatable)")
    p.add_argument("-o", "--output", help="Write output to file instead of stdout")
    p.add_argument("--stdout", action="store_true", help="Print digest to stdout (skip file auto-open)")
    p.add_argument("--no-open", action="store_true", help="Do not open output file after writing")
    p.add_argument("--self-test", action="store_true", help="Run built-in tests and exit")
    return p


class _SelfTests(unittest.TestCase):
    def test_cfg_test_module_removed(self) -> None:
        src = textwrap.dedent(
            """
            fn keep() {}

            #[cfg(test)]
            mod tests {
                #[test]
                fn a() { assert_eq!(1, 1); }
            }

            fn also_keep() {}
            """
        )
        out = strip_test_scopes(src)
        self.assertIn("fn keep()", out)
        self.assertIn("fn also_keep()", out)
        self.assertNotIn("mod tests", out)
        self.assertNotIn("fn a()", out)

    def test_test_attr_function_removed(self) -> None:
        src = textwrap.dedent(
            """
            #[test]
            fn tiny_test() {
                let s = "{}";
                assert!(!s.is_empty());
            }
            fn keep() {}
            """
        )
        out = strip_test_scopes(src)
        self.assertIn("fn keep()", out)
        self.assertNotIn("tiny_test", out)

    def test_mod_tests_removed_without_cfg(self) -> None:
        src = textwrap.dedent(
            """
            mod tests {
                fn helper() {}
            }
            pub fn run() {}
            """
        )
        out = strip_test_scopes(src)
        self.assertIn("pub fn run", out)
        self.assertNotIn("helper", out)

    def test_inline_test_attribute_removes_entire_item_only(self) -> None:
        src = textwrap.dedent(
            """
            #[test] fn gone() { assert_eq!(2, 2); }
            fn keep() {}
            """
        )
        out = strip_test_scopes(src)
        self.assertIn("fn keep()", out)
        self.assertNotIn("fn gone()", out)
        self.assertNotIn("assert_eq!(2, 2)", out)

    def test_cfg_test_use_line_removed(self) -> None:
        src = textwrap.dedent(
            """
            #[cfg(test)] use super::*;
            pub fn keep() {}
            """
        )
        out = strip_test_scopes(src)
        self.assertIn("pub fn keep()", out)
        self.assertNotIn("use super::*", out)

    def test_find_rust_files_with_include_exclude(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            (root / "src" / "a.rs").write_text("pub fn a(){}\n", encoding="utf-8")
            (root / "src" / "a_test.rs").write_text("pub fn t(){}\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "it.rs").write_text("fn it(){}\n", encoding="utf-8")

            found = find_rust_files(root, ["src/**", "tests/**"], ["**/*_test.rs", "tests/**"])
            rels = [p.relative_to(root).as_posix() for p in found]
            self.assertEqual(rels, ["src/a.rs"])

    def test_parse_source_slug(self) -> None:
        src = parse_source("SauersML/gnomon")
        self.assertEqual(src.kind, "remote")
        self.assertEqual(src.clone_url, "https://github.com/SauersML/gnomon.git")

    def test_default_exclude_target_applies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            (root / "src" / "keep.rs").write_text("pub fn ok(){}\n", encoding="utf-8")
            (root / "target").mkdir()
            (root / "target" / "drop.rs").write_text("pub fn nope(){}\n", encoding="utf-8")
            out = ingest(str(root), ["**/*.rs"], [])
            self.assertIn("src/keep.rs", out)
            self.assertNotIn("target/drop.rs", out)

    def test_non_text_file_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            p = root / "src" / "weird.rs"
            p.write_bytes(b"fn ok(){}\n\x00\x01\x02")
            out = ingest(str(root), ["**/*.rs"], [])
            self.assertNotIn("src/weird.rs", out)


def _run_self_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_SelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_tests()

    if not args.source:
        parser.error("source is required unless --self-test is used")

    try:
        digest = ingest(args.source, args.include, args.exclude)
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.stdout:
        try:
            sys.stdout.write(digest)
        except BrokenPipeError:  # pragma: no cover - shell piping behavior
            return 0
        return 0

    output_path = Path(args.output) if args.output else (Path.cwd() / "rust_digest.txt")
    output_path.write_text(digest, encoding="utf-8")

    if not args.no_open:
        try:
            open_file_with_default_app(output_path)
        except Exception as exc:  # pragma: no cover - platform/shell dependent
            print(f"warning: wrote {output_path} but could not open automatically: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
