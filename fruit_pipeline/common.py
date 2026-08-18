from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"configuração deve ser um objeto YAML: {path}")
    return value


def project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path, length: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    return value if length is None else value[:length]


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def atomic_write_text(path: Path, text: str, *, durable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any, *, durable: bool = True) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        durable=durable,
    )


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def relative_or_absolute(path: Path, base: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_zip_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in archive.infolist():
        pure = PurePosixPath(info.filename)
        unix_mode = info.external_attr >> 16
        is_symlink = (unix_mode & 0o170000) == 0o120000
        if pure.is_absolute() or ".." in pure.parts or not pure.parts or is_symlink:
            raise ValueError(f"entrada insegura no ZIP: {info.filename!r}")
        yield info


def extract_zip_atomic(
    archive_path: Path,
    target: Path,
    force: bool = False,
    *,
    excluded_roots: frozenset[str] = frozenset(),
) -> None:
    """Valida e extrai um ZIP em diretório temporário antes da troca atômica."""
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise FileExistsError(f"destino já existe: {target}; use --force")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.extract-", dir=target.parent)
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                info
                for info in _safe_zip_members(archive)
                if PurePosixPath(info.filename).parts[0] not in excluded_roots
            ]
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f"CRC inválido no ZIP: {bad_member}")
            archive.extractall(temporary, members=members)
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def require_empty_or_force(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"diretório não vazio: {path}; use --force")
