from __future__ import annotations

import http.cookiejar
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .common import extract_zip_atomic, image_files, project_path

USER_AGENT = "synthetic-fruit-pipeline/0.1"
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class DataSource:
    key: str
    drive_id: str
    archive_name: str
    expected_bytes: int
    target_key: str
    minimum_images: int


SOURCES = {
    "prepared": DataSource(
        key="prepared",
        drive_id="1YIBQHP9dPglBPG68jbxD7MYx0SlHK3z_",
        archive_name="prepared-assets.zip",
        expected_bytes=1_544_768_252,
        target_key="assets",
        minimum_images=583,
    ),
    "fruits": DataSource(
        key="fruits",
        drive_id="1EWSv230GBdlouaRArZxv0WFjKhsPNQqT",
        archive_name="fruits-closeup.zip",
        expected_bytes=236_417_183,
        target_key="raw/fruits",
        minimum_images=127,
    ),
    "backgrounds": DataSource(
        key="backgrounds",
        drive_id="13c1s7xFXj08VocUZBS6BhvTcGXjohDZs",
        archive_name="backgrounds2.zip",
        expected_bytes=1_431_673_341,
        target_key="raw/backgrounds",
        minimum_images=228,
    ),
}


def _opener() -> urllib.request.OpenerDirector:
    cookies = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))


def _request(url: str, start: int | None = None) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if start is not None:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def _confirmed_url(source: DataSource, opener: urllib.request.OpenerDirector) -> str:
    base = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {"id": source.drive_id, "export": "download", "authuser": "0"}
    )
    with opener.open(_request(base, 0), timeout=60) as response:
        content_type = response.headers.get_content_type()
        if content_type != "text/html":
            return base
        body = response.read(1 << 20).decode("utf-8", errors="replace")
    fields = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', body))
    if fields.get("id") != source.drive_id or "confirm" not in fields:
        raise RuntimeError(
            f"Google Drive não ofereceu o download público de {source.archive_name}"
        )
    return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        fields
    )


def _remote_total(headers) -> int | None:
    content_range = headers.get("Content-Range", "")
    if "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    content_length = headers.get("Content-Length")
    return int(content_length) if content_length else None


def download_google_drive(source: DataSource, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        size = destination.stat().st_size
        if size == source.expected_bytes:
            print(f"reutilizando arquivo completo: {destination}")
            return
        raise RuntimeError(
            f"arquivo existente tem {size} bytes, esperado {source.expected_bytes}: "
            f"{destination}"
        )

    for attempt in range(1, 4):
        start = partial.stat().st_size if partial.exists() else 0
        opener = _opener()
        try:
            url = _confirmed_url(source, opener)
            with opener.open(_request(url, start), timeout=120) as response:
                content_type = response.headers.get_content_type()
                if content_type == "text/html":
                    raise RuntimeError("Google Drive devolveu HTML em vez do arquivo")
                status = getattr(response, "status", response.getcode())
                total = _remote_total(response.headers)
                if total is not None and total != source.expected_bytes:
                    raise RuntimeError(
                        f"tamanho remoto mudou: {total}, esperado {source.expected_bytes}"
                    )
                append = start > 0 and status == 206
                if start > 0 and not append:
                    print(
                        "servidor não retomou o download; reiniciando o arquivo parcial"
                    )
                    start = 0
                mode = "ab" if append else "wb"
                written = start
                last_report = time.monotonic()
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report >= 5:
                            percent = 100 * written / source.expected_bytes
                            print(
                                f"{source.archive_name}: {written / 1024**3:.2f} GiB "
                                f"({percent:.1f}%)",
                                flush=True,
                            )
                            last_report = time.monotonic()
            if partial.stat().st_size != source.expected_bytes:
                raise RuntimeError(
                    f"download incompleto: {partial.stat().st_size}/"
                    f"{source.expected_bytes} bytes"
                )
            partial.replace(destination)
            return
        except Exception:
            if attempt == 3:
                raise
            print(f"tentativa {attempt}/3 falhou; o download será retomado", flush=True)
            time.sleep(attempt * 2)


def target_for(source: DataSource, pipeline_config: dict) -> Path:
    paths = pipeline_config["paths"]
    if source.key == "prepared":
        return project_path(paths["assets"])
    _, child = source.target_key.split("/", 1)
    return project_path(paths["raw"]) / child


def verify_extracted(source: DataSource, target: Path) -> dict:
    files = image_files(target)
    if len(files) < source.minimum_images:
        raise RuntimeError(
            f"{source.key}: somente {len(files)} imagens em {target}; "
            f"mínimo esperado {source.minimum_images}"
        )
    if source.key == "prepared":
        required = ("backgrounds", "backgrounds_map", "pictures_trimmed")
        missing = [name for name in required if not (target / name).is_dir()]
        if missing:
            raise RuntimeError(f"pacote preparado sem diretórios: {missing}")
    return {"source": source.key, "target": str(target), "images": len(files)}


def _remove_ignored_prepared_assets(source: DataSource, target: Path) -> None:
    if source.key != "prepared":
        return
    ignored = target / "lights"
    if ignored.is_symlink() or ignored.is_file():
        ignored.unlink()
    elif ignored.is_dir():
        shutil.rmtree(ignored)


def obtain_source(
    source: DataSource,
    pipeline_config: dict,
    *,
    force: bool = False,
    keep_archive: bool = False,
    verify_only: bool = False,
) -> dict:
    target = target_for(source, pipeline_config)
    if verify_only:
        return verify_extracted(source, target)
    if target.exists() and not force:
        try:
            _remove_ignored_prepared_assets(source, target)
            report = verify_extracted(source, target)
            print(f"reutilizando dados extraídos: {target}")
            return report
        except RuntimeError as error:
            raise RuntimeError(f"extração existente incompleta: {error}; use --force")

    archive_dir = project_path(pipeline_config["paths"]["archives"])
    archive = archive_dir / source.archive_name
    download_google_drive(source, archive)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"arquivo baixado não é um ZIP válido: {archive}")
    extract_zip_atomic(
        archive,
        target,
        force=force,
        excluded_roots=frozenset({"lights"})
        if source.key == "prepared"
        else frozenset(),
    )
    _remove_ignored_prepared_assets(source, target)
    report = verify_extracted(source, target)
    if not keep_archive:
        archive.unlink()
    return report


def selected_sources(mode: str) -> list[DataSource]:
    if mode == "prepared":
        return [SOURCES["prepared"]]
    if mode == "raw":
        return [SOURCES[name] for name in ("fruits", "backgrounds")]
    if mode == "all":
        return [SOURCES[name] for name in ("prepared", "fruits", "backgrounds")]
    return [SOURCES[mode]]
