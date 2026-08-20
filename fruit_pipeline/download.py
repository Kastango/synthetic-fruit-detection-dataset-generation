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

from .common import extract_zip_atomic, image_files, project_path, sha256_file

USER_AGENT = "synthetic-fruit-pipeline/0.1"
CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class DataSource:
    key: str
    drive_id: str
    archive_name: str
    expected_bytes: int
    target_key: str | None = None
    minimum_images: int = 0


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
    if source.target_key is None:
        raise ValueError(f"fonte sem destino de extração: {source.key}")
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


def download_real_source(pipeline_config: dict, force: bool = False) -> Path:
    """Baixa e valida o ZIP da base manual configurada."""
    source = pipeline_config["real_dataset"]["source"]
    download_url = str(source["download_url"])
    archive_name = str(source["archive_name"])
    expected_bytes = int(source["expected_bytes"])
    expected_sha256 = str(source["sha256"])
    archive = project_path(pipeline_config["paths"]["archives"]) / archive_name
    if archive.exists() and not force:
        _verify_download(archive, expected_bytes, expected_sha256)
        print(f"reutilizando arquivo privado validado: {archive}")
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    if force:
        archive.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    parsed_url = urllib.parse.urlparse(download_url)
    query = urllib.parse.parse_qs(parsed_url.query)
    drive_id = query.get("id", [None])[0]
    if parsed_url.hostname != "drive.usercontent.google.com" or not drive_id:
        raise ValueError("real_dataset.source.download_url inválida")
    download_google_drive(
        DataSource(
            key="real",
            drive_id=str(drive_id),
            archive_name=archive_name,
            expected_bytes=expected_bytes,
        ),
        archive,
    )

    _verify_download(archive, expected_bytes, expected_sha256)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"arquivo baixado não é um ZIP válido: {archive}")
    return archive


def _verify_download(
    path: Path, expected_bytes: int | None, expected_sha256: str | None
) -> None:
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"tamanho inválido para {path}: {path.stat().st_size}, "
            f"esperado {expected_bytes}"
        )
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"SHA-256 inválido para {path}: {actual}, esperado {expected_sha256}"
            )


def download_http(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Download HTTP retomável, com validação criptográfica opcional."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        _verify_download(destination, expected_bytes, expected_sha256)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        start = partial.stat().st_size if partial.exists() else 0
        try:
            with urllib.request.urlopen(_request(url, start), timeout=120) as response:
                content_type = response.headers.get_content_type()
                if content_type == "text/html":
                    raise RuntimeError("servidor devolveu HTML em vez do arquivo")
                status = getattr(response, "status", response.getcode())
                append = start > 0 and status == 206
                if start and not append:
                    start = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    shutil.copyfileobj(response, handle, CHUNK_BYTES)
            _verify_download(partial, expected_bytes, expected_sha256)
            partial.replace(destination)
            return destination
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt * 2)
    raise AssertionError("laço de tentativas terminou sem resultado")


def obtain_external_archive(
    pipeline_config: dict,
    name: str,
    *,
    source: Path | None = None,
    force: bool = False,
) -> Path:
    """Obtém e valida o arquivo de um teste externo configurado."""
    datasets = pipeline_config.get("external_datasets", {})
    if name not in datasets:
        raise KeyError(f"dataset externo não configurado: {name}")
    config = datasets[name]
    destination = (
        project_path(pipeline_config["paths"]["archives"])
        / "external"
        / str(config["archive_name"])
    )
    expected_bytes = config.get("expected_bytes")
    expected_sha256 = config.get("sha256")
    if source is not None:
        source = source.resolve()
        _verify_download(source, expected_bytes, expected_sha256)
        if source != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        return destination
    url = config.get("download_url")
    if not url:
        raise RuntimeError(f"{name} não possui URL direta; informe --external-source")
    try:
        return download_http(
            str(url),
            destination,
            expected_bytes=int(expected_bytes) if expected_bytes is not None else None,
            expected_sha256=str(expected_sha256) if expected_sha256 else None,
            force=force,
        )
    except Exception as error:
        raise RuntimeError(
            f"download automático de {name} falhou ({error}); baixe pela página oficial "
            f"{config.get('landing_page')} e informe --external-source"
        ) from error


def selected_sources(mode: str) -> list[DataSource]:
    if mode == "prepared":
        return [SOURCES["prepared"]]
    if mode == "raw":
        return [SOURCES[name] for name in ("fruits", "backgrounds")]
    if mode == "all":
        return [SOURCES[name] for name in ("prepared", "fruits", "backgrounds")]
    return [SOURCES[mode]]
