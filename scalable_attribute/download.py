"""Download a published model and verify its frozen file identity."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def verify(path, record):
    if path.stat().st_size != record["size_bytes"]:
        raise ValueError("Model file size mismatch: " + path.name)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError("Model SHA-256 mismatch: " + path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", choices=("512", "1k", "2k", "4k", "8k", "16k", "32k", "all"), required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "configs/checkpoints.json")
    parser.add_argument("--base-url", help="Optional approved mirror containing the same verified files")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [r for r in manifest["models"] if args.point in ("all", r["point"])]
    if not records:
        parser.error("Selected model is absent from the manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = args.output / record["filename"]
        if path.exists():
            verify(path, record)
            print("Verified " + str(path))
            continue
        url = (args.base_url.rstrip("/") + "/" + path.name) if args.base_url else record.get("url")
        if not url:
            parser.error("Model assets await release approval. Place the supplied file in " + str(path) + " or provide --base-url.")
        if not url.startswith("https://"):
            parser.error("Model downloads require an HTTPS URL")
        partial = path.with_suffix(path.suffix + ".partial")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            verify(partial, record)
            partial.rename(path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        print("Verified " + str(path))


if __name__ == "__main__":
    main()
