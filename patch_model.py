"""Buang field quantization_config dari .keras agar bisa dibuka Keras < 3.13."""
import json
import shutil
import zipfile
from pathlib import Path

SRC = Path(__file__).parent / "model" / "resnet_96x1024.keras"
BAK = SRC.with_suffix(".keras.bak")


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k != "quantization_config"}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


if not BAK.exists():
    shutil.copy2(SRC, BAK)

with zipfile.ZipFile(BAK) as zin:
    with zipfile.ZipFile(SRC, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                data = json.dumps(strip(json.loads(data))).encode()
            zout.writestr(item, data)

print("patched:", SRC, SRC.stat().st_size, "bytes")
