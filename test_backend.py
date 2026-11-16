
from pathlib import Path

from predictor import (
    load_predictor,
    preprocess_image,
    predict_image
)

# Folder testing
TEST_DIR = Path(__file__).parent / "test_images"

# Load model
print("Loading model...")
model, classes = load_predictor()

print("Model berhasil dimuat!")
print("Jumlah kelas:", len(classes))

# Ambil semua gambar
image_files = sorted(
    TEST_DIR.glob("*"),
    key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem
)

if not image_files:
    raise FileNotFoundError("Tidak ada gambar di folder test_images!")

print(f"\nJumlah gambar: {len(image_files)}")

results = []

# Testing semua gambar
for image_path in image_files:

    if not image_path.is_file():
        continue

    try:
        with open(image_path, "rb") as file:
            image_bytes = file.read()

        original, processed = preprocess_image(image_bytes)

        result = predict_image(
            model,
            classes,
            processed
        )

        results.append(result)

        print("\n------------------------------")
        print("Nama gambar:", image_path.name)
        print("Prediksi:", result["predicted_class"])
        print("Confidence:", f'{result["confidence"]:.2%}')
        print("Confident:", result["is_confident"])

    except Exception as error:
        print(f"\nError pada {image_path.name}: {error}")

# Ringkasan
print("\n========== RINGKASAN TESTING ==========")
print("Total gambar berhasil diproses:", len(results))

if results:
    confident_count = sum(
        result["is_confident"] for result in results
    )

    print("Prediksi melewati threshold:", confident_count)
    print("Prediksi di bawah threshold:", len(results) - confident_count)

print("\nTesting selesai!")

