from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

pytesseract.pytesseract.tesseract_cmd = "D:/Tools/Tesseract/tesseract.exe"

'''def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def convert_pdf_to_images(pdf_path, output_folder="data/images"):
    os.makedirs(output_folder, exist_ok=True)
    pages = convert_from_path(pdf_path, poppler_path=r'D:/Tools/poppler-24.08.0/Library/bin')
    image_paths = []
    for i, page in enumerate(pages):
        path = os.path.join(output_folder, f"page_{i}.jpg")
        page.save(path, 'JPEG')
        image_paths.append(path)
    return image_paths'''

pytesseract.pytesseract.tesseract_cmd = r"D:/Tools/Tesseract/tesseract.exe"
POPLER_PATH = r"D:/Tools/poppler-24.08.0/Library/bin"


def image_to_text(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def pdf_to_images(pdf_path, output_folder="data/images"):
    os.makedirs(output_folder, exist_ok=True)
    pages = convert_from_path(pdf_path, poppler_path=POPLER_PATH)
    image_paths = []
    for i, page in enumerate(pages):
        path = os.path.join(output_folder, f"page_{i}.jpg")
        page.save(path, "JPEG")
        image_paths.append(path)
    return image_paths


def extract_text_from_file(path, temp_image_folder="data/images"):
    ext = os.path.splitext(path)[1].lower()
    full_text = ""
    if ext in [".pdf"]:
        images = pdf_to_images(path, output_folder=temp_image_folder)
        for img in images:
            full_text += image_to_text(img) + "\n"
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        full_text = image_to_text(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()
    return full_text
