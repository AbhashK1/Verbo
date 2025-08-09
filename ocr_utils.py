from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

pytesseract.pytesseract.tesseract_cmd = "D:/Tools/Tesseract/tesseract.exe"


def extract_text_from_image(image_path):
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
    return image_paths
