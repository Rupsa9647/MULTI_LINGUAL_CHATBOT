import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import re
import os
import json
import uuid
from pathlib import Path
from langdetect import detect, DetectorFactory

# ---------------- CONFIG ----------------
DetectorFactory.seed = 42
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MAX_TOKENS = 320    # target tokens per chunk (~words)
MIN_TOKENS = 50    # minimum tokens per chunk


# ---------------- HELPERS ----------------
def ocr_image_to_text(image_path, langs="eng+ben+hin+urd+chi_sim+chi_tra"):
    """Run OCR on an image and return extracted text."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=langs)
    return text


def clean_text(text: str) -> str:
    """Remove unwanted spaces, zero-width chars, etc."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u200c', '').replace('\u200b', '')
    return text.strip()


def count_tokens(text: str) -> int:
    """Rough token count (approx by word count)."""
    return len(re.findall(r'\w+', text))


def detect_language_hint(text: str) -> str:
    """Detect language of a text chunk."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


# ---------------- CHUNKING ----------------
def multilingual_chunk(text, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS):
    """Split multilingual text into balanced chunks."""
    text = re.sub(r'\r\n', '\n', text)
    paragraphs = [p.strip() for p in re.split(r'\n{1,}', text) if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para_tokens = count_tokens(para)
        current_tokens = count_tokens(current_chunk)

        if current_tokens + para_tokens <= max_tokens:
            current_chunk = (current_chunk + " " + para).strip() if current_chunk else para
        else:
            if current_tokens >= min_tokens:
                chunks.append(clean_text(current_chunk))
            if para_tokens > max_tokens:
                sentences = re.split(r'(?<=[।.!?])\s+', para)
                temp_chunk = ""
                for sent in sentences:
                    sent_tokens = count_tokens(sent)
                    if count_tokens(temp_chunk) + sent_tokens <= max_tokens:
                        temp_chunk = (temp_chunk + " " + sent).strip() if temp_chunk else sent
                    else:
                        if count_tokens(temp_chunk) >= min_tokens:
                            chunks.append(clean_text(temp_chunk))
                        temp_chunk = sent
                if temp_chunk and count_tokens(temp_chunk) >= min_tokens:
                    chunks.append(clean_text(temp_chunk))
                current_chunk = ""
            else:
                current_chunk = para

    if current_chunk and count_tokens(current_chunk) >= min_tokens:
        chunks.append(clean_text(current_chunk))

    return chunks


# ---------------- STORAGE ----------------
def save_chunk_to_disk(output_dir: Path, pdf_path: Path, page_num: int, chunk_num: int, text: str):
    """Save a single text chunk + metadata as JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_id = f"page{page_num}_chunk{chunk_num}_{uuid.uuid4().hex[:6]}"

    metadata = {
        "id": chunk_id,
        "page": page_num,
        "chunk_num": chunk_num,
        "word_count": len(text.split()),
        "char_count": len(text),
        "language_hint": detect_language_hint(text),
        "source_file": str(pdf_path)
    }

    out_path = output_dir / f"{chunk_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "text": text}, f, ensure_ascii=False, indent=2)

    return str(out_path)


# ---------------- MAIN PIPELINE ----------------
def process_pdf(pdf_path: str, output_base_dir: str = "data/chunks"):
    """
    Process a PDF:
    - Extract text from pages
    - Perform OCR if needed
    - Chunk and save all chunks as JSON
    Returns: (list_of_chunk_paths, chunk_output_dir)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌ PDF not found: {pdf_path}")

    output_dir = Path(output_base_dir) / pdf_path.stem
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    all_chunk_paths = []

    for page_num, page in enumerate(doc, 1):
        print(f"🔹 Processing page {page_num}/{len(doc)}...")
        text = page.get_text("text")

        # fallback to OCR if text missing or too short
        if len(text.strip()) < 10:
            print(f"⚠️ Using OCR for page {page_num}")
            pix = page.get_pixmap()
            temp_img = output_dir / f"page_{page_num}.png"
            pix.save(temp_img)
            text = ocr_image_to_text(temp_img)
            temp_img.unlink(missing_ok=True)

        chunks = multilingual_chunk(text)

        for chunk_num, chunk_text in enumerate(chunks, 1):
            chunk_path = save_chunk_to_disk(output_dir, pdf_path, page_num, chunk_num, chunk_text)
            all_chunk_paths.append(chunk_path)

    print(f"✅ Completed: {len(all_chunk_paths)} chunks saved in {output_dir}")
    return all_chunk_paths, str(output_dir)


# ---------------- RUNNER ----------------
if __name__ == "__main__":
    pdf_file = input("📄 Enter PDF path: ").strip()
    try:
        chunk_files, chunk_dir = process_pdf(pdf_file)
        print(f"\n✅ PDF processed successfully!")
        print(f"Chunks saved in: {chunk_dir}")
        print(f"Total chunks: {len(chunk_files)}")

        # 👉 Now chunk_dir can be passed directly to embedding stage
        # Example:
        # embed_chunks_from_dir(chunk_dir)

    except Exception as e:
        print(f"❌ Error: {e}")
