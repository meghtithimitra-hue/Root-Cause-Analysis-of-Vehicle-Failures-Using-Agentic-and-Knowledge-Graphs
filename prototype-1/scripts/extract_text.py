import fitz

pdf_path = "data/automotive.pdf"

doc = fitz.open(pdf_path)

full_text = ""

for page in doc:
    full_text += page.get_text()

with open("graph/full_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Text extracted successfully!")