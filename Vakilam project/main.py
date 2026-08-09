from pypdf import PdfReader

reader = PdfReader("law.pdf")
text = ""

for page in reader.pages:
    text += page.extract_text()

with open("text_output.txt", "w", encoding="utf-8") as data:
    data.write(text)
