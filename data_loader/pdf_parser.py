import pdfplumber
import re
from typing import List
from langchain.schema import Document

def extract_structured_documents(pdf_path: str) -> List[Document]:
    parsed_docs: List[Document] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # 1) Extract full‑page text, for later paragraph splitting
            full_text = page.extract_text() or ""
            
            # 2) Extract tables
            #    pdfplumber identifies tables by detecting ruling lines/grid
            for table in page.find_tables():
                # `table.extract()` gives you a list of rows, each a list of cell‑strings
                rows = table.extract()
                if len(rows) <= 1:
                    continue
                header, *data_rows = rows
                title = header[0]  # or derive your own title logic
                for row in data_rows:
                    # join non‑empty cells
                    row_text = " | ".join(cell for cell in row if cell)
                    parsed_docs.append(Document(
                        page_content=f"{title} | {row_text}",
                        metadata={"source": f"page_{i}"}
                    ))
            
            # 3) Remove table text from the raw string so paragraphs don’t duplicate
            #    (you could also blank out areas by bbox, but simple regex works too)
            #    Here we remove any lines containing table‑style delimiters (e.g. lots of spaces)
            cleaned = "\n".join(
                line for line in full_text.split("\n")
                if not re.match(r".{2,}\s{2,}.+", line)
            )
            
            # 4) Split into paragraphs on double newlines
            for para in cleaned.split("\n\n"):
                p = para.strip()
                if len(p) > 50:
                    parsed_docs.append(Document(
                        page_content=p,
                        metadata={"source": f"page_{i}"}
                    ))
    
    print(f"✅ Loaded {len(parsed_docs)} documents")
    return parsed_docs
