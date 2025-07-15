import re
from typing import List
from langchain.schema import Document
from pypdf import PdfReader

def extract_structured_documents(pdf_path: str) -> List[Document]:
    reader = PdfReader(pdf_path)
    raw_docs = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        doc = Document(page_content=txt.strip(), metadata={"source": f"page_{i}"})
        raw_docs.append(doc)

    parsed_docs = []
    for doc in raw_docs:
        text = doc.page_content
        tables = re.findall(r"(Clinical Data Table|Side Effects Table|Comparison Table)(.*?)\n\n", text, re.DOTALL)
        for title, table_text in tables:
            rows = [line for line in table_text.split("\n") if line.strip()]
            for r in rows[1:]:
                parsed_docs.append(Document(
                    page_content=f"{title} | {r}",
                    metadata={"source": doc.metadata["source"]}
                ))

        cleaned_text = re.sub(r"(Clinical Data Table|Side Effects Table|Comparison Table)(.*?)\n\n", "", text, flags=re.DOTALL)
        for para in cleaned_text.split("\n\n"):
            if len(para.strip()) > 50:
                parsed_docs.append(Document(
                    page_content=para.strip(),
                    metadata={"source": doc.metadata["source"]}
                ))

    print(f"✅ Loaded {len(parsed_docs)} documents")
    return parsed_docs