from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
import time
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
import pdfplumber
from openai import OpenAI
import numpy as np
import faiss
import pickle
import re
import uuid

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

app = FastAPI(title="PDF to FAISS Embedding API", version="1.0.0")

# OpenAI client
client = OpenAI()

# Constants
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_MAX_INPUT_TOKENS = 20000
DEFAULT_MAX_OUTPUT_TOKENS = 5000
DEFAULT_BATCH_PAGES = 2
TOKENS_PER_CHAR = 0.25
RETRY_LIMIT = 3
RETRY_DELAY_BASE = 2.0

class PDFProcessRequest(BaseModel):
    pdf_path: str
    model: str = DEFAULT_MODEL
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    batch_pages: int = DEFAULT_BATCH_PAGES

class PDFProcessResponse(BaseModel):
    message: str
    chunks_created: int
    table_count: int
    text_count: int
    download_url: str

def estimate_tokens(text): 
    return int(len(text) * TOKENS_PER_CHAR)

def extract_pages(pdf_path: str) -> Dict[int, str]:
    """Extract pages from PDF using pdfplumber"""
    pages = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text(layout=True) or ""
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table and len(table) > 1:
                        md_table = "\n".join([
                            "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |" 
                            for row in table
                        ])
                        text += "\n\n" + md_table
            pages[page_num] = text
    return pages

def batch_text(batch_pages: List[Tuple[int, str]]) -> str:
    return "".join([
        f"\n\n--- PAGE {p} START ---\n{txt}\n--- PAGE {p} END ---\n" 
        for p, txt in batch_pages
    ])

def sys_prompt() -> str:
    return (
        "You are a precise PDF-to-Markdown converter. Rules:\n"
        "- Output ONLY Markdown\n"
        "- Preserve wording & punctuation\n"
        "- Rejoin broken lines/hyphens\n"
        "- Remove headers/footers/page numbers\n"
        "- Maintain reading order\n"
        "- Preserve headings, lists, tables\n"
        "- Convert tables to Markdown tables\n"
        "- Do NOT add or summarize content"
    )

def usr_prompt(text: str) -> str:
    return f"Convert the following PDF content to Markdown:\n\n{text}\n"

def gpt_call_retry(model, system_prompt, user_prompt, max_output_tokens):
    delay = RETRY_DELAY_BASE
    for attempt in range(RETRY_LIMIT):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=max_output_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == RETRY_LIMIT - 1:
                raise
            logger.warning(f"GPT call failed ({e}). Retrying in {delay:.1f}s…")
            time.sleep(delay)
            delay *= 2

def plan_batches(pages, max_input_tokens, start_size):
    base = estimate_tokens(sys_prompt())
    items, i, batches = list(pages.items()), 0, []
    
    while i < len(items):
        size = start_size
        while size > 0:
            candidate = items[i:i+size]
            toks = base + estimate_tokens(usr_prompt(batch_text(candidate)))
            if toks <= max_input_tokens:
                batches.append(candidate)
                i += size
                break
            size -= 1
        if size == 0:
            batches.append(items[i:i+1])
            i += 1
    return batches

def pdf_to_markdown(pdf_path: str, model: str, max_in: int, max_out: int, batch_pages: int) -> str:
    """Convert PDF to markdown content"""
    logger.info(f"Extracting from PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    logger.info(f"Extracted {len(pages)} pages")
    
    batches = plan_batches(pages, max_in, batch_pages)
    logger.info(f"Created {len(batches)} batches")
    
    sys_msg, parts = sys_prompt(), []
    
    for idx, batch in enumerate(batches, 1):
        logger.info(f"Processing batch {idx}/{len(batches)}")
        content = gpt_call_retry(model, sys_msg, usr_prompt(batch_text(batch)), max_out)
        parts.append(content)
    
    return "\n\n".join(parts).strip()

# Markdown chunking functions
def is_table_line(line):
    """Check if line is part of markdown table"""
    line = line.strip()
    return (line.count('|') >= 2 and 
            not line.startswith(('#', '```', '>', '-', '*', '+')))

def extract_tables(text):
    """Find all markdown tables"""
    lines = text.split('\n')
    tables = []
    start = None
    table_lines = []
    
    for i, line in enumerate(lines):
        if is_table_line(line):
            if start is None:
                start = i
            table_lines.append(line)
        else:
            if start is not None and len(table_lines) >= 2:
                tables.append((start, i-1, '\n'.join(table_lines)))
            start = None
            table_lines = []
    
    if start is not None and len(table_lines) >= 2:
        tables.append((start, len(lines)-1, '\n'.join(table_lines)))
    
    return tables

def is_heading(line):
    """Check if line is a markdown heading"""
    return line.strip().startswith('#')

def should_keep_with_next(line, next_lines):
    """Determine if current line should be kept with following content"""
    if is_heading(line):
        return True
    
    if len(line.strip()) < 100 and next_lines:
        for next_line in next_lines[:3]:
            if next_line.strip():
                if not is_heading(next_line):
                    return True
                break
    return False

def get_context_overlap(text, char_limit=200):
    """Get the last char_limit characters from text as context"""
    if not text or len(text) <= char_limit:
        return text
    
    truncated = text[-char_limit:]
    space_index = truncated.find(' ')
    if space_index > 0:
        return truncated[space_index:].strip()
    return truncated.strip()

def chunk_markdown(content, file_name):
    """Improved chunking that preserves context and keeps headings with content"""
    lines = content.split('\n')
    tables = extract_tables(content)
    
    # Mark table lines
    table_line_nums = set()
    for start, end, _ in tables:
        table_line_nums.update(range(start, end + 1))
    
    chunks = []
    current_chunk = []
    previous_text_content = ""
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # If line is part of table, handle table with context
        if i in table_line_nums:
            if current_chunk:
                text = '\n'.join(current_chunk).strip()
                if len(text) > 10:
                    chunks.append({'text': text, 'type': 'text', 'file_name': file_name})
                    previous_text_content = text
                current_chunk = []
            
            # Find and add the complete table with context
            for start, end, table_content in tables:
                if start <= i <= end:
                    context = get_context_overlap(previous_text_content, 200)
                    if context:
                        table_with_context = f"{context}\n\n{table_content}"
                    else:
                        table_with_context = table_content
                    
                    chunks.append({
                        'text': table_with_context,
                        'type': 'table',
                        'file_name': file_name
                    })
                    i = end + 1
                    break
            continue
        
        # Handle regular content
        if line.strip() == '':
            if current_chunk:
                last_line = current_chunk[-1] if current_chunk else ""
                upcoming_lines = lines[i+1:i+4] if i+1 < len(lines) else []
                
                if not should_keep_with_next(last_line, upcoming_lines):
                    text = '\n'.join(current_chunk).strip()
                    if len(text) > 10:
                        chunks.append({'text': text, 'type': 'text', 'file_name': file_name})
                        previous_text_content = text
                    current_chunk = []
                else:
                    current_chunk.append(line)
            else:
                pass
        else:
            current_chunk.append(line)
        
        i += 1
    
    # Add final chunk
    if current_chunk:
        text = '\n'.join(current_chunk).strip()
        if len(text) > 10:
            chunks.append({'text': text, 'type': 'text', 'file_name': file_name})
    
    return chunks

def create_embeddings(chunks, output_dir: Path):
    """Create and store embeddings"""
    texts = [chunk['text'] for chunk in chunks]
    
    response = client.embeddings.create(input=texts, model="text-embedding-3-large")
    embeddings = np.array([v.embedding for v in response.data], dtype=np.float32)
    
    index = faiss.IndexFlatIP(3072)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save to output directory
    faiss_path = output_dir / "faiss_index.bin"
    metadata_path = output_dir / "metadata.pkl"
    
    faiss.write_index(index, str(faiss_path))
    
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)
    
    return faiss_path, metadata_path

@app.post("/process-pdf", response_model=PDFProcessResponse)
async def process_pdf_endpoint(request: PDFProcessRequest):
    """
    Process a PDF file: convert to markdown, create embeddings, and return FAISS index
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {request.pdf_path}")
    
    # Create unique working directory
    work_id = str(uuid.uuid4())
    work_dir = Path(f"temp/work_{work_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Convert PDF to markdown
        logger.info(f"Starting PDF processing for: {request.pdf_path}")
        markdown_content = pdf_to_markdown(
            request.pdf_path, 
            request.model, 
            request.max_input_tokens, 
            request.max_output_tokens, 
            request.batch_pages
        )
        
        # Step 2: Chunk markdown
        file_name = os.path.basename(request.pdf_path)
        chunks = chunk_markdown(markdown_content, file_name)
        
        # Step 3: Create embeddings and save FAISS index
        faiss_path, metadata_path = create_embeddings(chunks, work_dir)
        
        # Step 4: Create zip file with both outputs
        zip_path = work_dir / "faiss_output.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(faiss_path, "faiss_index.bin")
            zipf.write(metadata_path, "metadata.pkl")
        
        # Count statistics
        table_count = sum(1 for c in chunks if c['type'] == 'table')
        text_count = len(chunks) - table_count
        
        logger.info(f"✅ Processing complete. Created {len(chunks)} chunks ({table_count} tables, {text_count} text)")
        
        return PDFProcessResponse(
            message="PDF processed successfully",
            chunks_created=len(chunks),
            table_count=table_count,
            text_count=text_count,
            download_url=f"/download/{work_id}"
        )
        
    except Exception as e:
        # Clean up on error
        if work_dir.exists():
            shutil.rmtree(work_dir)
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/download/{work_id}")
async def download_faiss_files(work_id: str):
    """Download the generated FAISS index and metadata files as a zip"""
    zip_path = Path(f"temp/work_{work_id}/faiss_output.zip")
    
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Files not found or expired")
    
    return FileResponse(
        path=zip_path,
        filename="faiss_output.zip",
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=faiss_output.zip"}
    )

@app.post("/upload-and-process")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it directly
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create unique working directory
    work_id = str(uuid.uuid4())
    work_dir = Path(f"temp/work_{work_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded file
        pdf_path = work_dir / file.filename
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the uploaded file
        request = PDFProcessRequest(pdf_path=str(pdf_path))
        return await process_pdf_endpoint(request)
        
    except Exception as e:
        # Clean up on error
        if work_dir.exists():
            shutil.rmtree(work_dir)
        raise HTTPException(status_code=500, detail=f"Error processing uploaded PDF: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PDF to FAISS API is running"}

# Cleanup task (run periodically to clean temp files)
@app.on_event("startup")
async def startup_event():
    """Create temp directory on startup"""
    Path("temp").mkdir(exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8504)