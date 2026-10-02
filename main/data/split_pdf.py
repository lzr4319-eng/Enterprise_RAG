import os
import sys
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_path, output_dir, chunk_size=20):
    """
    将 PDF 按 chunk_size 页数拆分
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    generated_files = []

    for i in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        end_page = min(i + chunk_size, total_pages)
        
        for page_num in range(i, end_page):
            writer.add_page(reader.pages[page_num])
            
        output_filename = f"{base_name}_part_{i // chunk_size + 1:03d}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "wb") as f:
            writer.write(f)
        
        generated_files.append(output_path)
        print(f"Created chunk: {output_path}")

    return generated_files

if __name__ == "__main__":
    split_pdf(sys.argv[1], sys.argv[2], int(sys.argv[3]))