import fitz  # PyMuPDF

def extract_pdf_pagewise(pdf_path):
    doc = fitz.open(pdf_path)
    items = []
    
    for page in doc:
        text = page.get_text()
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Detect table rows: line contains '%' and digits
            is_table_row = ('%' in line) and any(ch.isdigit() for ch in line)
            
            if is_table_row:
                # Merge consecutive lines that look like table rows
                table_lines = [line]
                j = i + 1
                while j < len(lines) and ('%' in lines[j] and any(ch.isdigit() for ch in lines[j])):
                    table_lines.append(lines[j].strip())
                    j += 1
                merged = " ".join(table_lines)
                items.append(("paragraph", merged))
                i = j
                continue
            
            # Improved heading detection
            is_heading = False
            # Avoid numeric-only lines (like "0.96")
            if not line.replace('.', '').replace(',', '').isdigit():
                if line.isupper() and len(line) > 4:
                    is_heading = True
                elif line.endswith(':'):
                    is_heading = True
                elif line[0].isdigit() and '.' in line[:5] and not line.replace('.', '').isdigit():
                    is_heading = True
                elif line.startswith('#'):
                    is_heading = True
            
            label = "heading" if is_heading else "paragraph"
            items.append((label, line))
            i += 1
    
    return items
