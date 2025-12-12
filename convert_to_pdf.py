"""
Convert the Mathematical Implementation Report from Markdown to PDF
"""
import subprocess
import os

# Try using markdown-pdf first
try:
    from markdown_pdf import MarkdownPdf, Section
    
    print("Converting MATHEMATICAL_IMPLEMENTATION_REPORT.md to PDF...")
    
    # Read the markdown file
    with open('MATHEMATICAL_IMPLEMENTATION_REPORT.md', 'r') as f:
        content = f.read()
    
    # Create PDF
    pdf = MarkdownPdf(toc_level=3)
    pdf.add_section(Section(content, toc=True))
    pdf.meta["title"] = "Mathematical Implementation & Data Analysis Report"
    pdf.meta["author"] = "Arun Munagala"
    
    output_path = 'MATHEMATICAL_IMPLEMENTATION_REPORT.pdf'
    pdf.save(output_path)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ PDF created successfully: {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    
except Exception as e:
    print(f"markdown-pdf failed: {e}")
    print("\nTrying alternative method with pandoc...")
    
    # Try pandoc if available
    try:
        result = subprocess.run([
            'pandoc',
            'MATHEMATICAL_IMPLEMENTATION_REPORT.md',
            '-o', 'MATHEMATICAL_IMPLEMENTATION_REPORT.pdf',
            '--pdf-engine=pdflatex',
            '-V', 'geometry:margin=1in',
            '--toc',
            '--toc-depth=3',
            '-V', 'fontsize=11pt',
            '-V', 'title=Mathematical Implementation & Data Analysis Report',
            '-V', 'author=Arun Munagala',
            '-V', 'date=December 10, 2025'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            file_size = os.path.getsize('MATHEMATICAL_IMPLEMENTATION_REPORT.pdf') / (1024 * 1024)
            print(f"✓ PDF created successfully with pandoc")
            print(f"  File size: {file_size:.2f} MB")
        else:
            print(f"Pandoc error: {result.stderr}")
            print("\nInstalling pandoc...")
            subprocess.run(['brew', 'install', 'pandoc', 'basictex'], check=False)
            print("Please run this script again after pandoc installation completes")
            
    except FileNotFoundError:
        print("Pandoc not found. Trying Python-based conversion...")
        
        # Fallback: Simple conversion using reportlab
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            print("Using reportlab for PDF conversion...")
            
            with open('MATHEMATICAL_IMPLEMENTATION_REPORT.md', 'r') as f:
                content = f.read()
            
            # Create PDF
            doc = SimpleDocTemplate(
                'MATHEMATICAL_IMPLEMENTATION_REPORT.pdf',
                pagesize=letter,
                leftMargin=1*inch,
                rightMargin=1*inch,
                topMargin=1*inch,
                bottomMargin=1*inch
            )
            
            styles = getSampleStyleSheet()
            story = []
            
            # Simple markdown to PDF (basic conversion)
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], styles['Title']))
                    story.append(Spacer(1, 0.2*inch))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['Heading1']))
                    story.append(Spacer(1, 0.1*inch))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], styles['Heading2']))
                elif line.strip():
                    # Escape special characters and add paragraph
                    safe_line = line.replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(safe_line, styles['Normal']))
                else:
                    story.append(Spacer(1, 0.1*inch))
            
            doc.build(story)
            
            file_size = os.path.getsize('MATHEMATICAL_IMPLEMENTATION_REPORT.pdf') / (1024 * 1024)
            print(f"✓ PDF created successfully with reportlab")
            print(f"  File size: {file_size:.2f} MB")
            
        except ImportError:
            print("Installing reportlab...")
            subprocess.run(['pip', 'install', 'reportlab'], check=True)
            print("Please run this script again")
