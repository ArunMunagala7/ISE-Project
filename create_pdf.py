"""
Convert Markdown to properly formatted PDF using markdown2 and reportlab
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import re

print("Converting Markdown to PDF...")

# Read markdown
with open('MATHEMATICAL_IMPLEMENTATION_REPORT.md', 'r') as f:
    content = f.read()

# Create PDF
pdf_file = 'MATHEMATICAL_IMPLEMENTATION_REPORT.pdf'
doc = SimpleDocTemplate(
    pdf_file,
    pagesize=letter,
    leftMargin=0.75*inch,
    rightMargin=0.75*inch,
    topMargin=1*inch,
    bottomMargin=0.75*inch
)

# Custom styles
styles = getSampleStyleSheet()

# Title style
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1a1a1a'),
    spaceAfter=30,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

# Heading styles
h1_style = ParagraphStyle(
    'CustomH1',
    parent=styles['Heading1'],
    fontSize=18,
    textColor=colors.HexColor('#2c3e50'),
    spaceBefore=20,
    spaceAfter=12,
    fontName='Helvetica-Bold'
)

h2_style = ParagraphStyle(
    'CustomH2',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#34495e'),
    spaceBefore=16,
    spaceAfter=8,
    fontName='Helvetica-Bold'
)

h3_style = ParagraphStyle(
    'CustomH3',
    parent=styles['Heading3'],
    fontSize=12,
    textColor=colors.HexColor('#7f8c8d'),
    spaceBefore=12,
    spaceAfter=6,
    fontName='Helvetica-Bold'
)

# Body text style
body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontSize=10,
    leading=14,
    alignment=TA_JUSTIFY,
    spaceAfter=6
)

# Code style
code_style = ParagraphStyle(
    'Code',
    parent=styles['Code'],
    fontSize=9,
    leading=11,
    fontName='Courier',
    backColor=colors.HexColor('#f4f4f4'),
    leftIndent=20,
    rightIndent=20,
    spaceBefore=6,
    spaceAfter=6
)

# Build story
story = []

# Process markdown
lines = content.split('\n')
in_code_block = False
code_block = []

i = 0
while i < len(lines):
    line = lines[i]
    
    # Code blocks
    if line.startswith('```'):
        if in_code_block:
            # End code block
            code_text = '\n'.join(code_block)
            # Escape special chars for reportlab
            code_text = code_text.replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(f'<font face="Courier" size="8">{code_text}</font>', code_style))
            story.append(Spacer(1, 0.1*inch))
            code_block = []
            in_code_block = False
        else:
            in_code_block = True
        i += 1
        continue
    
    if in_code_block:
        code_block.append(line)
        i += 1
        continue
    
    # Title (first # line)
    if line.startswith('# ') and i < 5:
        title_text = line[2:].strip()
        story.append(Paragraph(title_text, title_style))
        story.append(Spacer(1, 0.3*inch))
    
    # H1
    elif line.startswith('## '):
        h1_text = line[3:].strip()
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(h1_text, h1_style))
    
    # H2
    elif line.startswith('### '):
        h2_text = line[4:].strip()
        story.append(Paragraph(h2_text, h2_style))
    
    # H3
    elif line.startswith('#### '):
        h3_text = line[5:].strip()
        story.append(Paragraph(h3_text, h3_style))
    
    # Horizontal rule
    elif line.strip() == '---':
        story.append(Spacer(1, 0.15*inch))
        from reportlab.platypus import HRFlowable
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 0.15*inch))
    
    # Bullet points
    elif line.strip().startswith('- ') or line.strip().startswith('* '):
        bullet_text = line.strip()[2:]
        bullet_text = bullet_text.replace('<', '&lt;').replace('>', '&gt;')
        # Check for checkmarks
        if bullet_text.startswith('✅'):
            bullet_text = '<font color="green">✓</font> ' + bullet_text[2:]
        elif bullet_text.startswith('✓'):
            bullet_text = '<font color="green">✓</font> ' + bullet_text[2:]
        elif bullet_text.startswith('⚠'):
            bullet_text = '<font color="orange">!</font> ' + bullet_text[2:]
        story.append(Paragraph('• ' + bullet_text, body_style))
    
    # Numbered lists
    elif re.match(r'^\d+\. ', line.strip()):
        list_text = re.sub(r'^\d+\. ', '', line.strip())
        list_text = list_text.replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(list_text, body_style))
    
    # Regular paragraphs
    elif line.strip():
        # Skip table of contents markers
        if line.strip().startswith('[') or line.strip().startswith('#'):
            i += 1
            continue
        
        # Escape special chars
        text = line.strip().replace('<', '&lt;').replace('>', '&gt;')
        
        # Bold text
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Italic text  
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Inline code
        text = re.sub(r'`(.*?)`', r'<font face="Courier" size="9">\1</font>', text)
        
        story.append(Paragraph(text, body_style))
    
    # Empty lines
    else:
        story.append(Spacer(1, 0.08*inch))
    
    i += 1

# Add metadata
doc.title = "Mathematical Implementation & Data Analysis Report"
doc.author = "Arun Munagala"
doc.subject = "EKF-SLAM Implementation"

# Build PDF
print("Building PDF...")
doc.build(story)

import os
file_size = os.path.getsize(pdf_file) / (1024 * 1024)
print(f"\n✓ PDF created successfully!")
print(f"  File: {pdf_file}")
print(f"  Size: {file_size:.2f} MB")
print(f"  Pages: ~{len(story) // 40} pages")
