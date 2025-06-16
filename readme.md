# PDF to Markdown Converter

A complete solution for converting PDF documents to Markdown format using the Marker library. Consists of a FastAPI backend server and a Streamlit web interface.

## Components

### Backend (FastAPI Server)
- Handles PDF processing and conversion
- Supports multiple output formats (Markdown, JSON, HTML)
- OCR capabilities with configurable options
- Image extraction and encoding
- LLM integration for enhanced processing

### Frontend (Streamlit App)
- User-friendly web interface
- File upload and download functionality
- Real-time conversion preview
- Configurable processing parameters

## Features

- Upload PDF files through a web interface
- Convert PDFs to Markdown with OCR support
- Configurable conversion settings
- Preview converted Markdown in rendered and raw formats
- Download converted files
- Extract and handle images from PDFs
- Page range selection for partial conversion
- LLM-powered processing for higher quality results

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn (ASGI server)
- Streamlit
- Requests
- Pillow (PIL)
- Marker library
- Click (for CLI)

## Installation

1. Install required packages:
```bash
pip install fastapi uvicorn streamlit requests pillow marker-pdf click
```

2. Update the `GEMINI_API_KEY` variable in the Streamlit app with your Google Gemini API key

## Usage

### Starting the Backend Server

1. Run the FastAPI server:
```bash
uvicorn backend:app --reload --port=8001                                                             ─╯

```

### Starting the Frontend App

1. Ensure the backend server is running

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to the displayed URL (usually http://localhost:8501)

4. Upload a PDF file using the file uploader

5. Configure conversion settings in the sidebar (optional)

6. Click "Convert to Markdown" to process the file

7. View the results and download the converted Markdown file

## Configuration Options

### OCR Settings
- **Force OCR**: Process all pages with OCR
- **Strip existing OCR**: Remove existing OCR text and re-process

### Processing Settings
- **Use LLM**: Enable higher quality processing with language models
- **Format lines**: Apply OCR formatting to document lines
- **Disable math**: Skip mathematical content in OCR output

### Other Options
- **Page range**: Specify which pages to convert (e.g., "0,5-10,20")
- **Debug mode**: Show additional processing information

## Output

The application provides:
- Rendered Markdown preview
- Raw Markdown text view
- Downloadable .md file
- Conversion statistics (characters, words, lines)
- Extracted images saved to output directory

## API Dependencies

- Marker FastAPI server running on localhost:8001
- Google Gemini API for enhanced processing (when LLM option is enabled)

## File Structure

After conversion, files are saved to an `output` directory:
- `output.md`: The converted Markdown file
- Image files: Extracted images in JPEG format

