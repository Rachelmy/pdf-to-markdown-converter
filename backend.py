import traceback
import click
import os
import logging
import sys
from datetime import datetime

# Configure logging
def setup_logging():
    # Create a custom formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Error handler for stderr
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)
    
    return root_logger

# Initialize logger
logger = setup_logging()

import os

from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

import base64
from contextlib import asynccontextmanager
from typing import Optional, Annotated
import io

from fastapi import FastAPI, Form, File, UploadFile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
            example=None,
        ),
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'."
        ),
    ] = "markdown"
    use_llm: Annotated[
        bool,
        Field(
            description="Use LLM for higher quality processing. Defaults to False."
        ),
    ] = False
    gemini_api_key: Annotated[
        Optional[str],
        Field(
            description="API key for Gemini LLM service. Required if use_llm is True."
        ),
    ] = None


async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html"], "Invalid output format"
    try:
        logger.info(f"Starting PDF conversion with params: {params.model_dump()}")
        options = params.model_dump()
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
        logger.info(f"Successfully converted PDF. Extracted {len(images)} images.")
    except Exception as e:
        logger.error(f"Error during PDF conversion: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
        }

    try:
        encoded = {}
        for k, v in images.items():
            byte_stream = io.BytesIO()
            v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
            encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(
                settings.OUTPUT_ENCODING
            )
        logger.info(f"Successfully encoded {len(encoded)} images")
    except Exception as e:
        logger.error(f"Error encoding images: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": f"Error encoding images: {str(e)}",
        }

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    use_llm: Optional[bool] = Form(default=False),
    gemini_api_key: Optional[str] = Form(default=None),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    # Log all received parameters
    logger.info("Received upload request with parameters:")
    logger.info(f"  page_range: {page_range}")
    logger.info(f"  force_ocr: {force_ocr}")
    logger.info(f"  paginate_output: {paginate_output}")
    logger.info(f"  output_format: {output_format}")
    logger.info(f"  use_llm: {use_llm}")
    logger.info(f"  gemini_api_key: {'[PRESENT]' if gemini_api_key else '[NOT PRESENT]'}")
    logger.info(f"  file: {file.filename}")

    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    try:
        with open(upload_path, "wb+") as upload_file:
            file_contents = await file.read()
            upload_file.write(file_contents)
        logger.info(f"Successfully saved uploaded file to: {upload_path}")

        # Convert boolean parameters from string if needed
        force_ocr_bool = force_ocr if isinstance(force_ocr, bool) else force_ocr.lower() == 'true'
        paginate_output_bool = paginate_output if isinstance(paginate_output, bool) else paginate_output.lower() == 'true'
        use_llm_bool = use_llm if isinstance(use_llm, bool) else use_llm.lower() == 'true'

        params = CommonParams(
            filepath=upload_path,
            page_range=page_range,
            force_ocr=force_ocr_bool,
            paginate_output=paginate_output_bool,
            output_format=output_format,
            use_llm=use_llm_bool,
            gemini_api_key=gemini_api_key,
        )
        logger.info(f"Created CommonParams with values: {params.model_dump()}")
        results = await _convert_pdf(params)
        
        # Clean up uploaded file
        os.remove(upload_path)
        logger.info(f"Removed temporary file: {upload_path}")
        
        return results
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        logger.error(traceback.format_exc())
        # Clean up uploaded file in case of error
        if os.path.exists(upload_path):
            os.remove(upload_path)
            logger.info(f"Removed temporary file after error: {upload_path}")
        return {
            "success": False,
            "error": str(e)
        }


@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn
    
    logger.info(f"Starting server on {host}:{port}")
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )
