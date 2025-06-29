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

from fastapi import FastAPI, Form, File, UploadFile, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

# Import our custom modules
from database import create_tables, get_db, User
from auth import (
    authenticate_user, create_user, get_current_user, create_access_token,
    log_login_attempt, get_user_by_username
)

app_data = {}

UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Security scheme
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()
    
    # Create database tables
    create_tables()
    logger.info("Database tables created/verified")

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Pydantic models for authentication
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    email: Optional[str] = None
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

# Authentication endpoints
@app.post("/auth/login", response_model=Token)
async def login(user_data: UserLogin, request: Request, db=Depends(get_db)):
    """Login endpoint"""
    try:
        logger.info(f"Login attempt for user: {user_data.username}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request client: {request.client}")
        
        # Authenticate user
        user = authenticate_user(db, user_data.username, user_data.password)
        
        if not user:
            # Log failed login attempt
            log_login_attempt(db, user_data.username, False, request, "Invalid credentials")
            logger.warning(f"Failed login attempt for user: {user_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Log successful login attempt
        log_login_attempt(db, user_data.username, True, request)
        logger.info(f"Successful login for user: {user_data.username}")
        
        # Create access token
        access_token = create_access_token(data={"sub": user.username})
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            username=user.username
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/auth/register", response_model=dict)
async def register(user_data: UserRegister, request: Request, db=Depends(get_db)):
    """Register endpoint"""
    try:
        logger.info(f"Registration attempt for user: {user_data.username}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request client: {request.client}")
        
        # Check if username already exists
        existing_user = get_user_by_username(db, user_data.username)
        if existing_user:
            logger.warning(f"Registration failed - username already exists: {user_data.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Create new user
        create_user(db, user_data.username, user_data.password, user_data.email)
        logger.info(f"Successfully registered user: {user_data.username}")
        
        return {"message": "User registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Dependency to get current user
async def get_current_user_dependency(credentials: HTTPAuthorizationCredentials = Depends(security), db=Depends(get_db)):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_current_user(credentials.credentials, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@app.get("/")
async def root():
    """Redirect to login page"""
    return HTMLResponse(
        """
        <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/login">
        </head>
        <body>
            <p>Redirecting to login...</p>
        </body>
        </html>
        """
    )

@app.get("/login")
async def login_page():
    """Serve login page"""
    try:
        with open("login.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(html_content)
    except FileNotFoundError:
        return HTMLResponse(
            """
            <h1>Login Page Not Found</h1>
            <p>login.html file not found. Please ensure the file exists in the same directory as the backend.</p>
            """
        )

@app.get("/app")
async def app_page():
    """Serve the main application page (authentication handled by frontend)"""
    try:
        with open("frontend.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(html_content)
    except FileNotFoundError:
        return HTMLResponse(
            """
            <h1>Application Not Found</h1>
            <p>frontend.html file not found. Please ensure the file exists in the same directory as the backend.</p>
            """
        )

@app.get("/auth/verify")
async def verify_auth(current_user: User = Depends(get_current_user_dependency)):
    """Verify if the user is authenticated"""
    return {"authenticated": True, "username": current_user.username}

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
async def convert_pdf(params: CommonParams, current_user: User = Depends(get_current_user_dependency)):
    """Convert PDF endpoint (requires authentication)"""
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
    current_user: User = Depends(get_current_user_dependency)
):
    """Convert PDF upload endpoint (requires authentication)"""
    # Log all received parameters
    logger.info(f"Received upload request from user {current_user.username} with parameters:")
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

if __name__ == "__main__":
    server_cli()
