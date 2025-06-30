# PDF Converter Pro - AI-Powered Document Processing

A complete, secure solution for converting PDF documents to Markdown, JSON, or HTML format using the Marker library. Features a modern web interface with user authentication, PostgreSQL database backend, and Docker support.

## 🚀 Features

### Core Functionality
- **Multi-format Conversion**: Convert PDFs to Markdown, JSON, or HTML
- **AI-Powered Processing**: Optional LLM integration for enhanced quality
- **OCR Support**: Configurable OCR with force mode for scanned documents
- **Image Extraction**: Automatically extract and encode images from PDFs
- **Page Range Selection**: Convert specific pages or ranges
- **Batch Processing**: Handle multiple files with comprehensive output

### Security & Authentication
- **User Authentication**: Secure login/registration system
- **Session Management**: JWT-based tokens with 8-hour expiration
- **Login Monitoring**: Track all login attempts with detailed analytics
- **Password Security**: Bcrypt hashing for secure password storage
- **Protected Endpoints**: All conversion endpoints require authentication

### User Interface
- **Modern Dark Theme**: Beautiful, responsive design
- **Real-time Preview**: Toggle between rendered and raw output
- **Drag & Drop Upload**: Intuitive file handling
- **Progress Tracking**: Visual progress indicators
- **Download Options**: ZIP files with all content and images
- **Mobile Responsive**: Works seamlessly on all devices

### Admin Features
- **Login Events Dashboard**: Monitor all authentication attempts
- **User Statistics**: Track successful/failed logins and unique users
- **Security Analytics**: IP tracking, user agent logging, failure reasons

## 🏗️ Architecture

### Backend (FastAPI)
- **FastAPI Server**: High-performance async web framework
- **PostgreSQL Database**: Persistent user and login data storage
- **JWT Authentication**: Secure token-based session management
- **CORS Support**: Cross-origin resource sharing enabled
- **Static File Serving**: Serve CSS and other static assets

### Frontend (HTML/JavaScript)
- **Vanilla JavaScript**: No framework dependencies
- **Markdown Rendering**: Client-side markdown parsing with KaTeX math support
- **Session Storage**: Secure token management per browser tab
- **Real-time Updates**: Dynamic content loading and updates

### Database (PostgreSQL)
- **User Management**: Store user accounts and credentials
- **Login Tracking**: Record all authentication attempts
- **Docker Integration**: Containerized database with persistent storage

## 📋 Requirements

### System Requirements
- Python 3.7+
- Docker and Docker Compose
- 4GB+ RAM (for PDF processing)
- 2GB+ disk space

### Python Dependencies
- FastAPI
- Uvicorn (ASGI server)
- SQLAlchemy (database ORM)
- Passlib (password hashing)
- Python-Jose (JWT tokens)
- Psycopg2 (PostgreSQL adapter)
- Marker library (PDF processing)
- Click (CLI interface)

## 🛠️ Installation

### Quick Start with Docker

1. **Clone the repository**:
```bash
git clone <repository-url>
cd convert_markdown
```

2. **Start the application**:
```bash
chmod +x start.sh
./start.sh
```

3. **Access the application**:
- Open your browser to `http://localhost:8000`
- You'll be redirected to the login page
- Register a new account or login with existing credentials

### Manual Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start PostgreSQL database**:
```bash
docker compose up -d postgres
```

3. **Start the FastAPI application**:
```bash
python backend.py
```

4. **Access the application** at `http://localhost:8000`

## 🚀 Usage

### Getting Started

1. **Registration**: Create a new account with username and password
2. **Login**: Authenticate with your credentials
3. **Upload PDF**: Drag and drop or browse for a PDF file
4. **Configure Options**: Set output format, page range, and processing options
5. **Convert**: Click "Convert PDF" to process your document
6. **Download**: Get results as a ZIP file containing all content and images

### Configuration Options

#### Output Formats
- **Markdown**: Standard markdown with LaTeX math support
- **JSON**: Structured data format
- **HTML**: Web-ready HTML output

#### Processing Options
- **Page Range**: Specify pages (e.g., "0,5-10,20")
- **Force OCR**: Enable OCR for all pages (may reduce quality for text-based PDFs)
- **Paginate Output**: Add page separators to output
- **AI Enhancement**: Use Gemini API for higher quality processing

#### AI Enhancement (Optional)
- Requires Google Gemini API key
- Provides enhanced text processing and formatting
- Improves accuracy for complex documents

### Admin Dashboard

Access the login events dashboard at `/login-events` to monitor:
- Total login attempts
- Successful vs failed logins
- Unique users
- IP addresses and user agents
- Failure reasons and timestamps

## 📁 Project Structure

```
convert_markdown/
├── backend.py              # FastAPI application
├── auth.py                 # Authentication logic
├── database.py             # Database models and connection
├── frontend.html           # Main application interface
├── login.html              # Login/registration page
├── login_events.html       # Admin dashboard
├── styles.css              # Application styling
├── start.sh                # Startup script
├── docker-compose.yml      # Docker configuration
├── requirements.txt        # Python dependencies
├── uploads/                # Temporary file storage
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

The application uses the following environment variables:

- `DATABASE_URL`: PostgreSQL connection string (default: `postgresql://postgres:postgres@localhost:5432/pdf_converter`)
- `SECRET_KEY`: JWT secret key (default: `your-secret-key`)
- `ACCESS_TOKEN_EXPIRE_HOURS`: Token expiration time (default: 8)

### Docker Configuration

The `docker-compose.yml` file configures:
- PostgreSQL database with persistent storage
- Database credentials and port mapping
- Volume mounts for data persistence

## 🔒 Security Features

### Authentication
- **Password Hashing**: Bcrypt with salt rounds
- **JWT Tokens**: Secure session management
- **Token Expiration**: 8-hour session limits
- **Session Storage**: Per-tab authentication

### Data Protection
- **Input Validation**: All user inputs validated
- **SQL Injection Protection**: Parameterized queries
- **CORS Configuration**: Controlled cross-origin access
- **Error Handling**: Secure error messages

### Monitoring
- **Login Tracking**: All authentication attempts logged
- **IP Logging**: Track user locations
- **User Agent Logging**: Browser and device information
- **Failure Analysis**: Detailed failure reason tracking

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   - Ensure PostgreSQL container is running: `docker compose ps`
   - Check database logs: `docker compose logs postgres`

2. **Authentication Issues**:
   - Clear browser session storage
   - Check token expiration (8 hours)
   - Verify database is accessible

3. **PDF Processing Errors**:
   - Check file size (50MB limit)
   - Ensure PDF is not corrupted
   - Try different processing options

4. **Port Conflicts**:
   - Change port in `backend.py` or `docker-compose.yml`
   - Check for other services using port 8000

### Logs and Debugging

- **Application Logs**: Check console output when running `python backend.py`
- **Database Logs**: `docker compose logs postgres`
- **Browser Console**: Check for JavaScript errors in browser dev tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Marker Library**: Core PDF processing functionality
- **FastAPI**: Modern web framework
- **PostgreSQL**: Reliable database backend
- **KaTeX**: Math rendering support
- **Font Awesome**: Icon library

---

**PDF Converter Pro** - Transform your documents with AI-powered precision and enterprise-grade security.


