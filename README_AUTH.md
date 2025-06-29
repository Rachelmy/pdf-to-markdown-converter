# PDF Converter Pro - Authentication System

This document describes the authentication system that has been added to the PDF Converter Pro application.

## Features

### 1. User Authentication
- **Login System**: Users must log in with username and password to access the application
- **Registration**: New users can create accounts with username, password, and optional email
- **JWT Tokens**: Secure authentication using JSON Web Tokens
- **Session Management**: Automatic token validation and session handling

### 2. Database Storage
- **PostgreSQL Database**: User credentials and login attempts are stored in PostgreSQL
- **Docker Integration**: Database runs in a Docker container for easy setup
- **Secure Password Hashing**: Passwords are hashed using bcrypt before storage

### 3. Login Attempt Tracking
- **Comprehensive Logging**: All login attempts (successful and failed) are recorded
- **IP Address Tracking**: Records the IP address of login attempts
- **User Agent Logging**: Stores browser/client information
- **Failure Reason Tracking**: Records why login attempts failed

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);
```

### Login Attempts Table
```sql
CREATE TABLE login_attempts (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    failure_reason VARCHAR(255)
);
```

## Setup Instructions

### 1. Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ installed
- pip package manager

### 2. Quick Start
```bash
# Make the startup script executable
chmod +x start.sh

# Run the startup script
./start.sh
```

### 3. Manual Setup
```bash
# Start PostgreSQL database
docker-compose up -d postgres

# Install Python dependencies
pip install -r requirements.txt

# Start the application
python backend.py
```

## Usage

### 1. Access the Application
- Open your browser and go to `http://localhost:8000`
- You will be redirected to the login page

### 2. Registration
- Click "Don't have an account? Register here"
- Fill in the registration form:
  - Username (required)
  - Email (optional)
  - Password (minimum 6 characters)
  - Confirm Password
- Click "Register"

### 3. Login
- Enter your username and password
- Click "Login"
- Upon successful login, you'll be redirected to the main application

### 4. Using the Application
- The main PDF conversion interface is now protected behind authentication
- Your username is displayed in the top-right corner
- Use the "Logout" button to sign out

## API Endpoints

### Authentication Endpoints
- `POST /auth/login` - User login
- `POST /auth/register` - User registration

### Protected Endpoints
- `GET /app` - Main application page (requires authentication)
- `POST /marker` - PDF conversion API (requires authentication)
- `POST /marker/upload` - PDF upload and conversion (requires authentication)

### Public Endpoints
- `GET /` - Redirects to login page
- `GET /login` - Login page

## Security Features

### Password Security
- Passwords are hashed using bcrypt
- Minimum password length of 6 characters
- Secure password verification

### Token Security
- JWT tokens with 30-minute expiration
- Secure token validation
- Automatic token refresh handling

### Database Security
- PostgreSQL with proper indexing
- Prepared statements to prevent SQL injection
- Secure connection handling

## Monitoring and Logging

### Login Attempt Monitoring
All login attempts are logged with the following information:
- Username attempted
- IP address
- User agent (browser/client info)
- Success/failure status
- Timestamp
- Failure reason (if applicable)

### Application Logging
- Comprehensive logging of all operations
- Error tracking and debugging information
- User activity monitoring

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure Docker is running
   - Check if PostgreSQL container is started: `docker-compose ps`
   - Restart the database: `docker-compose restart postgres`

2. **Authentication Token Expired**
   - Tokens expire after 30 minutes
   - Simply log in again to get a new token

3. **Port Already in Use**
   - Change the port in `backend.py` or stop other services using port 8000

### Database Management

**View Login Attempts:**
```sql
SELECT * FROM login_attempts ORDER BY timestamp DESC LIMIT 10;
```

**View Users:**
```sql
SELECT username, email, created_at, is_active FROM users;
```

**Reset Database:**
```bash
docker-compose down
docker volume rm convert_markdown_postgres_data
docker-compose up -d postgres
```

## Development

### Adding New Protected Endpoints
To protect a new endpoint, add the authentication dependency:

```python
from fastapi import Depends
from auth import get_current_user_dependency

@app.get("/protected-endpoint")
async def protected_endpoint(current_user: User = Depends(get_current_user_dependency)):
    return {"message": f"Hello {current_user.username}!"}
```

### Customizing Authentication
- Modify `auth.py` for custom authentication logic
- Update `database.py` for schema changes
- Customize token expiration in `auth.py`

## Production Deployment

### Security Considerations
1. Change the `SECRET_KEY` in `auth.py`
2. Use environment variables for database credentials
3. Enable HTTPS in production
4. Set up proper firewall rules
5. Regular database backups
6. Monitor login attempts for security threats

### Environment Variables
```bash
export DATABASE_URL="postgresql://user:password@host:port/database"
export SECRET_KEY="your-secure-secret-key"
```

## Support

For issues or questions:
1. Check the application logs
2. Review the database for login attempt patterns
3. Ensure all dependencies are properly installed
4. Verify Docker and PostgreSQL are running correctly 