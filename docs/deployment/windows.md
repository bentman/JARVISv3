# Windows Deployment Guide for JARVISv3

This guide provides step-by-step instructions for deploying JARVISv3 on Windows systems.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Processor**: Modern multi-core CPU (x86_64 architecture)
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: SSD with 50GB+ free space for models and dependencies
- **Graphics**: NVIDIA/AMD/Intel GPU with 4GB+ VRAM (recommended for heavy tasks)

### Required Software
- **Python**: 3.11 or later
- **Node.js**: 18 or later
- **Docker Desktop**: For containerized deployment
- **Git**: For source code management

## Installation Steps

### 1. Install Prerequisites

#### Python Installation
1. Download Python 3.11+ from [python.org](https://python.org)
2. Run the installer with "Add Python to PATH" enabled
3. Verify installation:
   ```bash
   python --version
   pip --version
   ```

#### Node.js Installation
1. Download Node.js 18+ from [nodejs.org](https://nodejs.org)
2. Run the installer
3. Verify installation:
   ```bash
   node --version
   npm --version
   ```

#### Docker Desktop Installation
1. Download Docker Desktop from [docker.com](https://docker.com)
2. Install and start Docker Desktop
3. Verify installation:
   ```bash
   docker --version
   docker-compose --version
   ```

### 2. Clone Repository

```bash
git clone <repository-url>
cd JARVISv3
```

### 3. Backend Setup

#### Create Virtual Environment
```bash
cd backend
python -m venv .venv
```

#### Activate Virtual Environment
```bash
# Command Prompt
.venv\Scripts\activate

# PowerShell
.\.venv\Scripts\Activate.ps1
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Frontend Setup

```bash
cd frontend
npm install
```

### 5. Configuration

#### Create Environment File
```bash
cp .env.example .env
```

#### Configure Environment Variables
Edit the `.env` file with your specific settings:

```env
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
FRONTEND_PORT=3000

# Database Configuration
DATABASE_URL=postgresql://JARVISv3:password@localhost:5432/JARVISv3
REDIS_URL=redis://localhost:6379/0

# Budget and Resource Limits
MONTHLY_BUDGET_USD=100.0
TOKEN_BUDGET_PER_WORKFLOW=1000
MAX_CONTEXT_SIZE_BYTES=100000

# Privacy and Security
PRIVACY_LEVEL=medium

# Authentication
JWT_SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-api-key
```

### 6. Database Setup

#### Option 1: Docker Compose (Recommended)
```bash
docker-compose up -d postgres redis
```

#### Option 2: Local Installation
1. Install PostgreSQL and Redis locally
2. Create database and user as specified in `DATABASE_URL`

### 7. Model Setup

#### Automatic Model Download
JARVISv3 will automatically download appropriate models based on your hardware:

```bash
# Start the application to trigger model download
cd backend
python -m uvicorn main:app --reload
```

#### Manual Model Configuration
For advanced users, you can specify models in the configuration:

```env
# Model Configuration
DEFAULT_MODEL=llama-2-7b
MODEL_PATH=./models/
```

### 8. Hardware Optimization

#### GPU Acceleration
For NVIDIA GPUs:
1. Install CUDA Toolkit
2. Install cuDNN
3. Verify GPU detection in JARVISv3 logs

For AMD GPUs:
1. Install ROCm drivers
2. Configure OpenVINO for AMD support

#### NPU Support
For systems with NPUs:
1. Install vendor-specific drivers
2. Configure OpenVINO for NPU acceleration

### 9. Running JARVISv3

#### Development Mode
```bash
# Terminal 1: Start backend
cd backend
python -m uvicorn main:app --reload

# Terminal 2: Start frontend
cd frontend
npm run dev
```

#### Production Mode
```bash
# Using Docker Compose
docker-compose up --build

# Using PM2 (Node.js process manager)
npm install -g pm2
pm2 start ecosystem.config.js
```

### 10. Verification

#### Health Check
Visit `http://localhost:8000/health` to verify the system is running.

#### API Documentation
Visit `http://localhost:8000/api/docs` for interactive API documentation.

#### Frontend Interface
Visit `http://localhost:3000` to access the web interface.

## Troubleshooting

### Common Issues

#### Python Virtual Environment Not Activating
```bash
# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Port Already in Use
```bash
# Check which process is using the port
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

#### Model Download Failures
1. Check internet connection
2. Verify sufficient disk space
3. Check firewall settings
4. Try manual model download from Hugging Face

#### GPU Not Detected
1. Verify GPU drivers are installed
2. Check CUDA/cuDNN installation
3. Review JARVISv3 logs for hardware detection errors

### Performance Optimization

#### Windows-Specific Optimizations
1. **Disable Windows Defender Real-time Protection** during model downloads
2. **Set Power Plan to High Performance**
3. **Enable Windows Subsystem for Linux (WSL2)** for better Docker performance

#### Memory Management
1. **Increase Virtual Memory** if experiencing memory issues
2. **Close unnecessary applications** to free up RAM
3. **Use SSD storage** for faster model loading

### Security Considerations

#### Firewall Configuration
```bash
# Allow JARVISv3 through Windows Firewall
netsh advfirewall firewall add rule name="JARVISv3 Backend" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="JARVISv3 Frontend" dir=in action=allow protocol=TCP localport=3000
```

#### User Permissions
1. Run Docker Desktop as administrator if needed
2. Ensure proper file system permissions for model directories
3. Use strong passwords for database and API keys

## Maintenance

### Updates
```bash
# Pull latest changes
git pull

# Update dependencies
cd backend
pip install -r requirements.txt --upgrade

cd ../frontend
npm install --upgrade
```

### Logs and Monitoring
```bash
# View application logs
docker-compose logs -f

# Monitor system resources
# Use Windows Task Manager or Resource Monitor
```

### Backup and Recovery
1. **Database Backup**:
   ```bash
   pg_dump -h localhost -U JARVISv3 JARVISv3 > backup.sql
   ```

2. **Configuration Backup**:
   ```bash
   # Backup .env files and configuration
   cp .env .env.backup
   ```

3. **Model Backup**:
   ```bash
   # Backup models directory
   cp -r models models_backup
   ```

## Support

For additional support:
- Check the [JARVISv3 Documentation](../README.md)
- Review [Troubleshooting Guide](./troubleshooting.md)
- Join the community discussions
- Report issues on GitHub

## Next Steps

After successful deployment:
1. Configure user accounts and permissions
2. Set up monitoring and alerting
3. Customize workflows for your specific use cases
4. Explore advanced features and integrations

This completes the Windows deployment guide for JARVISv3. The system should now be running and ready for use.
