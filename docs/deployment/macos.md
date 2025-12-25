# macOS Deployment Guide for JARVISv3

This guide provides step-by-step instructions for deploying JARVISv3 on macOS systems.

## Prerequisites

### System Requirements
- **Operating System**: macOS 12 (Monterey) or later
- **Processor**: Apple Silicon (M1/M2/M3) or Intel x86_64
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: SSD with 50GB+ free space for models and dependencies
- **Graphics**: Metal-compatible GPU (recommended for heavy tasks)

### Required Software
- **Python**: 3.11 or later
- **Node.js**: 18 or later
- **Docker Desktop**: For containerized deployment
- **Homebrew**: Package manager for macOS
- **Git**: For source code management

## Installation Steps

### 1. Install Prerequisites

#### Install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Install Python
```bash
# Using Homebrew
brew install python@3.11

# Or download from python.org
# Verify installation
python3 --version
pip3 --version
```

#### Install Node.js
```bash
# Using Homebrew
brew install node

# Or download from nodejs.org
# Verify installation
node --version
npm --version
```

#### Install Docker Desktop
1. Download Docker Desktop for Mac from [docker.com](https://docker.com)
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
python3 -m venv .venv
```

#### Activate Virtual Environment
```bash
source .venv/bin/activate
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
```bash
# Install PostgreSQL
brew install postgresql
brew services start postgresql

# Install Redis
brew install redis
brew services start redis

# Create database
createdb JARVISv3
```

### 7. Model Setup

#### Automatic Model Download
JARVISv3 will automatically download appropriate models based on your hardware:

```bash
# Start the application to trigger model download
cd backend
python3 -m uvicorn main:app --reload
```

#### Manual Model Configuration
For advanced users, you can specify models in the configuration:

```env
# Model Configuration
DEFAULT_MODEL=llama-2-7b
MODEL_PATH=./models/
```

### 8. Hardware Optimization

#### Apple Silicon Optimization
For Apple Silicon Macs (M1/M2/M3):
1. JARVISv3 automatically detects Apple Silicon
2. Uses optimized models for ARM64 architecture
3. Leverages Metal Performance Shaders for GPU acceleration

#### Intel Mac Optimization
For Intel Macs:
1. Uses x86_64 optimized models
2. Leverages Metal for GPU acceleration
3. Automatic fallback to CPU if GPU unavailable

#### GPU Acceleration
```bash
# Verify GPU detection
python3 -c "import GPUtil; print(GPUtil.getGPUs())"
```

### 9. Running JARVISv3

#### Development Mode
```bash
# Terminal 1: Start backend
cd backend
python3 -m uvicorn main:app --reload

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

#### Permission Denied Errors
```bash
# Fix file permissions
chmod +x backend/.venv/bin/activate
chmod +x frontend/node_modules/.bin/*
```

#### Port Already in Use
```bash
# Check which process is using the port
lsof -i :8000

# Kill the process (replace PID with actual process ID)
kill -9 <PID>
```

#### Model Download Failures
1. Check internet connection
2. Verify sufficient disk space
3. Check firewall settings
4. Try manual model download from Hugging Face

#### GPU Not Detected
1. Verify Metal is enabled in System Preferences
2. Check macOS version compatibility
3. Review JARVISv3 logs for hardware detection errors

### Performance Optimization

#### macOS-Specific Optimizations
1. **Disable Gatekeeper** for development (not recommended for production):
   ```bash
   sudo spctl --master-disable
   ```

2. **Increase File Limits**:
   ```bash
   # Add to ~/.zshrc or ~/.bash_profile
   ulimit -n 65536
   ```

3. **Optimize for Apple Silicon**:
   ```bash
   # Use Rosetta for x86_64 compatibility if needed
   arch -x86_64 zsh
   ```

#### Memory Management
1. **Close unnecessary applications** to free up RAM
2. **Use Activity Monitor** to monitor resource usage
3. **Enable swap optimization**:
   ```bash
   sudo sysctl -w vm.swapusage
   ```

### Security Considerations

#### Firewall Configuration
macOS automatically configures firewall for development servers. For production:

```bash
# Check firewall status
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Allow specific applications
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/your/app
```

#### User Permissions
1. **Use dedicated user account** for production deployment
2. **Set proper file permissions**:
   ```bash
   chown -R $USER:$USER JARVISv3/
   chmod -R 755 JARVISv3/
   ```

3. **Use Keychain** for storing sensitive credentials

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
# Use Activity Monitor or top command
top
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

## Apple Silicon Specific Notes

### Native ARM64 Support
- JARVISv3 automatically detects and optimizes for Apple Silicon
- Uses ARM64-optimized models and libraries
- Leverages Metal Performance Shaders for GPU acceleration

### Rosetta 2 Compatibility
If you encounter compatibility issues:
```bash
# Run terminal with Rosetta
arch -x86_64 zsh

# Install dependencies with Rosetta
arch -x86_64 pip install -r requirements.txt
```

### Performance Tips
1. **Use native ARM64 Python** for best performance
2. **Enable Metal acceleration** in JARVISv3 settings
3. **Monitor thermal throttling** during heavy workloads
4. **Use external cooling** if needed for sustained workloads

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

This completes the macOS deployment guide for JARVISv3. The system should now be running and ready for use.
