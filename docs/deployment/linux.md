# Linux Deployment Guide for JARVISv3

This guide provides step-by-step instructions for deploying JARVISv3 on Linux systems.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 20.04+, Debian 11+, CentOS 8+, or equivalent
- **Processor**: Modern multi-core CPU (x86_64/ARM64)
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: SSD with 50GB+ free space for models and dependencies
- **Graphics**: NVIDIA/AMD/Intel GPU with 4GB+ VRAM (recommended for heavy tasks)

### Required Software
- **Python**: 3.11 or later
- **Node.js**: 18 or later
- **Docker**: Latest version
- **Git**: For source code management

## Installation Steps

### 1. Install Prerequisites

#### Update System
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y
```

#### Install Python
```bash
# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv python3-pip -y

# CentOS/RHEL
sudo yum install python3.11 python3.11-venv python3-pip -y

# Verify installation
python3 --version
pip3 --version
```

#### Install Node.js
```bash
# Using NodeSource repository (recommended)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

#### Install Docker
```bash
# Install Docker
sudo apt install docker.io docker-compose -y

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER

# Verify installation
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
sudo apt install postgresql postgresql-contrib -y

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres createdb JARVISv3
sudo -u postgres createuser --interactive JARVISv3

# Install Redis
sudo apt install redis-server -y
sudo systemctl start redis-server
sudo systemctl enable redis-server
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

#### GPU Acceleration Setup

##### NVIDIA GPU Setup
```bash
# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install cuDNN
# Download from NVIDIA developer website and install
```

##### AMD GPU Setup
```bash
# Install ROCm
wget https://repo.radeon.com/amdgpu-install/5.6/ubuntu/focal/amdgpu-install_5.6.50600-1_all.deb
sudo dpkg -i amdgpu-install_5.6.50600-1_all.deb
sudo amdgpu-install --usecase=hiplibsdk,rocm,opencl

# Verify installation
rocminfo
```

##### Intel GPU Setup
```bash
# Install Intel oneAPI
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-mkl-devel intel-oneapi-dnn-devel
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

# Using systemd (recommended for production)
sudo cp deployment/linux/jarvisv3.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jarvisv3
sudo systemctl start jarvisv3
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
sudo lsof -i :8000

# Kill the process (replace PID with actual process ID)
sudo kill -9 <PID>
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

#### Linux-Specific Optimizations
1. **Increase File Limits**:
   ```bash
   # Add to /etc/security/limits.conf
   * soft nofile 65536
   * hard nofile 65536
   ```

2. **Optimize Network Settings**:
   ```bash
   # Add to /etc/sysctl.conf
   net.core.somaxconn = 65535
   net.ipv4.tcp_max_syn_backlog = 65535
   net.core.netdev_max_backlog = 5000
   ```

3. **Enable Transparent Huge Pages**:
   ```bash
   echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
   ```

#### Memory Management
1. **Configure Swap**:
   ```bash
   # Create swap file
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Monitor Memory Usage**:
   ```bash
   # Use htop for monitoring
   sudo apt install htop
   htop
   ```

### Security Considerations

#### Firewall Configuration
```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 8000
sudo ufw allow 3000
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --reload
```

#### User Permissions
1. **Create dedicated user** for JARVISv3:
   ```bash
   sudo useradd -m -s /bin/bash jarvisv3
   sudo usermod -aG docker jarvisv3
   ```

2. **Set proper file permissions**:
   ```bash
   sudo chown -R jarvisv3:jarvisv3 /opt/JARVISv3
   sudo chmod -R 755 /opt/JARVISv3
   ```

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
htop
iotop
nvidia-smi  # For NVIDIA GPUs
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

## Distribution-Specific Notes

### Ubuntu/Debian
- Use `apt` package manager
- Default Python is usually sufficient
- Docker installation via official repository

### CentOS/RHEL/Fedora
- Use `yum` or `dnf` package manager
- May need EPEL repository for some packages
- SELinux considerations for production

### Arch Linux
- Use `pacman` package manager
- Rolling release - keep system updated
- AUR packages available for some dependencies

### ARM64 Support
For ARM64 systems (Raspberry Pi 4, AWS Graviton, etc.):
```bash
# Install ARM64-specific dependencies
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Use ARM64-optimized models
# JARVISv3 automatically detects ARM64 architecture
```

## Production Deployment

### Systemd Service
Create `/etc/systemd/system/jarvisv3.service`:

```ini
[Unit]
Description=JARVISv3 Backend Service
After=network.target

[Service]
Type=exec
User=jarvisv3
Group=jarvisv3
WorkingDirectory=/opt/JARVISv3/backend
Environment=PATH=/opt/JARVISv3/backend/.venv/bin
ExecStart=/opt/JARVISv3/backend/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy
Create `/etc/nginx/sites-available/jarvisv3`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### SSL/TLS Configuration
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
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

This completes the Linux deployment guide for JARVISv3. The system should now be running and ready for use.
