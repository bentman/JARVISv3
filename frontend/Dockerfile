# Build stage for production frontend
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (including dev dependencies for build)
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Output built files to /frontend for shared volume access
RUN mkdir -p /frontend && cp -r /app/dist/* /frontend/

# Keep container running for volume sharing
CMD ["tail", "-f", "/dev/null"]
