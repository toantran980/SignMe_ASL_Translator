# 1. Use Node.js base, using Node.js 24 LTS for better stability with Expo
FROM node:24

# Install system libs for TensorFlow.js / Expo / Graphics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    python3 \
    make \
    g++ \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install JS dependencies
COPY package*.json ./
# Prefer a clean CI install, but fall back to a (more lenient) install if lockfile is out of sync
RUN npm ci --legacy-peer-deps --loglevel=warn || npm install --legacy-peer-deps --loglevel=warn
# Install ngrok (used by Expo for tunnel mode) globally so it doesn't prompt interactively
RUN npm install -g @expo/ngrok --loglevel=warn

# Copy JSX code and assets
COPY . .

# Port 8081 is the default for React Native Metro Bundler
EXPOSE 8081

# Port for Expo's internal communication
EXPOSE 19000

# Start the bundler
CMD ["npx", "expo", "start", "--tunnel"]  

