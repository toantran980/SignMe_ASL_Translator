# 1. Use Node.js base, using Node.js 22 LTS for better stability with Expo
FROM node:22

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
RUN npm install --legacy-peer-deps --loglevel verbose

# Copy JSX code and assets
COPY . .

# Port 8081 is the default for React Native Metro Bundler
EXPOSE 8081

# Port for Expo's internal communication
EXPOSE 19000

# Start the bundler
CMD ["npx", "expo", "start", "--tunnel"]  

