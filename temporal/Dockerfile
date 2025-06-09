# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG NODE_VERSION=20.11.1

#FROM node:${NODE_VERSION}-slim

FROM ubuntu:22.04

# Install Node.js
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Use production node environment by default.
# ENV NODE_ENV production


WORKDIR /app

# Run the application as a non-root user.
# USER node

COPY package*.json ./

# Copy specific directories to the Docker image
COPY src/ ./src/

# Copy eval scripts from parent directory
COPY scripts/ ./scripts/


# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.npm to speed up subsequent builds.
# Leverage a bind mounts to package.json and package-lock.json to avoid having to copy them into
# into this layer.

RUN apt-get update && \
    apt-get install -y docker.io && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN  npm install --omit=dev

# RUN mkdir /usr/src/app/node_modules/.cache && chmod -R 777 /usr/src/app/node_modules/.cache

# Expose the port that the application listens on.
EXPOSE 3800

# Run the application.
CMD ["node", "src/workers/evaluation.js"]


