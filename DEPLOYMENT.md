# Production Deployment Guide

This guide covers deploying the **packaged minimal release** for production use.

> For development setup, see the main [README.md](README.md) building section.

## Quick Start

1. **Download and extract** `deferred-diffusion-*.tar.gz` from the [releases page](https://github.com/JoeGaffney/deferred-diffusion/releases)

2. **Set environment variables** (see [Environment Variables](#environment-variables) section below)

3. **Change into the directory** containing the `docker-compose.yml` file

4. **Ensure Docker Desktop is installed** on the server

5. **Pull and run the containers**:
   ```bash
   docker compose down
   docker compose pull
   docker compose up -d --no-build
   ```

> **Note**: The packaged release contains pre-built containers and clients. Models will be downloaded automatically on first use or during testing.

## System Requirements

- **Storage**: An NVMe drive with **at least 500GB** of available space is recommended
- **GPU**: Nvidia GPU with at least 12GB VRAM. 24GB recommended
- **Environment Variables**: Ensure all required environment variables are set on the host

## Environment Variables

### Server Configuration

Set these on the host where containers run:

```env
OPENAI_API_KEY=your-openai-key # For OpenAI services
RUNWAYML_API_SECRET=your-runway-secret # For RunwayML services
REPLICATE_API_TOKEN=your-replicate-token # For Replicate API access
HF_TOKEN=your-huggingface-token # For Hugging Face model access
DDIFFUSION_API_KEYS=Welcome1!,Welcome2! # API keys for authentication
```

### Client Configuration

Set these where client tools (Houdini, Nuke) are used:

```env
DDIFFUSION_API_ADDRESS=http://127.0.0.1:5000 # API server address
DDIFFUSION_API_KEY=Welcome1! # API key for client authentication
```

## Testing Deployment

Tests are included inside the containers and can be run to verify functionality and download missing models.

### Local Model Tests

Will run the basic tests on most models to pull models and verify basic text-to-image / text-to-video.

```bash
docker compose exec gpu-workers pytest -m "basic" -vs
```

```bash
docker compose exec gpu-workers pytest tests/images/local/test_flux.py -vs
docker compose exec gpu-workers pytest tests/images/local -vs
docker compose exec gpu-workers pytest tests/texts/local -vs
docker compose exec gpu-workers pytest tests/videos/local -vs
```

### External Service Tests

```bash
docker compose exec gpu-workers pytest tests/images/external -vs
docker compose exec gpu-workers pytest tests/texts/external -vs
docker compose exec gpu-workers pytest tests/videos/external -vs
```

## Troubleshooting

### Docker Optimization (Windows)

To optimize volumes and virtual disk after model deletions:

1. **Kill Docker Desktop and related processes**:

   ```bash
   Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
   Stop-Process -Name "com.docker.*" -Force -ErrorAction SilentlyContinue
   Stop-Process -Name "vmmemWSL" -Force -ErrorAction SilentlyContinue
   Stop-Process -Name "wslhost" -Force -ErrorAction SilentlyContinue
   Stop-Process -Name "wsl" -Force -ErrorAction SilentlyContinue
   ```

2. **Ensure Docker and WSL are closed, then optimize**:
   ```bash
   Optimize-VHD -Path "Y:\DOCKER\DockerDesktopWSL\disk\docker_data.vhdx" -Mode Full
   ```
