# Self-Hosting & Deployment Guide

This guide covers deploying the **packaged minimal release** for production or shared server use.

The provided compose configurations are meant to illustrate service boundaries, dependencies, and runtime expectations rather than prescribe a specific production topology.

> For system requirements and environment variables, see the main [README.md](README.md).
> For development setup, see the main [README.md](README.md) building section.

## Quick Start

1. **Download and extract** `deferred-diffusion-*.tar.gz` from the [releases page](https://github.com/JoeGaffney/deferred-diffusion/releases)

2. **Set environment variables** (see [README.md](README.md#environment-variables))

3. **Change into the directory** containing the `docker-compose.yml` file

4. **Ensure Docker Desktop is installed** on the server

5. **Pull and run the containers**:
   ```bash
   docker compose down
   docker compose pull
   docker compose up -d --no-build
   ```

> **Note**: The packaged release contains pre-built containers and clients. Models will be downloaded automatically on first use or during testing.

## Testing Deployment

Tests are included inside the containers and can be run to verify functionality and download missing models.

### Local Model Tests

Will run the basic tests on most models to pull models and verify basic text-to-image / text-to-video.

```bash
docker compose exec gpu-workers pytest -m "basic" -vs
```

### Testing specific Models / Tasks

```bash
docker compose exec gpu-workers pytest tests/images/local/test_flux_1.py -vs
docker compose exec gpu-workers pytest tests/images/external/test_flux_1_pro.py -vs
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
