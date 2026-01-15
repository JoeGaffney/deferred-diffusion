# Deferred Diffusion Clients

These are examples on how to simply get things on the path you could use rez or any other way preferred way to get the modules and plugins loaded.

Adjust directories depending on where you have the folders and the versions of your application. Examples are given for a windows environment.

## Environment Variables

For client applications:

```env
DDIFFUSION_API_ADDRESS=http://127.0.0.1:5000 # API server address
DDIFFUSION_API_KEY=Welcome1! # API key for client authentication
```

## Integration Tests

The `it_tests` directory contains integration tests that verify the API functionality from a client perspective. These tests use the same generated clients and environment variables as the application clients.

## Houdini

### Python Modules

The following need to be available to Houdini for the API client and agents to work.

- httpx

You can install like this to put on roaming path.

```bash
"C:\Program Files\Side Effects Software\Houdini 20.5\bin\hython.exe" -m pip install httpx
```

### Env file

```env
HOUDINI_PATH = C:/development/deferred-diffusion/clients/houdini;&
HOUDINI_OTLSCAN_PATH = C:/development/deferred-diffusion/clients/houdini;&
PYTHONPATH = C:/development/deferred-diffusion/clients/houdini/python;&
```

## Nuke

### Python modules

The following need to be available to Nuke for the API client to work.

- httpx
- attrs

You can install like this.

```bash
"C:\Program Files\Nuke14.0\python.exe" -m pip install httpx attrs
```

### Adding to the path

Update your

- C:\Users\USERNAME\.nuke\init.py

```python
import nuke

# Centralized Nuke plugin path (your custom directory)
custom_plugin_path = r"C:\development\deferred-diffusion\clients\nuke"

# Add your custom plugin paths
nuke.pluginAddPath(custom_plugin_path)
print(f"Custom plugin paths from {custom_plugin_path} have been added.")
```

## Agentic Layer

Example how to call the api through an pydantic-ai agent using MCP and ag-ui protocol for the ui
