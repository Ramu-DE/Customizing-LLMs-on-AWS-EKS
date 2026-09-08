# Module 200 – Strands Agent Architecture

> Deploys a FastAPI-based AI agent using the Strands Agents SDK that connects to the vLLM inference server (Module 100) and uses Mistral's native tool-calling capability to answer time and weather queries for any location on Earth.

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        EKS Cluster – default namespace                        │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  System Node Pool (m5.xlarge) – CPU-only workloads                   │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │           Pod: strands-weather-agent (Deployment)               │  │   │
│  │  │  Image: $ECR_REPO:latest (custom-built from Dockerfile)         │  │   │
│  │  │  Port: 8000  |  replicas: 1                                     │  │   │
│  │  │                                                                  │  │   │
│  │  │  ENV:                                                            │  │   │
│  │  │    MODEL_ENDPOINT = http://vllm-serve-svc:8000/v1               │  │   │
│  │  │    MODEL_ID       = ministral                                    │  │   │
│  │  │                                                                  │  │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │  │   │
│  │  │  │              strands-agent.py (FastAPI app)               │  │  │   │
│  │  │  │                                                            │  │  │   │
│  │  │  │  GET  /health     → { "status": "healthy" }               │  │  │   │
│  │  │  │  POST /agent      → { query: "..." }                      │  │  │   │
│  │  │  │                     → { status, response }                │  │  │   │
│  │  │  │                                                            │  │  │   │
│  │  │  │  Strands Agent:                                            │  │  │   │
│  │  │  │    model=OpenAIModel(base_url, model_id)                  │  │  │   │
│  │  │  │    tools=[current_time, current_weather]                  │  │  │   │
│  │  │  └──────────────────────────────────────────────────────────┘  │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                        │   │
│  │   Service: strands-weather-agent (ClusterIP, port 80 → 8000)         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                         │                                                     │
│                         │ HTTP /v1/chat/completions (tool calls)              │
│                         ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node Pool (g6e.2xlarge)                                          │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │           Pod: mistral (vLLM Deployment)                        │  │   │
│  │  │  --enable-auto-tool-choice  --tool-call-parser=mistral          │  │   │
│  │  │  --enable-prefix-caching    (agents module version)             │  │   │
│  │  │  --max-model-len=4096                                           │  │   │
│  │  │  Port: 8000                                                     │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                             │                                         │   │
│  │   Service: vllm-serve-svc (ClusterIP, port 8000)                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘

External APIs used by the agent tools:
  ┌────────────────────────────────────────────┐
  │  Nominatim (OpenStreetMap)                  │
  │  https://nominatim.openstreetmap.org        │
  │  Geocodes location names → lat/lon          │
  ├────────────────────────────────────────────┤
  │  Open-Meteo Weather API                    │
  │  https://api.open-meteo.com/v1/forecast     │
  │  Returns current weather by lat/lon         │
  ├────────────────────────────────────────────┤
  │  TimezoneFinder (local library)             │
  │  Maps lat/lon → timezone string             │
  │  Used by current_time tool                  │
  └────────────────────────────────────────────┘
```

---

## Agent Tool-Call Flow

```
User Request: POST /agent { "query": "What is the weather in Seattle?" }
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Strands Agent (strands-agents[openai]==1.50.2)                       │
│                                                                       │
│  Step 1 – Initial LLM Call                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  POST http://vllm-serve-svc:8000/v1/chat/completions           │  │
│  │  {                                                              │  │
│  │    "model": "ministral",                                        │  │
│  │    "messages": [{"role":"user", "content": query}],            │  │
│  │    "tools": [current_time_schema, current_weather_schema],     │  │
│  │    "tool_choice": "auto",                                       │  │
│  │    "max_tokens": 1000,                                          │  │
│  │    "temperature": 0.7                                           │  │
│  │  }                                                              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼ Model returns tool call in [TOOL_CALLS] format            │
│  Step 2 – Tool Execution                                              │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Tool: current_weather(location="Seattle")                      │  │
│  │  1. geolocator.geocode("Seattle")   → lat=47.6, lon=-122.3     │  │
│  │  2. GET open-meteo.com/v1/forecast?lat=47.6&lon=-122.3         │  │
│  │     → { temp: 18.5°C, weather_code: 2 }                        │  │
│  │  3. weather_codes[2] = "Partly cloudy"                         │  │
│  │  Returns: "Partly cloudy, 18.5°C"                              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼ Tool result appended to message history                   │
│  Step 3 – Final LLM Call                                              │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  POST /v1/chat/completions (with tool result in messages)       │  │
│  │  Model synthesizes natural language response                   │  │
│  │  Returns: "The current weather in Seattle is partly cloudy,    │  │
│  │            with a temperature of 18.5°C."                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
Response: { "status": "success", "response": "The current weather..." }
```

---

## Agent Tool Definitions

### `current_time` Tool

```python
@tool
def current_time(location: str) -> str:
    """Get the current time for a location."""

    Flow:
    1. geolocator.geocode(location)           # Nominatim geocoding
       Returns: LocationData(lat, lon, ...)
    
    2. tf.timezone_at(lng=lon, lat=lat)        # TimezoneFinder
       Returns: "America/Los_Angeles"
    
    3. datetime.now(pytz.timezone(tz))         # Current local time
       Returns: "2026-09-08 10:48 AM PDT"

    Tool schema exposed to LLM:
      name: "current_time"
      description: "Get the current time for a location."
      parameters:
        location: { type: string, description: "City or location name" }
```

### `current_weather` Tool

```python
@tool
def current_weather(location: str) -> str:
    """Get the current weather for a location."""

    Flow:
    1. geolocator.geocode(location)
       Returns: LocationData(lat=47.6, lon=-122.3)
    
    2. GET https://api.open-meteo.com/v1/forecast
           ?latitude=47.6&longitude=-122.3
           &current=temperature_2m,weather_code
           &timezone=auto
       Returns: { "current": { "temperature_2m": 18.5, "weather_code": 2 } }
    
    3. weather_codes lookup:
       { 0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
         3: "Overcast", 45: "Fog", 61: "Light rain", 63: "Moderate rain",
         65: "Heavy rain", 71: "Light snow", 95: "Thunderstorm", ... }
    
    Returns: "Partly cloudy, 18.5°C"

    Tool schema exposed to LLM:
      name: "current_weather"
      description: "Get the current weather for a location."
      parameters:
        location: { type: string, description: "City or location name" }
```

---

## vLLM Configuration for Agents (vllm-deployment-agents.yaml)

This module uses a **modified vLLM deployment** with prefix caching enabled for improved agent performance:

```
vLLM Arguments (agents-specific additions):

--enable-prefix-caching
    Caches KV states for repeated prompt prefixes.
    Critical for agents: system prompt + tool schemas are repeated
    on every LLM call within a multi-step agent loop.
    Reduces TTFT by 50-80% for subsequent calls in the same session.

--max-model-len=4096
    Reduced from 8192 to 4096 for agents workload.
    Agent prompts + tool schemas + history typically < 3000 tokens.
    Allows more concurrent agent sessions in KV cache.

--enable-auto-tool-choice
    Enables automatic detection of tool call intent.
    Model can choose to call tools or respond directly.

--tool-call-parser=mistral
    Parses Mistral's [TOOL_CALLS] output format.
    Extracts: function name, arguments (JSON).

nodeSelector:
    nvidia.com/gpu.present: "true"    # Different from Module 100
    (Module 100 uses: karpenter.sh/nodepool: gpu)
```

---

## FastAPI Application (strands-agent.py)

```
Application: FastAPI – "Time and Weather Agent"
Python:       3.12-slim (Dockerfile)
Port:         8000
CMD:          uvicorn app:app --host 0.0.0.0 --port 8000

Model Configuration:
  OpenAIModel(
    client_args = {
      api_key:  "xxxxxxxxxxxx"  (vLLM does not check API keys)
      base_url: os.getenv("MODEL_ENDPOINT")  # http://vllm-serve-svc:8000/v1
    }
    model_id = os.getenv("MODEL_ID")         # ministral
    params = {
      max_tokens:  1000
      temperature: 0.7
    }
  )

Endpoints:
  GET  /health  → { "status": "healthy" }
  POST /agent   → { "query": "..." }
                ← { "status": "success", "response": "..." }
```

---

## Python Dependencies (requirements.txt)

```
fastapi==0.141.1          # Web framework
uvicorn==0.52.1           # ASGI server
pydantic==2.13.4          # Request/response validation
requests==2.34.2          # HTTP client (weather API calls)
strands-agents[openai]==1.50.2   # Strands Agents SDK with OpenAI client
geopy==2.5.0              # Geocoding (Nominatim)
timezonefinder==8.2.5     # Timezone from lat/lon (local lookup)
pytz==2026.3.post1        # Timezone-aware datetime handling
```

---

## Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt      # Install all dependencies

COPY strands-agent.py app.py             # Rename to app.py for uvicorn

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Kubernetes Resources

### Deployment (vllm-deployment-agents.yaml)

```
Deployment: strands-weather-agent
  replicas: 1
  image: $ECR_REPO:latest

  ENV:
    MODEL_ENDPOINT: http://vllm-serve-svc:8000/v1
    MODEL_ID:       ministral

Service: strands-weather-agent
  type: ClusterIP
  port: 80 → targetPort: 8000
```

---

## Build and Deploy

```bash
# Build and push container to ECR
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
export ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/strands-agent"

# Create ECR repository
aws ecr create-repository --repository-name strands-agent --region ${AWS_REGION}

# Authenticate Docker to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${ECR_REPO}

# Build and push
docker build -t ${ECR_REPO}:latest .
docker push ${ECR_REPO}:latest

# Deploy with image substitution
sed "s|\$ECR_REPO|${ECR_REPO}|g" vllm-deployment-agents.yaml | kubectl apply -f -

# Test the agent
kubectl port-forward svc/strands-weather-agent 8080:80 &

curl -X POST http://localhost:8080/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather and current time in Tokyo?"}'
```
