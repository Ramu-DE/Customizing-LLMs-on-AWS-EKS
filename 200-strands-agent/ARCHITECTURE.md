# Module 200 – Strands Agent: Agentic AI Inference in Action

> New to AI inference? Read [CONCEPTS.md](../CONCEPTS.md) first.
> Prerequisite: Module 100 (vLLM) must be running.

---

## What This Module Does

Module 100 showed a model that *answers questions*. This module shows a model that *takes actions*.

An **AI agent** goes beyond text generation. It can:
1. Receive a question from a user
2. Decide it needs real-world information to answer it
3. **Call an external tool** (a function, an API) to get that information
4. Use the tool's result to compose a final, accurate answer

This is only possible because the Ministral-3-8B model supports **tool calling** – a structured protocol where the model outputs a function call specification instead of plain text, which the agent framework executes on the model's behalf.

---

## AI Inference Concepts Demonstrated Here

### Agentic AI: Beyond Question-Answering

The Red Hat article describes how AI inference is expanding to increasingly complex use cases. Agents are the next step: AI that plans, decides what information it needs, fetches it, and synthesises an answer.

```
Simple LLM (Module 100):
  User:  "What is the weather in Tokyo?"
  Model: "I don't have access to real-time weather data."  ← knows nothing current

AI Agent (Module 200):
  User:  "What is the weather in Tokyo?"
  Agent: I need to call current_weather("Tokyo")
         → API returns: "Partly cloudy, 22°C"
  Model: "The current weather in Tokyo is partly cloudy with a temperature of 22°C."
                                                           ← grounded in reality
```

### Online Inference with Multi-Step Reasoning

An agent performs **multiple online inference calls** for a single user question:

```
User question
    │
    ▼
Call 1 to vLLM: "Given this question and these tools, what should I do?"
    │ Model responds: "Call current_weather(location='Tokyo')"
    ▼
Execute tool: current_weather("Tokyo") → API → "Partly cloudy, 22°C"
    │
    ▼
Call 2 to vLLM: "Given the question AND the tool result, write the final answer"
    │ Model responds: "The weather in Tokyo is partly cloudy, 22°C"
    ▼
Return answer to user
```

Each vLLM call is online inference. The agent orchestrates the loop.

### Prefix Caching – Why Agents Need It

The agent sends the same system prompt and tool schemas on **every** LLM call. Without caching, vLLM recomputes the KV tensors for this repeated prefix every time.

With `--enable-prefix-caching`, vLLM stores those KV tensors after the first call. Subsequent calls with the same prefix skip the prefill computation entirely.

```
Without prefix caching (each agent call):
  Prefill: system_prompt (200 tokens) + tools (300 tokens) + history (growing)
  Every call recomputes all 500+ fixed tokens. Wasteful.

With prefix caching (--enable-prefix-caching):
  First call: compute KV for 500 fixed tokens → store in GPU cache
  Later calls: load from cache → skip recomputation → 50-80% faster TTFT
```

This is especially important for agents because they make 2–5 LLM calls per user question.

---

## Full Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EKS Cluster – default namespace                                              │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  System Node (m5.xlarge) – CPU only                                   │   │
│  │                                                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  Deployment: strands-weather-agent (replicas: 1)               │  │   │
│  │  │  Image: $ECR_REPO:latest (built from Dockerfile)               │  │   │
│  │  │  Port: 8000 (FastAPI + Uvicorn)                                 │  │   │
│  │  │                                                                  │  │   │
│  │  │  ENV:                                                            │  │   │
│  │  │    MODEL_ENDPOINT = http://vllm-serve-svc:8000/v1              │  │   │
│  │  │    MODEL_ID       = ministral                                   │  │   │
│  │  │                                                                  │  │   │
│  │  │  Tools available:                                               │  │   │
│  │  │    current_time(location)    → Nominatim + pytz                │  │   │
│  │  │    current_weather(location) → Nominatim + Open-Meteo API      │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  Service: strands-weather-agent (ClusterIP, port 80 → 8000)          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                │  POST /v1/chat/completions (with tool schemas)              │
│                ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  GPU Node (g6e.2xlarge) – NVIDIA L40S                                 │   │
│  │                                                                       │   │
│  │  Deployment: mistral (vLLM agents variant)                           │   │
│  │  Extra flags vs Module 100:                                          │   │
│  │    --enable-prefix-caching     (reuse system prompt KV cache)       │   │
│  │    --max-model-len=4096        (shorter ctx for faster agent calls)  │   │
│  │    --enable-auto-tool-choice                                          │   │
│  │    --tool-call-parser=mistral                                         │   │
│  │                                                                       │   │
│  │  Service: vllm-serve-svc (ClusterIP, port 8000)                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘

External APIs used at tool-call time (called by the agent pod, not vLLM):
  Nominatim API: https://nominatim.openstreetmap.org   → geocoding (text → lat/lon)
  Open-Meteo API: https://api.open-meteo.com/v1/forecast → live weather data
  TimezoneFinder: local Python library (no network call)  → lat/lon → timezone
```

---

## Tool-Call Flow: Step by Step

```
POST /agent { "query": "What's the weather and local time in Berlin?" }
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  strands-agent.py: agent(request.query)                                   │
│                                                                           │
│  ┌─ STEP 1: First LLM Call ────────────────────────────────────────────┐ │
│  │                                                                       │ │
│  │  POST http://vllm-serve-svc:8000/v1/chat/completions               │ │
│  │  {                                                                   │ │
│  │    "model": "ministral",                                             │ │
│  │    "messages": [                                                     │ │
│  │      { "role": "user", "content": "What's the weather and time      │ │
│  │                                    in Berlin?" }                     │ │
│  │    ],                                                                │ │
│  │    "tools": [                                                        │ │
│  │      {                                                               │ │
│  │        "type": "function",                                           │ │
│  │        "function": {                                                 │ │
│  │          "name": "current_weather",                                  │ │
│  │          "description": "Get the current weather for a location",   │ │
│  │          "parameters": {                                             │ │
│  │            "type": "object",                                         │ │
│  │            "properties": {                                           │ │
│  │              "location": { "type": "string" }                       │ │
│  │            }                                                         │ │
│  │          }                                                           │ │
│  │        }                                                             │ │
│  │      },                                                              │ │
│  │      { ... current_time schema ... }                                 │ │
│  │    ],                                                                │ │
│  │    "tool_choice": "auto",                                            │ │
│  │    "max_tokens": 1000,                                               │ │
│  │    "temperature": 0.7                                                │ │
│  │  }                                                                   │ │
│  │                                                                       │ │
│  │  vLLM (Ministral) responds with tool calls:                         │ │
│  │  [TOOL_CALLS] [                                                      │ │
│  │    { "name": "current_weather", "arguments": {"location":"Berlin"} } │ │
│  │    { "name": "current_time",    "arguments": {"location":"Berlin"} } │ │
│  │  ]                                                                   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─ STEP 2: Execute Tools ─────────────────────────────────────────────┐ │
│  │                                                                       │ │
│  │  current_weather("Berlin"):                                          │ │
│  │    1. geolocator.geocode("Berlin")                                   │ │
│  │       → lat=52.52, lon=13.40                                         │ │
│  │    2. GET api.open-meteo.com/v1/forecast                             │ │
│  │         ?latitude=52.52&longitude=13.40                              │ │
│  │         &current=temperature_2m,weather_code&timezone=auto          │ │
│  │       → { "temperature_2m": 17.3, "weather_code": 1 }               │ │
│  │    3. weather_codes[1] = "Mainly clear"                              │ │
│  │    Returns: "Mainly clear, 17.3°C"                                   │ │
│  │                                                                       │ │
│  │  current_time("Berlin"):                                             │ │
│  │    1. geolocator.geocode("Berlin") → lat=52.52, lon=13.40           │ │
│  │    2. tf.timezone_at(lng=13.40, lat=52.52) → "Europe/Berlin"        │ │
│  │    3. datetime.now(pytz.timezone("Europe/Berlin"))                   │ │
│  │       → "2026-09-08 08:03 PM CEST"                                  │ │
│  │    Returns: "2026-09-08 08:03 PM CEST"                               │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─ STEP 3: Final LLM Call ────────────────────────────────────────────┐ │
│  │                                                                       │ │
│  │  POST /v1/chat/completions                                            │ │
│  │  messages now include:                                                │ │
│  │    - original user question                                           │ │
│  │    - tool call results (weather + time)                               │ │
│  │  Model synthesises: "In Berlin it is currently 8:03 PM (CEST).      │ │
│  │    The weather is mainly clear with a temperature of 17.3°C."        │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
{ "status": "success", "response": "In Berlin it is currently 8:03 PM..." }
```

---

## Application Code (strands-agent.py): Every Part Explained

### Model Initialisation

```python
model = OpenAIModel(
    client_args={
        "api_key": "xxxxxxxxxxxx",
        # vLLM accepts any non-empty API key by default.
        # This placeholder satisfies the OpenAI client library's requirement.
        # To add real auth: add --api-key=<secret> to vLLM args.

        "base_url": os.getenv("MODEL_ENDPOINT")
        # http://vllm-serve-svc:8000/v1
        # Kubernetes DNS resolves vllm-serve-svc to the ClusterIP Service.
        # The agent talks to vLLM via the cluster's internal network.
        # No public internet exposure for the inference backend.
    },
    model_id=os.getenv("MODEL_ID"),
    # "ministral" – must match --served-model-name in vLLM deployment.

    params={
        "max_tokens": 1000,
        # Maximum tokens the model generates per LLM call.
        # Agent calls are usually short (a tool call spec or a short answer).
        # 1000 tokens is generous but prevents runaway generation.

        "temperature": 0.7,
        # Controls randomness. 0 = always pick the most likely token.
        # 0.7 = some creativity but mostly deterministic.
        # For tool selection, lower is more reliable (0.0–0.3).
        # For final answer text, 0.7 gives natural-sounding responses.
    }
)
```

### Tool Definitions

```python
@tool
def current_time(location: str) -> str:
    """Get the current time for a location.
    
    Args:
        location: The city or location name to get the time for
    """
    # The docstring IS the tool schema.
    # Strands SDK reads it and generates the JSON tool definition
    # that gets sent to the model in the "tools" array.
    # The model reads the description to decide when to call this tool.
    
    # Implementation:
    location_data = geolocator.geocode(location)
    # Nominatim: free OpenStreetMap geocoding API.
    # "Berlin" → LocationData(lat=52.52, lon=13.40, address=...)
    
    timezone = tf.timezone_at(lng=location_data.longitude,
                              lat=location_data.latitude)
    # TimezoneFinder: local Python library (~50 MB database).
    # Finds timezone from coordinates. No network call.
    # (52.52, 13.40) → "Europe/Berlin"
    
    dt = datetime.now(pytz.timezone(timezone))
    # pytz converts current UTC time to the target timezone.
    
    return dt.strftime("%Y-%m-%d %I:%M %p %Z")
    # Returns: "2026-09-08 08:03 PM CEST"

@tool
def current_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city or location name to get the weather for
    """
    location_data = geolocator.geocode(location)
    lat, lon = location_data.latitude, location_data.longitude
    
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,weather_code&timezone=auto")
    # Open-Meteo: free weather API, no API key needed.
    # Returns: { "current": { "temperature_2m": 17.3, "weather_code": 1 } }
    
    response = requests.get(url, timeout=10)
    data = response.json()
    temp = data["current"]["temperature_2m"]
    code = data["current"]["weather_code"]
    
    weather_codes = {
        0: "Clear sky",       # Perfect day
        1: "Mainly clear",    # Mostly sunny
        2: "Partly cloudy",   # Some clouds
        3: "Overcast",        # Fully cloudy
        45: "Fog",
        51: "Light drizzle",  53: "Moderate drizzle",
        61: "Light rain",     63: "Moderate rain",   65: "Heavy rain",
        71: "Light snow",     73: "Moderate snow",   75: "Heavy snow",
        95: "Thunderstorm",
        # Full list: https://open-meteo.com/en/docs#weathervariables
    }
    return f"{weather_codes.get(code, f'Unknown (code {code})')}, {temp}°C"
```

### FastAPI Endpoints

```python
app = FastAPI(title="Time and Weather Agent")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
    # Kubernetes liveness/readiness probe target.
    # Returns immediately – no inference needed for health checks.

@app.post("/agent")
async def agent_endpoint(request: QueryRequest):
    """Endpoint that uses time and weather agent"""
    agent = Agent(model=model, tools=[current_time, current_weather])
    # A new Agent instance is created per request.
    # This is stateless – no conversation history carried between requests.
    # For multi-turn conversations, persist agent state externally.
    
    response = agent(request.query)
    # This triggers the full tool-call loop:
    #   LLM call 1 → tool execution(s) → LLM call 2 → response
    
    return {
        "status": "success",
        "response": response
        # The Strands response object. Contains the final text answer.
    }
```

---

## Python Dependencies Explained

```
requirements.txt:

fastapi==0.141.1
  Web framework that turns Python functions into HTTP endpoints.
  Handles request parsing, validation, serialisation automatically.

uvicorn==0.52.1
  ASGI server that runs FastAPI. Handles concurrent HTTP connections.
  Production-grade alternative to Flask's built-in server.

pydantic==2.13.4
  Data validation library. Defines the shape of API request/response bodies.
  QueryRequest = pydantic model that enforces "query" must be a string.

requests==2.34.2
  Simple HTTP client. Used by current_weather tool to call Open-Meteo API.

strands-agents[openai]==1.50.2
  The Strands Agents SDK from AWS.
  [openai] extra: includes OpenAI-compatible client for talking to vLLM.
  Provides: @tool decorator, Agent class, tool-call loop logic.

geopy==2.5.0
  Geocoding library. Nominatim provider turns "Berlin" → (52.52, 13.40).
  Supports many providers: Google Maps, Here, OpenStreetMap, etc.

timezonefinder==8.2.5
  Determines timezone from geographic coordinates.
  Uses a local database (no network call). Fast and accurate.

pytz==2026.3.post1
  Python timezone library. Converts UTC times to any timezone.
  Required by timezonefinder for correct local time display.
```

---

## Dockerfile Explained

```dockerfile
FROM python:3.12-slim
# Lightweight Python 3.12 base image (~130 MB vs ~900 MB for full python:3.12).
# No GPU drivers needed – this container runs on a CPU node.

WORKDIR /app
# All subsequent commands run in /app. Sets container working directory.

COPY requirements.txt .
RUN pip install -r requirements.txt
# Install dependencies BEFORE copying app code.
# Docker layer caching: if requirements.txt doesn't change,
# this expensive layer is reused on rebuilds → faster CI.

COPY strands-agent.py app.py
# Rename to app.py so uvicorn finds it as: uvicorn app:app

EXPOSE 8000
# Documents the port (doesn't actually open it – that's Kubernetes' job).

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# --host 0.0.0.0: listen on all interfaces (required in containers)
# --port 8000: match EXPOSE and Kubernetes Service targetPort
```

---

## Kubernetes Deployment (vllm-deployment-agents.yaml)

### Strands Agent Deployment

```yaml
Deployment: strands-weather-agent
  replicas: 1
  image: $ECR_REPO:latest    # Your ECR repository (substitute before applying)

  env:
    MODEL_ENDPOINT: http://vllm-serve-svc:8000/v1
    # Kubernetes DNS name for the vLLM service.
    # Resolves inside the cluster. External traffic cannot reach vLLM directly.

    MODEL_ID: ministral
    # Must match --served-model-name in the vLLM deployment.
    # If you change the vLLM model name, change this too.

Service: strands-weather-agent
  type: ClusterIP        # Internal only. Expose via Ingress if needed publicly.
  port: 80               # External port
  targetPort: 8000       # Container port (FastAPI)
```

### vLLM Agents Variant

This variant adds `--enable-prefix-caching` and reduces `--max-model-len=4096`:

```yaml
Additional args vs Module 100 vllm-deployment.yml:

--enable-prefix-caching
  Why for agents: The system prompt + tool schemas (500-700 tokens) are
  identical in EVERY LLM call within the same agent session.
  Prefix caching stores their KV tensors after the first call.
  Benefit: 50-80% reduction in TTFT for calls 2, 3, 4 in a session.
  GPU memory: small overhead to store the cache entry.

--max-model-len=4096
  Why 4096 instead of 8192:
  Agent prompts are shorter than long-form chat.
  system_prompt (200) + tools (300) + user message (100) + history (200)
  = ~800 tokens. Plenty of room within 4096.
  Halving the max context doubles the number of concurrent agent sessions
  the KV cache can hold.

nodeSelector change:
  Module 100: karpenter.sh/nodepool: gpu
  Module 200: nvidia.com/gpu.present: "true"   ← different selector approach
  Both land on GPU nodes; the selector style is just a configuration choice.
```

---

## Testing the Agent

```bash
# Build and push to ECR first (one-time setup)
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
export ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/strands-agent"

aws ecr create-repository --repository-name strands-agent --region ${AWS_REGION}
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${ECR_REPO}
docker build -t ${ECR_REPO}:latest .
docker push ${ECR_REPO}:latest

# Deploy
sed "s|\$ECR_REPO|${ECR_REPO}|g" vllm-deployment-agents.yaml | kubectl apply -f -

# Wait for agent pod to be ready
kubectl wait pod -l app=strands-weather-agent --for=condition=Ready --timeout=300s

# Port-forward and test
kubectl port-forward svc/strands-weather-agent 8080:80 &

# Simple weather query (triggers 2 LLM calls + 1 tool call)
curl -X POST http://localhost:8080/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Seattle right now?"}'

# Multi-tool query (triggers 2 LLM calls + 2 tool calls)
curl -X POST http://localhost:8080/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What time is it and what is the weather in Tokyo and London?"}'

# Health check
curl http://localhost:8080/health
```
