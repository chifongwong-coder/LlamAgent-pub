"""
LlamAgent HTTP API Server.

Wraps LlamAgent as a RESTful API + WebSocket service using FastAPI.
FastAPI is an optional dependency; importing this module without it installed
will raise an ImportError with installation instructions.

Endpoints:
    POST /chat         — Chat
    POST /chat/stream  — Streaming chat via SSE (text chunks)
    POST /chat/stream/events — Streaming chat via SSE (structured events)
    GET  /status       — Health check
    GET  /modules      — Module list
    POST /upload       — Upload files to knowledge base
    GET  /sessions     — List saved sessions
    DELETE /sessions/{persona_id} — Delete a session
    WS   /ws/chat      — WebSocket streaming chat

Usage:
    python -m llamagent --mode api
    python -m llamagent.interfaces.api_server
    API docs: http://localhost:8000/docs (Swagger UI)
"""

import json
import os
import time
import asyncio
import logging
import threading
from collections import OrderedDict

# FastAPI and related dependencies: optional install
try:
    from fastapi import (
        FastAPI, HTTPException, Depends,
        WebSocket, WebSocketDisconnect, UploadFile, File,
    )
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    from typing import Optional
    from contextlib import asynccontextmanager
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ============================================================
# Logging configuration
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llamagent-api")


# ============================================================
# Global state
# ============================================================

# Server start time
START_TIME = time.time()

# Agent instance pool — maintains independent Agents for different sessions
# Uses OrderedDict for LRU eviction to prevent unbounded memory growth
MAX_SESSIONS = 100

# Session storage and rate limit counters
agent_sessions: OrderedDict = OrderedDict()
rate_limit_store: dict[str, list[float]] = {}

# Active ContinuousRunner instances per session
runner_sessions: dict[str, tuple] = {}  # sid -> (ContinuousRunner, threading.Thread)

# Auth token (configured via environment variable, empty string = dev mode, no auth)
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")

# Agent creation factory function — set by create_api_server()
_agent_factory = None


# ============================================================
# Pydantic data models (request & response)
# ============================================================
# Data structures defined with Pydantic's BaseModel.
# Three birds with one stone: parameter validation + auto serialization + API doc generation.

if HAS_FASTAPI:

    class ChatRequest(BaseModel):
        """Chat request"""
        message: str = Field(
            ...,
            min_length=1,
            max_length=10000,
            description="User message content",
            examples=["Help me check the weather in Beijing today"],
        )
        session_id: Optional[str] = Field(
            default=None,
            description="Session ID for maintaining context. If omitted, the default session is used",
        )

    class ChatStreamRequest(BaseModel):
        """Streaming chat request"""
        message: str = Field(
            ...,
            min_length=1,
            max_length=10000,
            description="User message content",
        )
        session_id: Optional[str] = Field(
            default=None,
            description="Session ID for maintaining context. If omitted, the default session is used",
        )

    class ChatResponse(BaseModel):
        """Chat response"""
        reply: str = Field(description="Agent's reply content")
        session_id: str = Field(description="Session ID")
        model: str = Field(description="Model name used")

    class StatusResponse(BaseModel):
        """Status response"""
        status: str = Field(description="Service status: healthy / degraded")
        version: str = Field(description="Service version")
        model: str = Field(description="Currently used model")
        uptime_seconds: float = Field(description="Service uptime in seconds")
        modules: dict = Field(description="Loaded modules")

    class ModuleInfo(BaseModel):
        """Module information"""
        name: str = Field(description="Module name")
        description: str = Field(description="Module description")
        loaded: bool = Field(description="Whether the module is loaded")

    class UploadResponse(BaseModel):
        """File upload response"""
        message: str = Field(description="Upload result description")
        files_processed: int = Field(description="Number of files processed")

    class ErrorResponse(BaseModel):
        """Standardized error response"""
        error: str = Field(description="Error type")
        message: str = Field(description="Error description")
        detail: Optional[str] = Field(
            default=None,
            description="Detailed info (only returned in DEBUG mode)",
        )

    class ModeRequest(BaseModel):
        """Mode switch request"""
        mode: str = Field(
            ...,
            description="Target mode: interactive, task",
            examples=["task"],
        )
        session_id: Optional[str] = Field(
            default=None,
            description="Session ID. If omitted, the default session is used",
        )

    class ModeResponse(BaseModel):
        """Mode response"""
        mode: str = Field(description="Current agent mode")
        config: dict = Field(description="Mode-related config values")

    class AbortResponse(BaseModel):
        """Abort response"""
        success: bool = Field(description="Whether abort signal was sent")
        message: str = Field(description="Status message")

    class RunnerStartRequest(BaseModel):
        """Runner start request"""
        trigger_type: str = Field(
            default="timer",
            description="Trigger type: 'timer' or 'file'",
            examples=["timer"],
        )
        interval: float = Field(
            default=60,
            description="Timer interval in seconds (used when trigger_type='timer')",
        )
        message: str = Field(
            default="check status",
            description="Task message for timer trigger",
        )
        watch_dir: str = Field(
            default=".",
            description="Directory to watch (used when trigger_type='file')",
        )
        session_id: Optional[str] = Field(
            default=None,
            description="Session ID. If omitted, the default session is used",
        )

    class RunnerStatusResponse(BaseModel):
        """Runner status response"""
        active: bool = Field(description="Whether a runner is active")
        tasks_completed: int = Field(description="Number of tasks completed")
        session_id: str = Field(description="Session ID")

    class InjectRequest(BaseModel):
        """Inject message request"""
        message: str = Field(
            ...,
            min_length=1,
            max_length=10000,
            description="Message to inject into the running agent",
        )
        session_id: Optional[str] = Field(
            default=None,
            description="Session ID. If omitted, the default session is used",
        )

    class InjectResponse(BaseModel):
        """Inject response"""
        reply: str = Field(description="Agent's reply to the injected message")
        session_id: str = Field(description="Session ID")

    class RunnerLogEntry(BaseModel):
        """Single runner log entry"""
        input: str = Field(description="Task input")
        output: str = Field(description="Task output")
        status: str = Field(description="Task status: completed or error")
        duration: float = Field(description="Task duration in seconds")

    class RunnerLogResponse(BaseModel):
        """Runner log response"""
        entries: list[RunnerLogEntry] = Field(description="Log entries")
        session_id: str = Field(description="Session ID")


# ============================================================
# Helper functions
# ============================================================

def _make_trigger(request):
    """Create a trigger from a RunnerStartRequest."""
    from llamagent.core.runner import TimerTrigger, FileTrigger
    if request.trigger_type == "file":
        return FileTrigger(request.watch_dir)
    return TimerTrigger(interval=request.interval, message=request.message)

def _get_agent(session_id: str | None = None):
    """
    Get an Agent instance (LRU eviction strategy).

    If a session_id is specified, returns the Agent for that session (creates one if it doesn't exist).
    If not specified, returns the default instance.
    Automatically evicts the least recently used session when MAX_SESSIONS is exceeded.
    """
    sid = session_id or "default"

    if sid in agent_sessions:
        # LRU update: move the accessed session to the end
        agent_sessions.move_to_end(sid)
        return agent_sessions[sid]

    # Create a new session
    logger.info("Creating new Agent instance: session=%s", sid)
    if _agent_factory:
        new_agent = _agent_factory()
    else:
        # Fallback: create a bare Agent directly
        from llamagent.core import LlamAgent, Config
        new_agent = LlamAgent(Config())

    agent_sessions[sid] = new_agent

    # Evict the oldest session
    while len(agent_sessions) > MAX_SESSIONS:
        evicted_sid, evicted_agent = agent_sessions.popitem(last=False)
        # Stop runner if active for the evicted session
        evicted_entry = runner_sessions.pop(evicted_sid, None)
        if evicted_entry:
            evicted_runner, _ = evicted_entry
            try:
                evicted_runner.stop()
            except Exception as e:
                logger.error("Failed to stop runner for evicted session=%s: %s", evicted_sid, e)
        # Shutdown agent (fires on_shutdown hooks: persistence save, cleanup, etc.)
        try:
            evicted_agent.shutdown()
        except Exception as e:
            logger.error("Failed to shutdown evicted agent session=%s: %s", evicted_sid, e)
        logger.info("LRU evicted expired session: session=%s", evicted_sid)

    return new_agent


# ============================================================
# FastAPI application factory
# ============================================================

def create_api_server(
    module_names: list[str] | None = None,
    persona_name: str | None = None,
) -> "FastAPI":
    """
    Create a FastAPI application instance.

    Args:
        module_names: List of modules to load (passed to create_agent)
        persona_name: Persona name

    Returns:
        A FastAPI instance with routes and middleware configured

    Raises:
        ImportError: Raised when FastAPI is not installed
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is not installed! Please run: pip install fastapi uvicorn[standard]\n"
            "Then try again."
        )

    from llamagent.core import LlamAgent, Config
    from llamagent.main import load_module, AVAILABLE_MODULES
    from llamagent.core.zone import ConfirmResponse
    from llamagent.interfaces.presets import apply_presets

    # Set the Agent factory function for _get_agent() to use when creating new sessions
    def _factory():
        config = Config()

        # Determine modules to load
        names = module_names if module_names is not None else list(AVAILABLE_MODULES.keys())

        # Apply presets BEFORE module registration (modules read config in on_attach)
        apply_presets(config, names)

        # Load persona if specified
        persona = None
        if persona_name:
            from llamagent.core import PersonaManager
            try:
                manager = PersonaManager(config.persona_file)
                persona = manager.get(persona_name)
                if not persona:
                    for p in manager.list():
                        if p.name == persona_name:
                            persona = p
                            break
            except Exception:
                pass

        agent = LlamAgent(config, persona=persona)

        for name in names:
            mod = load_module(name)
            if mod:
                agent.register_module(mod)

        # Auto-approve for API server (callers manage authorization via contract flow)
        agent.confirm_handler = lambda req: ConfirmResponse(allow=True)
        return agent

    global _agent_factory
    _agent_factory = _factory

    # --- Lifecycle management ---

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Code executed during server startup/shutdown."""
        global START_TIME
        START_TIME = time.time()

        logger.info("=" * 60)
        logger.info("LlamAgent API server starting...")
        logger.info("  Auth: %s", "Enabled" if API_AUTH_TOKEN else "Disabled (dev mode)")
        logger.info("  Max sessions: %d", MAX_SESSIONS)
        logger.info("=" * 60)

        # Create the default Agent instance
        agent_sessions["default"] = _agent_factory()
        logger.info("Default Agent instance created")

        yield  # Server running...

        # Shutdown phase
        logger.info("LlamAgent API server shutting down...")

        # Stop all active runners first (runners call agent.chat, so stop before agent shutdown)
        for sid, (runner, thread) in list(runner_sessions.items()):
            try:
                runner.stop()
                thread.join(timeout=5)
            except Exception as e:
                logger.error("Failed to stop runner for session=%s: %s", sid, e)
        runner_sessions.clear()

        for sid, agent in agent_sessions.items():
            try:
                agent.shutdown()
            except Exception as e:
                logger.error("Failed to shutdown agent session=%s: %s", sid, e)
        agent_sessions.clear()
        logger.info("All Agent instances cleaned up")

    # --- Create FastAPI app ---

    from llamagent import __version__

    app = FastAPI(
        title="LlamAgent API",
        description=(
            "LlamAgent RESTful API\n\n"
            "An AI Agent service with chat, tool calling, knowledge retrieval, "
            "and reasoning & planning capabilities.\n\n"
            "## Endpoints\n"
            "- POST /chat — Chat\n"
            "- POST /chat/stream — Streaming chat via SSE (text chunks)\n"
            "- POST /chat/stream/events — Streaming chat via SSE (structured events)\n"
            "- GET /status — Agent status\n"
            "- GET /modules — Module list\n"
            "- POST /upload — Upload files to knowledge base\n"
            "- POST /mode — Switch agent mode\n"
            "- GET /mode — Get current mode\n"
            "- POST /abort — Abort current task\n"
            "- POST /runner/start — Start continuous mode runner\n"
            "- POST /runner/stop — Stop continuous mode runner\n"
            "- GET /runner/status — Runner status\n"
            "- GET /runner/log — Runner task log\n"
            "- POST /inject — Inject message into running agent\n"
            "- GET /sessions — List saved sessions\n"
            "- DELETE /sessions/{persona_id} — Delete a session\n"
            "- WebSocket /ws/chat — Streaming chat\n"
        ),
        version=__version__,
        lifespan=lifespan,
        responses={
            401: {"model": ErrorResponse, "description": "Authentication failed"},
            429: {"model": ErrorResponse, "description": "Too many requests"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
        },
    )

    # --- CORS middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific frontend domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Rate limiting middleware ---

    class RateLimitMiddleware(BaseHTTPMiddleware):
        """Simple rate limiter: limits the maximum number of requests per IP within a time window."""

        def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
            super().__init__(app)
            self.max_requests = max_requests
            self.window_seconds = window_seconds

        async def dispatch(self, request: Request, call_next) -> Response:
            # Health check and docs endpoints are exempt from rate limiting
            exempt_paths = ("/status", "/docs", "/redoc", "/openapi.json")
            if request.url.path in exempt_paths:
                return await call_next(request)

            client_ip = request.client.host if request.client else "unknown"
            now = time.time()

            # Clean expired records + check rate limit
            timestamps = rate_limit_store.get(client_ip, [])
            timestamps = [t for t in timestamps if now - t < self.window_seconds]

            if len(timestamps) >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": f"Too many requests, please retry after {self.window_seconds} seconds",
                    }
                )

            timestamps.append(now)
            rate_limit_store[client_ip] = timestamps
            return await call_next(request)

    app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)

    # --- Auth dependency ---

    security = HTTPBearer(auto_error=False)

    async def verify_token(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> str:
        """
        Verify Bearer Token.

        Skips authentication when API_AUTH_TOKEN is not configured (dev mode).
        """
        if not API_AUTH_TOKEN:
            return "anonymous"

        if not credentials:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "missing_token",
                    "message": "Please provide an API Key in the header: Authorization: Bearer <your-key>",
                }
            )

        if credentials.credentials != API_AUTH_TOKEN:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Invalid API Key, please check your credentials",
                }
            )

        return credentials.credentials

    # ============================================================
    # API route definitions
    # ============================================================

    # ---- 1. Chat endpoint ----

    @app.post(
        "/chat",
        response_model=ChatResponse,
        tags=["Chat"],
        summary="Send a chat message",
        description="Send a message to LlamAgent and get a reply. Supports multiple sessions via session_id.",
    )
    async def chat(
        request: ChatRequest,
        token: str = Depends(verify_token),
    ):
        """Handle chat requests."""
        agent = _get_agent(request.session_id)
        session_id = request.session_id or "default"

        try:
            # asyncio.to_thread runs the synchronous agent.chat() in a thread pool,
            # preventing it from blocking FastAPI's event loop
            reply = await asyncio.to_thread(agent.chat, request.message)

            return ChatResponse(
                reply=reply,
                session_id=session_id,
                model=agent.config.model,
            )

        except Exception as e:
            logger.error("Chat processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "chat_failed",
                    "message": f"Error processing chat: {e}",
                }
            )

    # ---- 1b. Streaming chat endpoint (SSE) ----

    @app.post(
        "/chat/stream",
        tags=["Chat"],
        summary="Streaming chat via SSE",
        description="Send a message and receive the response as a Server-Sent Events stream.",
    )
    async def chat_stream(
        request: ChatStreamRequest,
        token: str = Depends(verify_token),
    ):
        """Handle streaming chat requests via SSE."""
        agent = _get_agent(request.session_id)

        def event_generator():
            try:
                for chunk in agent.chat_stream(request.message):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- 1c. Structured-event streaming endpoint (SSE, v3.0.3) ----

    @app.post(
        "/chat/stream/events",
        tags=["Chat"],
        summary="Streaming chat with structured events",
        description=(
            "Send a message and receive a Server-Sent Events stream of structured "
            "events: content / tool_call_start / tool_call_end / status / error / done. "
            "See interfaces/stream_protocol.py for the event schema."
        ),
    )
    async def chat_stream_events(
        request: ChatStreamRequest,
        token: str = Depends(verify_token),
    ):
        """Handle streaming chat requests and yield structured events via SSE."""
        agent = _get_agent(request.session_id)

        from llamagent.interfaces.stream_adapter import adapt_stream

        def event_generator():
            try:
                for event in adapt_stream(agent.chat_stream(request.message)):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error("Structured stream error: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'seq': -1, 'message': str(e)})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'seq': -1})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- 2. Status endpoint (no auth required) ----

    @app.get(
        "/status",
        response_model=StatusResponse,
        tags=["System"],
        summary="Agent status and health check",
        description="Returns the current Agent status. No authentication required.",
    )
    async def get_status():
        """Return Agent runtime status."""
        try:
            agent = _get_agent()
            status = agent.status()
            return StatusResponse(
                status="healthy",
                version=__version__,
                model=status.get("model", "Unknown"),
                uptime_seconds=round(time.time() - START_TIME, 2),
                modules=status.get("modules", {}),
            )
        except Exception:
            return StatusResponse(
                status="degraded",
                version=__version__,
                model="Unknown",
                uptime_seconds=round(time.time() - START_TIME, 2),
                modules={},
            )

    # ---- 3. Module list endpoint ----

    @app.get(
        "/modules",
        response_model=list[ModuleInfo],
        tags=["System"],
        summary="Get module list",
        description="Returns all available modules and their loading status.",
    )
    async def list_modules(token: str = Depends(verify_token)):
        """Return information about all available modules."""
        agent = _get_agent()
        loaded_modules = agent.list_modules()

        result = []
        for name, path in AVAILABLE_MODULES.items():
            desc = path
            if name in agent.modules:
                mod = agent.modules[name]
                desc = mod.description or path
            result.append(ModuleInfo(
                name=name,
                description=desc,
                loaded=name in loaded_modules,
            ))
        return result

    # ---- 4. Clear conversation endpoint ----

    @app.post(
        "/clear",
        tags=["Chat"],
        summary="Clear conversation history",
        description="Clears the conversation history for the specified session (or the default session).",
    )
    async def clear_conversation(
        session_id: str | None = None,
        token: str = Depends(verify_token),
    ):
        """Clear conversation history."""
        agent = _get_agent(session_id)
        agent.clear_conversation()
        return {"message": "Conversation history cleared", "session_id": session_id or "default"}

    # ---- 5. Mode switch endpoint ----

    @app.post(
        "/mode",
        response_model=ModeResponse,
        tags=["Mode"],
        summary="Switch agent mode",
        description="Switch the agent's authorization mode. Supported: interactive, task, continuous. For continuous mode, use POST /runner/start instead.",
    )
    async def set_mode(
        request: ModeRequest,
        token: str = Depends(verify_token),
    ):
        """Switch agent mode."""
        if request.mode == "continuous":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "use_runner_endpoint",
                    "message": "For continuous mode, use POST /runner/start which handles mode switching and trigger setup automatically.",
                }
            )

        agent = _get_agent(request.session_id)
        try:
            await asyncio.to_thread(agent.set_mode, request.mode)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail={"error": "mode_switch_failed", "message": str(e)})

        config = agent.config
        return ModeResponse(
            mode=agent.mode,
            config={
                "max_react_steps": config.max_react_steps,
                "max_duplicate_actions": config.max_duplicate_actions,
                "react_timeout": config.react_timeout,
                "max_observation_tokens": config.max_observation_tokens,
            },
        )

    @app.get(
        "/mode",
        response_model=ModeResponse,
        tags=["Mode"],
        summary="Get current agent mode",
        description="Returns the current agent mode and mode-related config values.",
    )
    async def get_mode(
        session_id: str | None = None,
        token: str = Depends(verify_token),
    ):
        """Get current agent mode."""
        agent = _get_agent(session_id)
        config = agent.config
        return ModeResponse(
            mode=agent.mode,
            config={
                "max_react_steps": config.max_react_steps,
                "max_duplicate_actions": config.max_duplicate_actions,
                "react_timeout": config.react_timeout,
                "max_observation_tokens": config.max_observation_tokens,
            },
        )

    # ---- 6. Abort endpoint ----

    @app.post(
        "/abort",
        response_model=AbortResponse,
        tags=["Mode"],
        summary="Abort current task",
        description="Send an abort signal to the agent. The current atomic operation will complete, but no further operations will execute.",
    )
    async def abort_task(
        session_id: str | None = None,
        token: str = Depends(verify_token),
    ):
        """Abort the current task."""
        agent = _get_agent(session_id)
        agent.abort()
        return AbortResponse(success=True, message="Abort signal sent")

    # ---- 7. Runner endpoints (continuous mode) ----

    @app.post(
        "/runner/start",
        tags=["Continuous"],
        summary="Start continuous mode runner",
        description="Switch to continuous mode and start a runner with the specified trigger. Auto-approves authorization prompts.",
    )
    async def start_runner(
        request: RunnerStartRequest,
        token: str = Depends(verify_token),
    ):
        """Start a ContinuousRunner with a trigger."""
        from llamagent.core.runner import ContinuousRunner

        sid = request.session_id or "default"
        agent = _get_agent(sid)

        if sid in runner_sessions:
            raise HTTPException(
                status_code=400,
                detail={"error": "runner_already_active", "message": "Runner already active for this session. Stop it first."},
            )

        # Switch to continuous mode
        try:
            await asyncio.to_thread(agent.set_mode, "continuous")
        except (ValueError, RuntimeError) as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "mode_switch_failed", "message": str(e)},
            )

        trigger = _make_trigger(request)
        runner = ContinuousRunner(agent, [trigger], poll_interval=1.0)

        runner_thread = threading.Thread(target=runner.run, daemon=True)
        runner_thread.start()

        runner_sessions[sid] = (runner, runner_thread)
        return {"status": "started", "session_id": sid}

    @app.post(
        "/runner/stop",
        tags=["Continuous"],
        summary="Stop continuous mode runner",
        description="Stop the active runner and switch back to interactive mode.",
    )
    async def stop_runner(
        session_id: str | None = None,
        token: str = Depends(verify_token),
    ):
        """Stop the ContinuousRunner for a session."""
        sid = session_id or "default"
        entry = runner_sessions.pop(sid, None)
        if not entry:
            raise HTTPException(
                status_code=400,
                detail={"error": "no_active_runner", "message": "No active runner for this session"},
            )

        runner, thread = entry
        runner.stop()
        thread.join(timeout=5)

        tasks_completed = len(runner.get_log())

        # Switch back to interactive mode
        agent = _get_agent(sid)
        try:
            await asyncio.to_thread(agent.set_mode, "interactive")
        except Exception:
            pass  # Best effort

        return {"status": "stopped", "tasks_completed": tasks_completed, "session_id": sid}

    @app.post(
        "/inject",
        response_model=InjectResponse,
        tags=["Continuous"],
        summary="Inject a message into running agent",
        description="Send a message to the agent while a continuous runner is active. The runner pauses to handle the injected message.",
    )
    async def inject_message(
        request: InjectRequest,
        token: str = Depends(verify_token),
    ):
        """Inject a message into the active runner."""
        sid = request.session_id or "default"
        entry = runner_sessions.get(sid)
        if not entry:
            raise HTTPException(
                status_code=400,
                detail={"error": "no_active_runner", "message": "No active runner for this session"},
            )

        runner, _ = entry
        try:
            reply = await asyncio.to_thread(runner.inject, request.message)
        except RuntimeError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": "inject_failed", "message": str(e)},
            )

        return InjectResponse(reply=reply, session_id=sid)

    @app.get(
        "/runner/status",
        response_model=RunnerStatusResponse,
        tags=["Continuous"],
        summary="Get runner status",
        description="Check whether a continuous runner is active and how many tasks have been completed.",
    )
    async def get_runner_status(
        session_id: str | None = None,
        token: str = Depends(verify_token),
    ):
        """Get the status of the runner for a session."""
        sid = session_id or "default"
        entry = runner_sessions.get(sid)
        if entry:
            runner, _ = entry
            return RunnerStatusResponse(
                active=True,
                tasks_completed=len(runner.get_log()),
                session_id=sid,
            )
        return RunnerStatusResponse(
            active=False,
            tasks_completed=0,
            session_id=sid,
        )

    @app.get(
        "/runner/log",
        response_model=RunnerLogResponse,
        tags=["Continuous"],
        summary="Get runner task log",
        description="Get the last N task log entries from the active runner.",
    )
    async def get_runner_log(
        session_id: str | None = None,
        limit: int = 50,
        token: str = Depends(verify_token),
    ):
        """Get task log entries from the runner."""
        sid = session_id or "default"
        entry = runner_sessions.get(sid)
        if not entry:
            raise HTTPException(
                status_code=400,
                detail={"error": "no_active_runner", "message": "No active runner for this session"},
            )

        runner, _ = entry
        log = runner.get_log()[-limit:]
        entries = [
            RunnerLogEntry(
                input=e.input,
                output=e.output,
                status=e.status,
                duration=round(e.duration, 2),
            )
            for e in log
        ]
        return RunnerLogResponse(entries=entries, session_id=sid)

    # ---- 8. File upload endpoint ----

    @app.post(
        "/upload",
        response_model=UploadResponse,
        tags=["Knowledge Base"],
        summary="Upload files to knowledge base",
        description="Upload .txt / .md / .pdf files to add to the knowledge base for retrieval during conversations.",
    )
    async def upload_file(
        files: list[UploadFile] = File(..., description="Files to upload"),
        token: str = Depends(verify_token),
    ):
        """Handle file uploads."""
        agent = _get_agent()

        if not agent.has_module("retrieval"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "retrieval_not_loaded",
                    "message": "Retrieval module is not loaded, cannot process file uploads",
                }
            )

        processed = 0
        for file in files:
            tmp_path = None
            try:
                content = await file.read()
                filename = file.filename or "unknown"

                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode='wb', suffix=f"_{filename}", delete=False
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                retrieval_module = agent.get_module("retrieval")
                await asyncio.to_thread(retrieval_module.load_documents, tmp_path)

                processed += 1

            except Exception as e:
                logger.error("File upload processing failed: %s - %s", file.filename, e)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return UploadResponse(
            message=f"Successfully processed {processed}/{len(files)} file(s)",
            files_processed=processed,
        )

    # ---- 9. Session browsing endpoints ----

    @app.get(
        "/sessions",
        tags=["Sessions"],
        summary="List saved sessions",
        description="Scan persistence directory for saved conversation sessions.",
    )
    async def get_sessions(
        session_id: str | None = None,
        token: str = Depends(verify_token),
    ):
        """List saved sessions."""
        agent = _get_agent(session_id)
        from llamagent.interfaces.sessions import list_sessions
        sessions = list_sessions(agent)
        return {"sessions": [
            {"persona_id": s["persona_id"], "turns": s["turns"],
             "updated_at": s["updated_at"], "preview": s["preview"]}
            for s in sessions
        ]}

    @app.delete(
        "/sessions/{persona_id}",
        tags=["Sessions"],
        summary="Delete a session",
        description="Delete a saved session by persona ID. Cannot delete the active session.",
    )
    async def remove_session(
        persona_id: str,
        session_id: str | None = None,
        token: str = Depends(verify_token),
    ):
        """Delete a session file."""
        agent = _get_agent(session_id)
        from llamagent.interfaces.sessions import delete_session
        filename = f"{persona_id}.json"
        success = delete_session(agent, filename)
        if not success:
            raise HTTPException(400, "Cannot delete session (active or not found)")
        return {"deleted": persona_id}

    # ---- 6. WebSocket streaming chat ----

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket streaming chat.

        Protocol:
        1. After connecting, if auth is configured, client sends {"type": "auth", "token": "<key>"}
        2. Client sends {"type": "message", "content": "Hello"}
        3. Server streams back {"type": "chunk", "content": "..."}
        4. Finally sends {"type": "done", "content": "full reply"}
        5. On error, returns {"type": "error", "content": "error message"}
        """
        await websocket.accept()
        logger.info("WebSocket client connected")

        # Authentication (if token is configured)
        if API_AUTH_TOKEN:
            try:
                auth_data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=5.0
                )

                if (auth_data.get("type") != "auth"
                        or auth_data.get("token") != API_AUTH_TOKEN):
                    await websocket.send_json({
                        "type": "error",
                        "content": "Authentication failed, please send a valid API Key",
                    })
                    await websocket.close(code=4001)
                    return

                await websocket.send_json({
                    "type": "auth_ok",
                    "content": "Authentication successful",
                })

            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "error",
                    "content": "Authentication timed out, please send credentials within 5 seconds of connecting",
                })
                await websocket.close(code=4002)
                return

        agent = _get_agent()

        try:
            while True:
                data = await websocket.receive_json()

                if data.get("type") != "message":
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Unknown message type: {data.get('type')}",
                    })
                    continue

                user_message = data.get("content", "")
                if not user_message.strip():
                    await websocket.send_json({
                        "type": "error",
                        "content": "Message content cannot be empty",
                    })
                    continue

                try:
                    if hasattr(agent, 'chat_stream') and agent.mode == "interactive":
                        # Real streaming: producer thread puts chunks into queue,
                        # event loop reads and sends immediately
                        import queue
                        chunk_queue: queue.Queue = queue.Queue()
                        _DONE = object()

                        def _produce():
                            try:
                                for chunk in agent.chat_stream(user_message):
                                    chunk_queue.put(chunk)
                            except Exception as e:
                                chunk_queue.put(e)
                            finally:
                                chunk_queue.put(_DONE)

                        loop = asyncio.get_event_loop()
                        loop.run_in_executor(None, _produce)

                        reply = ""
                        while True:
                            item = await asyncio.to_thread(chunk_queue.get)
                            if item is _DONE:
                                break
                            if isinstance(item, Exception):
                                raise item
                            reply += item
                            await websocket.send_json({
                                "type": "chunk",
                                "content": item,
                            })
                    else:
                        # Non-streaming fallback
                        reply = await asyncio.to_thread(agent.chat, user_message)
                        await websocket.send_json({
                            "type": "chunk",
                            "content": reply,
                        })

                    await websocket.send_json({
                        "type": "done",
                        "content": reply,
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error processing message: {e}",
                    })

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error("WebSocket error: %s", e, exc_info=True)

    # --- Global exception handler ---

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Catch-all exception handler: prevents uncaught exceptions from leaking stack traces to users."""
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Internal server error, please try again later",
                "detail": str(exc) if os.getenv("DEBUG") else None,
            }
        )

    return app


# ============================================================
# Launch entry point
# ============================================================

# Module-level app cache — stored here after launch_api_server creates it,
# so uvicorn multi-worker mode can find it via string reference
_cached_app = None
_cached_module_names = None
_cached_persona_name = None


def launch_api_server(
    module_names: list[str] | None = None,
    persona_name: str | None = None,
    host: str | None = None,
    port: int | None = None,
):
    """
    Launch the API server.

    Args:
        module_names: List of modules to load
        persona_name: Persona name
        host: Bind address (defaults to API_HOST env var, falls back to 0.0.0.0)
        port: Listening port (defaults to API_PORT env var, falls back to 8000)
    """
    if not HAS_FASTAPI:
        print(
            "Error: FastAPI is not installed!\n"
            "Please run: pip install fastapi uvicorn[standard]\n"
            "Then try again."
        )
        return

    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed!\n"
            "Please run: pip install uvicorn[standard]\n"
            "Then try again."
        )
        return

    # Cache parameters for uvicorn multi-worker mode factory function
    global _cached_module_names, _cached_persona_name, _cached_app
    _cached_module_names = module_names
    _cached_persona_name = persona_name

    host = host or os.getenv("API_HOST", "0.0.0.0")
    port = port or int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))

    print(f"\n{'='*50}")
    print(f"  LlamAgent API")
    print(f"  Address: http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"{'='*50}\n")

    if workers > 1:
        # Multi-worker mode: uvicorn needs to find the app via string reference,
        # using a factory function so each worker process creates its own app instance
        uvicorn.run(
            "llamagent.interfaces.api_server:_create_app_for_uvicorn",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            factory=True,
        )
    else:
        # Single worker mode: pass the app object directly, simpler
        app = create_api_server(module_names, persona_name)
        _cached_app = app
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )


def _create_app_for_uvicorn():
    """Factory function for uvicorn multi-worker mode: each worker creates its own app."""
    return create_api_server(_cached_module_names, _cached_persona_name)


def main():
    """python -m llamagent.interfaces.api_server entry point."""
    launch_api_server()


if __name__ == "__main__":
    main()
