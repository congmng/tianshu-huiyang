import asyncio
import json

from vllm import SamplingParams

from llumnix.entrypoints.vllm.v1_api_server import build_app


class _Output:
    def __init__(self, text):
        self.outputs = [type("Completion", (), {"text": text})()]


class _Engine:
    def __init__(self):
        self.calls = []
        self.shutdown_called = False

    def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params, request_id))

        async def outputs():
            yield _Output(" world")

        return outputs()

    async def abort(self, request_id):
        self.calls.append(("abort", request_id))

    def shutdown(self):
        self.shutdown_called = True


class _Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return dict(self.body)

    async def is_disconnected(self):
        return False


def _route(app, path, method):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == path and method in route.methods
    )


def test_v1_api_health_and_generate_without_model():
    engine = _Engine()
    app = build_app(engine)
    health = _route(app, "/health", "GET")
    response = asyncio.run(health())
    assert response.status_code == 200

    generate = _route(app, "/generate", "POST")
    response = asyncio.run(
        generate(_Request({"prompt": "hello", "request_id": "r1", "max_tokens": 1}))
    )
    assert response.status_code == 200
    assert json.loads(response.body) == {"text": ["hello world"]}
    assert engine.calls[0][2] == "r1"


def test_v1_api_streaming_wire_format_without_model():
    engine = _Engine()
    app = build_app(engine)
    generate = _route(app, "/generate", "POST")
    response = asyncio.run(
        generate(_Request({"prompt": "hi", "stream": True, "max_tokens": 1}))
    )
    assert response.media_type == "application/octet-stream"

    async def collect():
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(collect())
    assert json.loads(chunks[0].rstrip(b"\0")) == {"text": ["hi world"]}
