import asyncio
import json

import pytest
from fastapi import HTTPException
from vllm import SamplingParams

from llumnix.entrypoints.vllm.v1_api_server import build_app
from llumnix.entrypoints.vllm.v1_api_server import build_arg_parser


class _Output:
    def __init__(self, text):
        self.outputs = [type("Completion", (), {"text": text})()]


class _Engine:
    def __init__(self):
        self.calls = []
        self.shutdown_called = False
        self.released = []

    def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params, request_id))

        async def outputs():
            yield _Output(" world")

        return outputs()

    async def abort(self, request_id):
        self.calls.append(("abort", request_id))

    def shutdown(self):
        self.shutdown_called = True

    def release_request(self, request_id):
        self.released.append(request_id)


class _FailingEngine(_Engine):
    def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params, request_id))
        raise RuntimeError("engine startup failed")


def test_v1_cli_registers_complete_vllm_surface():
    parser = build_arg_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--quantization" in options
    assert "--kv-transfer-config" in options
    assert "--tensor-parallel-size" in options
    args = parser.parse_args(["--model", "dummy", "--max-num-seqs", "2"])
    assert args.max_num_seqs == 2
    assert args.dtype == "float16"


def test_v1_adapter_direct_generate_abort_uses_internal_alias(monkeypatch):
    from llumnix.backends.vllm.v1_engine import V1EngineAdapter

    adapter = object.__new__(V1EngineAdapter)
    adapter.engine_args = type("Args", (), {})()
    adapter.engine = _Engine()
    adapter._request_id_aliases = {}
    adapter.requests = {}
    adapter.running = []
    adapter.engine_args.kv_transfer_config = type(
        "KV", (), {"kv_role": "kv_producer"}
    )()
    monkeypatch.setattr(
        "llumnix.backends.vllm.v1_engine.p2p_connector_enabled", lambda _: True
    )
    monkeypatch.setattr(
        "llumnix.backends.vllm.v1_engine.decorate_p2p_request_id",
        lambda request_id, address: f"{request_id}@{address}",
    )
    adapter.generate("hello", object(), "public", "10.0.0.2:9")
    assert adapter._request_id_aliases["public"] == "public@10.0.0.2:9"
    asyncio.run(adapter.abort("public"))
    assert ("abort", ("public@10.0.0.2:9",)) in adapter.engine.calls


def test_v1_adapter_abort_request_cleans_public_alias(monkeypatch):
    from llumnix.backends.vllm.v1_engine import V1EngineAdapter

    adapter = object.__new__(V1EngineAdapter)
    adapter.engine = _Engine()
    adapter._request_id_aliases = {"public": "internal"}
    adapter.requests = {"public": (None, 0)}
    adapter.running = ["public"]
    async def cancel():
        await adapter.abort_request("public")

    asyncio.run(cancel())
    assert ("abort", ("internal",)) in adapter.engine.calls
    assert not adapter.requests
    assert not adapter._request_id_aliases
    assert not adapter.running


def test_v1_adapter_preserves_shared_pd_request_id(monkeypatch):
    from llumnix.backends.vllm.v1_engine import V1EngineAdapter

    adapter = object.__new__(V1EngineAdapter)
    adapter.engine_args = type("Args", (), {})()
    adapter.engine_args.kv_transfer_config = type("KV", (), {"kv_role": "kv_producer"})()
    adapter.engine = _Engine()
    adapter._request_id_aliases = {}
    adapter.requests = {}
    adapter.running = []
    monkeypatch.setattr("llumnix.backends.vllm.v1_engine.p2p_connector_enabled", lambda _: True)
    shared_id = "public___decode_addr_10.0.0.2:9______prefill_addr_10.0.0.3:9___"
    adapter.add_request(
        "public", None, float("inf"), "prompt", object(),
        llumnix_p2p_request_id=shared_id,
        llumnix_kv_decode_address="10.0.0.2:9",
    )
    assert adapter._request_id_aliases["public"] == shared_id
    assert adapter.engine.calls[0][2] == shared_id


def test_v1_adapter_rolls_back_bookkeeping_when_engine_start_fails():
    from llumnix.backends.vllm.v1_engine import V1EngineAdapter

    adapter = object.__new__(V1EngineAdapter)
    adapter.engine_args = type("Args", (), {"kv_transfer_config": None})()
    adapter.engine = _FailingEngine()
    adapter._request_id_aliases = {}
    adapter.requests = {}
    adapter.running = []
    with pytest.raises(RuntimeError, match="startup failed"):
        adapter.add_request("failed", None, 0, "prompt", object())
    assert not adapter.requests
    assert not adapter._request_id_aliases
    assert not adapter.running


class _Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return dict(self.body)

    async def is_disconnected(self):
        return False


class _RawRequest(_Request):
    async def json(self):
        return ["not", "an", "object"]


class _LegacyClient:
    def __init__(self):
        self.calls = []
        self.aborts = []

    async def generate(self, prompt, params, request_id):
        self.calls.append((prompt, params, request_id))

        class _Stream:
            async def generator(_self):
                yield type("Output", (), {
                    "prompt": prompt,
                    "outputs": [type("Completion", (), {"text": " world"})()],
                })()
        return _Stream()

    async def abort(self, request_id):
        self.aborts.append(request_id)


def _route(app, path, method):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == path and method in route.methods
    )


def test_main_api_validates_requests_and_preserves_public_request_id(monkeypatch):
    import llumnix.entrypoints.vllm.api_server as main_api

    client = _LegacyClient()
    monkeypatch.setattr(main_api, "llumnix_client", client)
    with pytest.raises(HTTPException) as error:
        asyncio.run(main_api.generate(_RawRequest({})))
    assert error.value.status_code == 400

    response = asyncio.run(main_api.generate(_Request({
        "prompt": "hello", "request_id": "public-id", "max_tokens": 1,
    })))
    assert response.status_code == 200
    assert json.loads(response.body) == {"text": ["hello world"]}
    assert client.calls[0][2] == "public-id"


def test_main_api_instance_list_exposes_v1_heterogeneous_state(monkeypatch):
    import llumnix.entrypoints.vllm.api_server as main_api
    from llumnix.instance_info import InstanceInfo

    info = InstanceInfo(
        instance_id="corex-v1", num_running_requests=2, num_waiting_requests=1,
        node_id="node-corex", node_ip="10.31.10.62",
        gpu_memory_total_bytes=32 * 1024**3,
        gpu_memory_free_bytes=20 * 1024**3,
        compute_capacity=1.5,
        dispatch_load_metric=0.25,
        kv_cache_block_hashes=frozenset({b"a", b"b"}),
    )

    class _InstanceClient:
        async def get_all_instances_info(self):
            return [info]

    monkeypatch.setattr(main_api, "llumnix_client", _InstanceClient())
    response = asyncio.run(main_api.get_instance_list())
    payload = json.loads(response.body)
    instance = payload["data"][0]
    assert instance["gpu_memory_total_bytes"] == 32 * 1024**3
    assert instance["gpu_memory_free_bytes"] == 20 * 1024**3
    assert instance["compute_capacity"] == 1.5
    assert instance["dispatch_load_metric"] == 0.25
    assert instance["kv_cache_affinity_blocks"] == 2
    assert instance["node_id"] == "node-corex"
    assert instance["node_ip"] == "10.31.10.62"


def test_main_api_instance_list_serializes_idle_load_sentinel(monkeypatch):
    import llumnix.entrypoints.vllm.api_server as main_api
    from llumnix.instance_info import InstanceInfo

    class _InstanceClient:
        async def get_all_instances_info(self):
            return [InstanceInfo(instance_id="idle-v1")]

    monkeypatch.setattr(main_api, "llumnix_client", _InstanceClient())
    response = asyncio.run(main_api.get_instance_list())
    assert response.status_code == 200
    assert json.loads(response.body)["data"][0]["dispatch_load_metric"] is None


def test_llumlet_aborts_engine_when_v1_output_stream_fails():
    from llumnix.llumlet.llumlet import Llumlet

    class _Backend:
        def __init__(self):
            self.requests = {"public": (None, 0)}
            self._request_id_aliases = {"public": "internal"}
            self.running = ["public"]
            self.aborts = []

        async def abort(self, request_id):
            self.aborts.append(request_id)

    async def broken_stream():
        raise RuntimeError("connector failed")
        yield None

    llumlet = object.__new__(Llumlet)
    llumlet.backend_engine = backend = _Backend()
    asyncio.run(llumlet._forward_v1_outputs("public", None, broken_stream()))
    assert backend.aborts == ["public"]
    assert not backend.requests
    assert not backend._request_id_aliases
    assert not backend.running


def test_client_abort_cancels_fallback_instance_when_manager_is_down():
    from llumnix.entrypoints.vllm.client import LlumnixClientVLLM
    import ray

    class _RemoteCall:
        def __init__(self, fn): self.fn = fn
        async def remote(self, *args): return await self.fn(*args)

    class _DeadManager:
        async def abort(_request_id):
            raise ray.exceptions.RayActorError("manager down")

    class _Fallback:
        async def abort(_request_id):
            calls.append(_request_id)

    calls = []
    client = object.__new__(LlumnixClientVLLM)
    client.manager = type("M", (), {"abort": _RemoteCall(_DeadManager.abort)})()
    client.instances = {"i0": type("I", (), {"abort": _RemoteCall(_Fallback.abort)})()}
    client._fallback_request_instance = {"r1": "i0"}
    client.instance_num_requests = {"i0": 1}
    asyncio.run(client.abort("r1"))
    assert calls == ["r1"]
    assert client.instance_num_requests["i0"] == 0


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
    assert engine.released == ["r1"]


def test_v1_api_rejects_invalid_request_without_engine_call():
    engine = _Engine()
    generate = _route(build_app(engine), "/generate", "POST")
    with pytest.raises(HTTPException) as error:
        asyncio.run(generate(_Request({"max_tokens": 1})))
    assert error.value.status_code == 400
    assert not engine.calls

    with pytest.raises(HTTPException) as error:
        asyncio.run(generate(_RawRequest({})))
    assert error.value.status_code == 400


def test_v1_api_streaming_wire_format_without_model():
    engine = _Engine()
    app = build_app(engine)
    generate = _route(app, "/generate", "POST")
    response = asyncio.run(
        generate(_Request({"prompt": "hi", "stream": True, "request_id": "r-stream", "max_tokens": 1}))
    )
    assert response.media_type == "application/octet-stream"

    async def collect():
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(collect())
    assert json.loads(chunks[0].rstrip(b"\0")) == {"text": ["hi world"]}
    assert engine.released == ["r-stream"]


def test_v1_api_stream_disconnect_aborts_request():
    engine = _Engine()
    app = build_app(engine)
    generate = _route(app, "/generate", "POST")
    response = asyncio.run(
        generate(_Request({"prompt": "hi", "stream": True, "request_id": "r-disconnect"}))
    )

    async def close_early():
        iterator = response.body_iterator
        await anext(iterator)
        await iterator.aclose()

    asyncio.run(close_early())
    assert ("abort", "r-disconnect") in engine.calls


def test_v1_api_lifespan_shuts_down_engine():
    engine = _Engine()
    app = build_app(engine)

    async def exercise():
        async with app.router.lifespan_context(app):
            assert not engine.shutdown_called

    asyncio.run(exercise())
    assert engine.shutdown_called
