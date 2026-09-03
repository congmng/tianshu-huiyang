import ast
from pathlib import Path


def test_global_serve_connects_to_configured_ray_head():
    """Global P/D launch must not create an accidental local Ray runtime."""
    source = Path("llumnix/entrypoints/vllm/serve.py").read_text()
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "connect_to_ray_cluster"
    ]
    assert len(calls) == 1
    names = {keyword.arg for keyword in calls[0].keywords}
    assert {"head_node_ip", "port", "namespace", "log_to_driver"} <= names
