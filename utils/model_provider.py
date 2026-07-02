import onnxruntime as ort


def providers_for(provider: str) -> list[str]:
    if provider == "CPU":
        return ["CPUExecutionProvider"]
    if provider == "GPU":
        available = ort.get_available_providers()
        if "DmlExecutionProvider" not in available:
            raise RuntimeError(f"ONNX Runtime DirectML is not available. Providers: {available}")
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"Unsupported model provider: {provider}")


def session_options_for(provider: str) -> ort.SessionOptions:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "CPU":
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 1
    return session_options
