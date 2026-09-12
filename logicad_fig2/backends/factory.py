"""Provider selection belongs here, never in algorithm stages."""
from .base import BackendBundle


def build_backends(config, usage_path=None, *, openai_client=None):
    roles = ["vlm", "formatter", "embedding"] + (["logic"] if config.enable_reasoner else [])
    client = openai_client
    if any(getattr(config, role + "_backend") == "openai" for role in roles) and client is None:
        from ..openai_client import OpenAIClient
        client = OpenAIClient(config, usage_path)
    result = {}
    for role in roles:
        model = getattr(config, role + "_model")
        if getattr(config, role + "_backend") == "openai":
            from .openai import OpenAIVisionBackend, OpenAITextBackend, OpenAIEmbeddingBackend
            cls = {"vlm": OpenAIVisionBackend, "embedding": OpenAIEmbeddingBackend}.get(role, OpenAITextBackend)
            backend = cls(client, model)
        elif role == "embedding":
            from .hf_embedding import HFEmbeddingBackend
            backend = HFEmbeddingBackend(model, device=config.embedding_device,
                                         revision=config.embedding_revision,
                                         local_files_only=config.local_files_only,
                                         batch_size=config.embedding_batch_size)
        else:
            from .hf_common import HFOptions
            from .hf_text import HFTextBackend
            from .hf_vlm import HFVisionBackend
            options = HFOptions(device=config.device, dtype=config.dtype, device_map=config.device_map,
                                load_in_4bit=config.load_in_4bit, load_in_8bit=config.load_in_8bit,
                                revision=getattr(config, role + "_revision"),
                                local_files_only=config.local_files_only,
                                offload_folder=config.offload_folder, max_memory=config.max_memory)
            backend = (HFVisionBackend(model, options, image_strategy=config.vlm_image_strategy)
                       if role == "vlm" else HFTextBackend(model, options))
        result["vision" if role == "vlm" else role] = backend
    return BackendBundle(**result, unload_on_switch=config.unload_on_switch)
