from src.models.architectures.multimodal import MultimodalModel

ARCHITECTURES = {"multimodal": MultimodalModel}


def build_model(name, **kwargs):
    return ARCHITECTURES[name](**kwargs)
