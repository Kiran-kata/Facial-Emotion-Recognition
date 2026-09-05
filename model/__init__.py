from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    'fer': {
        'h5': MODEL_DIR / 'fer_model_best.h5',
        'json': MODEL_DIR / 'fer_model_best.json',
    },
    'ferplus_mv': {
        'h5': MODEL_DIR / 'ferplus_model_mv_best.h5',
        'json': MODEL_DIR / 'ferplus_model_mv_best.json',
    },
    'ferplus_pd': {
        'h5': MODEL_DIR / 'ferplus_model_pd_best.h5',
        'json': MODEL_DIR / 'ferplus_model_pd_best.json',
        'tflite': MODEL_DIR / 'ferplus_model_pd_best.tflite',
    },
}


def get_available_models():
    """Return only the model entries that exist on disk."""
    available = {}
    for name, files in MODEL_FILES.items():
        present = {key: str(path) for key, path in files.items() if path.exists()}
        if present:
            available[name] = present
    return available


def get_default_tflite_model():
    """Return the default production TFLite model path used by the app."""
    pd_model = MODEL_FILES['ferplus_pd'].get('tflite')
    if pd_model and pd_model.exists():
        return str(pd_model)
    return None


__all__ = ['MODEL_DIR', 'MODEL_FILES', 'get_available_models', 'get_default_tflite_model']
