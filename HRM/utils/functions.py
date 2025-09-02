import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    if module_path.startswith(prefix):
        module = importlib.import_module(module_path)
    else:
        module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    if module_path.startswith(prefix):
        module = importlib.import_module(module_path)
    else:
        module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)
