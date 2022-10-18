import yaml
import os
import importlib
from importlib.machinery import SourceFileLoader

data_path = os.path.dirname(os.path.realpath(__file__))


def ensure_directory(path: str):
    try:
        os.makedirs(path)
    except Exception:
        pass

    pass


def save_dataset_definition(dataset):
    save_path = os.path.join(data_path, dataset.name)
    ensure_directory(save_path)
    with open(os.path.join(save_path, "dataset.yaml"), "w") as file, open(
        os.path.join(save_path, "loader.py"), "w"
    ) as file2:
        yaml.dump(dataset, file)
        file2.writelines(dataset.loader_func_definition)


def import_loader(name: str, loader_function_name: str):
    load_path = os.path.join(data_path, name)
    loader = SourceFileLoader(
        "loader", os.path.join(load_path, "loader.py")
    ).load_module()
    return getattr(loader, loader_function_name)

def load_dataset_definition(name: str):
    load_path = os.path.join(data_path, name)
    try:
        with open(os.path.join(load_path, "dataset.yaml"), "r") as file:
            return yaml.safe_load(file)
    except IOError as e:
        raise IOError(f"Failed to load Dataset '{name}'. {e}")
