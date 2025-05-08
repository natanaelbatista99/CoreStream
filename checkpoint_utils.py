# checkpoint_utils.py
import os
import json
import pickle

MAX_VERSIONS = 3

def get_model_versions(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return []

    prefix = f"corestream_model_v"
    
    return sorted(
        [f for f in os.listdir(checkpoint_path) if f.startswith(prefix) and f.endswith(".pkl")],
        key=lambda x: int(x.split("_v")[-1].split(".pkl")[0])
    )

def get_next_version(checkpoint_path):
    versions = get_model_versions(checkpoint_path)

    print(versions)

    return int(versions[-1].split("_v")[-1].split(".pkl")[0]) + 1 if versions else 1

def save_checkpoint(corestream_obj, index, version, checkpoint_path):
    model_filename = f"corestream_model_v{version}.pkl"
    model_path     = os.path.join(checkpoint_path, model_filename)

    with open(model_path, "wb") as f:
        pickle.dump(corestream_obj, f)

    checkpoint = {
        "last_index": index,
        "model_path": model_path,
        "version": version
    }

    checkjson_path = os.path.join(checkpoint_path, f"checkpoint.json")

    with open(checkjson_path, "w") as f:
        json.dump(checkpoint, f)

    print(f"Checkpoint salved (version {version})")
    cleanup_old_versions(checkpoint_path)

def load_checkpoint(checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path, f"checkpoint.json")

    if not os.path.exists(checkpoint_path):
        return None, None, None

    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)

    model_path = checkpoint["model_path"]
    version    = checkpoint["version"]
    last_index = checkpoint["last_index"]

    with open(model_path, "rb") as f:
        corestream_obj = pickle.load(f)

    print(f"Checkpoint loaded: version {version}, index {last_index}")

    return corestream_obj, last_index, version

def cleanup_old_versions(checkpoint_path):
    versions = get_model_versions(checkpoint_path)

    if len(versions) > MAX_VERSIONS:
        for old_file in versions[:-MAX_VERSIONS]:
            os.remove(os.path.join(checkpoint_path, old_file))
            print(f"Old version removed: {old_file}")

def remove_checkpoint(checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path, f"checkpoint.json")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint removed.")