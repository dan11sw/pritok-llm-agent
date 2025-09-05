from huggingface_hub import snapshot_download
from huggingface_hub import errors

import sys
import argparse
import os
import shutil
from typing import Type, Optional

ModelNames = {
 '1': 'Qwen/Qwen2.5-Coder-3B-Instruct'
}

ModelDirs = {
 '1': '/home/qwen2.5-coder'        
}

class LoaderOptions:
    __slots__ = ('clear_dir', 'revision')

    def __init__(self, clear_dir: bool = False, revision: str = "main"):
        self.clear_dir = clear_dir
        self.revision = revision


def load_model(model_name: str, local_dir: str, options: Optional[Type[LoaderOptions]] = None) -> bool:
    if options is not None:
        if options.clear_dir and os.path.exists(local_dir) and os.path.isdir(local_dir):
            try:
                shutil.rmtree(local_dir)
                print(f"Директория по пути {local_dir} удалена")
            except Exception as e:
                print(f"Ошибка при удалении папки {local_dir}: {e}")

    try:
        revision = options.revision if options is not None else "main"
        snapshot_download(repo_id=model_name, local_dir=local_dir, revision=revision)
        print(f"Модель {model_name} загружена")

        return True
    except Exception as e:
        print(f"Ошибка: {e}")

    return False

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="Пример передачи аргументов")
    parser.add_argument("--model", type=str, help="Название модели")
    parser.add_argument("--local_dir", type=str, help="Путь до данных модели")

    args = parser.parse_args()

    if args.model is None:
        print("Необходимо указать параметр --model")
        sys.exit()

    if args.local_dir is None:
        print("Необходимо указать параметр --local_dir")
        sys.exit()

    if args.model and args.local_dir:
        model_name = ModelNames[args.model] if args.model in ModelNames else args.model
        local_dir = ModelDirs[args.local_dir] if args.local_dir in ModelDirs else args.local_dir
        
        load_model(model_name, local_dir, LoaderOptions(True))




