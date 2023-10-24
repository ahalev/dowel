import os
import warnings
from pathlib import Path


def set_wandb_env_keys(api_key=None, username=None, api_key_file=None, search_paths=('.', )):
    api_key = _get_wandb_api_key(api_key, api_key_file, search_paths)

    if not api_key:
        return False

    os.environ['WANDB_API_KEY'] = api_key
    os.environ['WANDB_USERNAME'] = username

    return True


def _get_wandb_api_key(api_key, api_key_file, search_paths):
    if api_key:
        if api_key_file:
            warnings.warn("Ignoring 'api_key_file' as 'api_key' was passed.")
    elif api_key_file:
        key_file = Path(api_key_file)
        if key_file.is_absolute():
            key_files = [key_file]
        else:
            key_files = [Path(search_path).joinpath(key_file) for search_path in search_paths]

        api_key = None

        for file in key_files:
            if file.exists():
                with Path(file).open('r') as f:
                    api_key = f.read()
                    api_key = api_key.partition('\n')[0]

        if api_key is None:
            nlnt = '\n\t'
            msg = f"No such file(s): {nlnt.join(str(x.resolve()) for x in key_files)}\n" \
                  f"Pass '--logging.wandb.api_key <path_to_api_key>' at the command line to give the correct " \
                  f"path to a wandb api key or pass --logging.wandb.api_key null to disable wandb." \
                  f"Default value 'wandb_api_key.txt' searches in working directory and in rgen repo root directory."

            raise FileNotFoundError(msg)
    else:
        api_key = os.environ.get('WANDB_API_KEY')

    return api_key
