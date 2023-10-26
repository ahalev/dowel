import os
import warnings
from pathlib import Path


def set_wandb_env_keys(api_key=None, username=None, api_key_file=None, search_paths=('.', ), errors='warn'):
    api_key = _get_wandb_api_key(api_key, api_key_file, search_paths, errors)

    if not api_key:
        return False

    os.environ['WANDB_API_KEY'] = api_key
    os.environ['WANDB_USERNAME'] = username

    return True


def _get_wandb_api_key(api_key, api_key_file, search_paths, errors):
    assert errors in ('ignore', 'warn', 'raise'), "errors must be one of 'ignore', 'warn', 'raise'."

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
            try:
                api_key = os.environ['WANDB_API_KEY']
            except KeyError:
                nlnt = '\n\t'
                msg = f"No such file(s): {nlnt.join(str(x.resolve()) for x in key_files)}"

                if errors == 'ignore':
                    pass
                elif errors == 'warn':
                    warnings.warn(msg)
                else:
                    raise FileNotFoundError(msg)
    else:
        api_key = os.environ.get('WANDB_API_KEY')

    return api_key
