"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import argparse
import json
import os
import types

import mmengine

def override_cfg(
        args: argparse.Namespace,
        cfg : mmengine.Config
    ) -> mmengine.Config:
    """override configuration

    Args:
        args: arguments
        cfg : configuration

    Returns:
        cfg : updated configuration
    """
    if hasattr(args, "seed") and args.seed is not None:
        ### random seed ###
        cfg.general.seed = args.seed

    if hasattr(args, "result_dir") and args.result_dir is not None:
        ### output/result directory ###
        cfg.dirs.result_dir = args.result_dir

    if hasattr(args, "enable_vis") and args.enable_vis is not None:
        ### output/result directory ###
        enable_vis = args.enable_vis == 1
        cfg.visualizer.vis_rgbd = enable_vis

    if hasattr(args, "use_clip") and args.use_clip is not None:
        if args.use_clip == 0:
            # Disable CLIP by removing the clip section entirely.
            # activesgm.py checks `getattr(main_cfg, 'clip', None)`.
            if hasattr(cfg, 'clip'):
                del cfg['clip']
        # use_clip == 1 → keep whatever the config file defines (no action needed)

    return cfg


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to run NARUTO."
        )
    parser.add_argument("--cfg", type=str, default="configs/default.py",
                        help="NARUTO config")
    parser.add_argument("--result_dir", type=str, default=None, 
                        help="result directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; also used as the initial pose idx for Replica")
    parser.add_argument("--enable_vis", type=int, default=None,
                        help="enable visualization. 1: True, 0: False")
    parser.add_argument("--use_clip", type=int, default=None,
                        help="enable open-vocabulary CLIP index. 1: enable, 0: disable. "
                             "Overrides the [clip] section in the config file.")
    parser.add_argument("--stage", type=str, default='final',
                        help="ONLY for SplaTAM result evaluation ")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode: save keyframe RGB images to result_dir/keyframes/")
    # Accept both optional flag and positional scene
    parser.add_argument("--scene", type=str, default=None,
                        help="Override scene name (e.g., office0).")
    parser.add_argument("scene_positional", nargs="?", default=None,
                        help="Scene name as positional arg (e.g., office0), OR pass path/to/config.py (same as --cfg).")
    args = parser.parse_args()
    # `python activesgm.py configs/.../foo.py` — users often omit `--cfg`; first positional is then
    # a config file, not a scene name. Without this, load_cfg() treats it as `args.scene` and breaks
    # dirs.cfg_dir (e.g. .../ActiveOpenSem.py/habitat.py).
    if args.scene_positional and str(args.scene_positional).endswith(".py"):
        args.cfg = args.scene_positional
        args.scene_positional = None
    # Merge positional scene into named scene if provided
    if args.scene is None and args.scene_positional is not None:
        args.scene = args.scene_positional
    return args


def clean_config_for_json(cfg_dict):
    """Remove non-serializable objects (like modules) from config dict"""
    if isinstance(cfg_dict, mmengine.Config):
        cfg_dict = dict(cfg_dict)
    
    if isinstance(cfg_dict, dict):
        cleaned = {}
        for key, value in cfg_dict.items():
            if isinstance(value, types.ModuleType):
                # Replace module with its name as string
                cleaned[key] = f"<module: {value.__name__}>"
            elif isinstance(value, (types.FunctionType, types.BuiltinFunctionType, type)):
                # Replace functions and classes with their names
                cleaned[key] = f"<{type(value).__name__}: {getattr(value, '__name__', str(value))}>"
            elif isinstance(value, mmengine.Config):
                cleaned[key] = clean_config_for_json(dict(value))
            elif isinstance(value, dict):
                cleaned[key] = clean_config_for_json(value)
            elif isinstance(value, (list, tuple)):
                cleaned_list = []
                for item in value:
                    if isinstance(item, types.ModuleType):
                        cleaned_list.append(f"<module: {item.__name__}>")
                    elif isinstance(item, (types.FunctionType, types.BuiltinFunctionType, type)):
                        cleaned_list.append(f"<{type(item).__name__}: {getattr(item, '__name__', str(item))}>")
                    elif isinstance(item, (dict, mmengine.Config)):
                        cleaned_list.append(clean_config_for_json(item))
                    else:
                        cleaned_list.append(item)
                cleaned[key] = cleaned_list if isinstance(value, list) else tuple(cleaned_list)
            else:
                cleaned[key] = value
        return cleaned
    else:
        return cfg_dict


def save_cfg_to_json(cfg: mmengine.Config, filepath: str):
    """Save configuration to JSON file, cleaning non-serializable objects
    
    Args:
        cfg: mmengine.Config object
        filepath: path to save JSON file
    """
    import os
    cleaned_cfg = clean_config_for_json(cfg)
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(cleaned_cfg, f, indent=4, default=str)


def load_cfg(args: argparse.Namespace) -> mmengine.Config:
    """argument parsing and load configuration

    Args:
        args: arguments

    Returns:
        cfg : configuration

    """
    cfg = mmengine.Config.fromfile(args.cfg)

    # Optional scene override: keep derived dirs in sync.
    if hasattr(args, "scene") and args.scene:
        # Update general.scene
        if hasattr(cfg, "general"):
            cfg.general.scene = args.scene
        # Update directories that depend on scene
        if hasattr(cfg, "dirs"):
            data_dir = cfg.dirs.get("data_dir", "data/")
            dataset = cfg.general.get("dataset", "Replica") if hasattr(cfg, "general") else "Replica"
            cfg.dirs.cfg_dir = os.path.join("configs", dataset, args.scene)
        if hasattr(cfg, "sim") and getattr(cfg.sim, "method", None) == "habitat":
            cfg.sim.habitat_cfg = os.path.join(cfg.dirs.get("cfg_dir", ""), "habitat.py")
        if hasattr(cfg, "planner") and hasattr(cfg.dirs, "data_dir"):
            dataset = cfg.general.get("dataset", "Replica") if hasattr(cfg, "general") else "Replica"
            cfg.planner.SLAMData_dir = os.path.join(cfg.dirs.data_dir, dataset, args.scene)

    cfg = override_cfg(args, cfg)
    return cfg
