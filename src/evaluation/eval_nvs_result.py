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


import os
import sys
sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter

from src.data.pose_loader import habitat_pose_conversion, PoseLoader
from src.naruto.cfg_loader import argument_parsing, load_cfg, save_cfg_to_json
from src.planner import init_planner
from src.slam import init_SLAM_model
from src.simulator import init_simulator
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step
from src.visualization import init_visualizer


if __name__ == "__main__":
    info_printer = InfoPrinter("ActiveSem")
    timer = Timer()

    ##################################################
    ### argument parsing and load configuration
    ##################################################
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)
    os.makedirs(main_cfg.dirs.result_dir, exist_ok=True)
    save_cfg_to_json(main_cfg, os.path.join(main_cfg.dirs.result_dir, "main_cfg.json"))
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    ##################################################
    ### Fix random seed
    ##################################################
    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(getattr(main_cfg.general, "seed", 0))

    ##################################################
    ### initialize logger
    ##################################################
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(f'{log_savedir}')
    
    ##################################################
    ### initialize SLAM module
    ##################################################
    slam = init_SLAM_model(main_cfg, info_printer, logger)
    if hasattr(slam, "load_params_by_step"):
        slam.load_params_by_step(step=0, stage=args.stage)
    else:
        slam.load_params(stage=args.stage)

    # eval_result builds eval_dir = splatam/eval_<suffix>; training uses suffix "exploration_stage_N"
    # (params live there). If user passes "eval_exploration_stage_1", strip the duplicate "eval_".
    _suffix = args.stage[5:] if args.stage.startswith("eval_") else args.stage

    ##################################################
    ### Save Final Mesh and Checkpoint
    ##################################################
    # slam.print_and_save_result()
    slam.eval_result(eval_dir_suffix=_suffix, ignore_first_frame=True, save_frames=True)

    ##################################################
    ### Runtime Analysis
    ##################################################
    # timer.time_analysis()
