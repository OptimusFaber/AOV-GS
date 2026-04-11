import os
import sys
sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter

from src.data.pose_loader import habitat_pose_conversion, PoseLoader
from src.naruto.cfg_loader import load_cfg, save_cfg_to_json
from src.planner import init_planner
from src.slam import init_SLAM_model
from src.simulator import init_simulator
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step
from src.visualization import init_visualizer
import argparse

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
    parser.add_argument("--stage", type=str, default='final',
                        help="ONLY for SplaTAM result evaluation ")
    parser.add_argument("--step", type=int, default=1100,
                        help="ONLY for SplaTAM result evaluation ")
    args = parser.parse_args()
    return args


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
    save_cfg_to_json(main_cfg, os.path.join(main_cfg.dirs.result_dir, 'main_cfg.json'))
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    ##################################################
    ### Fix random seed
    ##################################################
    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)

    ##################################################
    ### initialize logger
    ##################################################
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(f'{log_savedir}')

    ##################################################
    ### initialize visualizer (if enable_vis=1)
    ##################################################
    visualizer = None
    if main_cfg.visualizer.vis_rgbd:
        info_printer("Initialize Visualizer (enable_vis=1)...", 0, "Initialization")
        visualizer = init_visualizer(main_cfg, info_printer)

    ##################################################
    ### initialize SLAM module
    ##################################################
    slam = init_SLAM_model(main_cfg, info_printer, logger)
    slam.load_params_by_step(step=args.step,stage=args.stage)

    ##################################################
    ### Save Final Mesh and Checkpoint
    ##################################################
    # slam.print_and_save_result()
    if main_cfg.general.dataset in ['MP3D']:
        slam.eval_semantic_result_mp3d(eval_dir_suffix=args.stage, ignore_first_frame=True, save_frames=True)
    else:
        slam.eval_semantic_result(
            eval_dir_suffix=args.stage,
            ignore_first_frame=True,
            save_frames=True,
            visualizer=visualizer,
            main_cfg=main_cfg if visualizer is not None else None,
        )


    ##################################################
    ### Runtime Analysis
    ##################################################
    # timer.time_analysis()
