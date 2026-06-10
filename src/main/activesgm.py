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

import json
import numpy as np
import os
import re
import sys
sys.path.append(os.getcwd())

from src.utils.display_utils import configure_headless_env
from src.utils.exploration_path_plot import save_exploration_path_topdown

configure_headless_env()
from tensorboardX import SummaryWriter
import torch

from src.naruto.cfg_loader import argument_parsing, load_cfg, save_cfg_to_json
from src.planner import init_planner
from src.slam import init_SLAM_model
from src.simulator import init_simulator
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step
from src.visualization import init_visualizer
from src.semantic import CLIPEncoder, OpenVocabIndex




if __name__ == "__main__":
    info_printer = InfoPrinter("ActiveSGM")
    timer = Timer()

    ##################################################
    ### argument parsing and load configuration
    ##################################################
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)
    info_printer(f"Result directory: {main_cfg.dirs.result_dir}", 0, "Initialization")
    # Save config to JSON (automatically cleans non-serializable objects)
    save_cfg_to_json(main_cfg, os.path.join(main_cfg.dirs.result_dir, 'main_cfg.json'))
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    ##################################################
    ### Fix random seed
    ##################################################
    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(getattr(main_cfg.general, 'seed', 0))

    ##################################################
    ### initialize logger
    ##################################################
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(f'{log_savedir}')
    
    ##################################################
    ### initialize simulator
    ##################################################
    sim = init_simulator(main_cfg, info_printer)

    ##################################################
    ### initialize SLAM module
    ##################################################
    slam = init_SLAM_model(main_cfg, info_printer, logger)
    map_iter_og = slam.config['mapping']['num_iters']

    ##################################################
    ### initialize planning module
    ##################################################
    planner = init_planner(main_cfg, info_printer)
    planner.update_sim(sim)
    # planner.init_local_planner()

    ##################################################
    ### initialize visualizer
    ##################################################
    visualizer = init_visualizer(main_cfg, info_printer)

    ##################################################
    ### initialize SAM+CLIP background extractor (optional)
    # Activated when main_cfg.slam.save_clip_features is True AND
    # a [sam_clip] section exists in the config.
    ##################################################
    sam_clip_extractor = None
    sam_clip_stats_snapshot = None
    _save_clip = getattr(main_cfg.slam, 'save_clip_features', False)
    _disable_sam_clip_env = os.getenv("ACTIVESGM_DISABLE_SAM_CLIP", "0").strip().lower() in ("1", "true", "yes", "on")
    if _disable_sam_clip_env and _save_clip:
        info_printer(
            "ACTIVESGM_DISABLE_SAM_CLIP is set — SAM+CLIP extractor disabled for this run.",
            0, "Initialization"
        )
        _save_clip = False
    sam_clip_cfg = getattr(main_cfg, 'sam_clip', None)
    if _save_clip and sam_clip_cfg is not None:
        from src.semantic.sam_clip_extractor import SAMCLIPExtractor
        _lang_feat_dir = os.path.join(main_cfg.dirs.result_dir, "language_features")
        _debug_seg_dir = (
            os.path.join(main_cfg.dirs.result_dir, "segmentframes")
            if getattr(args, 'debug', False) else None
        )
        _sam_clip_device = os.getenv("SAM_CLIP_DEVICE") or getattr(sam_clip_cfg, 'device', 'cuda:1')
        sam_clip_extractor = SAMCLIPExtractor(
            save_dir            = _lang_feat_dir,
            sam_ckpt_path       = getattr(sam_clip_cfg, 'sam_ckpt_path', 'ckpts/sam_vit_h_4b8939.pth'),
            clip_model          = getattr(sam_clip_cfg, 'clip_model', 'ViT-B-32'),
            clip_pretrained     = getattr(sam_clip_cfg, 'clip_pretrained', 'openai'),
            device              = _sam_clip_device,
            queue_size          = getattr(sam_clip_cfg, 'queue_size', 8),
            submit_timeout_s    = getattr(sam_clip_cfg, 'submit_timeout_s', 1.0),
            bbox_pad_px         = getattr(sam_clip_cfg, 'bbox_pad_px', 20),
            clip_batch_size     = getattr(sam_clip_cfg, 'clip_batch_size', 32),
            max_masks_per_frame = getattr(sam_clip_cfg, 'max_masks_per_frame', 120),
            corrclip_mask_merge = getattr(sam_clip_cfg, 'corrclip_mask_merge', True),
            corrclip_merge_sim_thresh = getattr(sam_clip_cfg, 'corrclip_merge_sim_thresh', 0.86),
            corrclip_merge_dist_px = getattr(sam_clip_cfg, 'corrclip_merge_dist_px', 80.0),
            corrclip_interclass_suppress_alpha = getattr(
                sam_clip_cfg, 'corrclip_interclass_suppress_alpha', 0.15
            ),
            corrclip_interclass_sim_thresh = getattr(
                sam_clip_cfg, 'corrclip_interclass_sim_thresh', 0.78
            ),
            corrclip_interclass_sigma_px = getattr(
                sam_clip_cfg, 'corrclip_interclass_sigma_px', 120.0
            ),
            debug_dir           = _debug_seg_dir,
            levels              = tuple(getattr(sam_clip_cfg, 'levels', ('p',))),
            save_fp16           = getattr(sam_clip_cfg, 'save_fp16', True),
        )
        sam_clip_extractor.start()
        info_printer(
            f"SAM+CLIP extractor started → {_lang_feat_dir}",
            0, "Initialization"
        )
        if _debug_seg_dir is not None:
            info_printer(
                f"Debug mode: SAM masks will be saved to {_debug_seg_dir}",
                0, "Debug"
            )
    elif _save_clip:
        info_printer(
            "slam.save_clip_features=True but no [sam_clip] section found — extractor disabled.",
            0, "Initialization"
        )

    # Dict to accumulate keyframe w2c poses for offline language-field training.
    # Saved as result_dir/keyframe_poses.json at the end.
    _kf_poses: dict = {}   # frame_id (int) → 4×4 w2c list
    _save_kf_poses = getattr(main_cfg.slam, 'save_keyframe_poses', False)

    # Debug: save keyframe RGB images to result_dir/keyframes/
    _debug_save_kf = getattr(args, 'debug', False)
    _debug_kf_dir = os.path.join(main_cfg.dirs.result_dir, "keyframes")
    if _debug_save_kf:
        os.makedirs(_debug_kf_dir, exist_ok=True)
        info_printer(f"Debug mode: keyframes will be saved to {_debug_kf_dir}", 0, "Debug")

    ##################################################
    ### initialize open-vocabulary CLIP index (optional)
    # Activated when main_cfg.clip is present in the config file.
    # The block is intentionally wrapped in try/except so that
    # existing configs without a [clip] section continue to work.
    ##################################################
    open_vocab_index = None
    clip_cfg = getattr(main_cfg, 'clip', None)
    if clip_cfg is not None:
        clip_device   = getattr(clip_cfg, 'device',       'cuda:1')
        clip_model    = getattr(clip_cfg, 'model_name',   'ViT-B-32')
        clip_pretrained = getattr(clip_cfg, 'pretrained', 'openai')
        clip_update_every = getattr(clip_cfg, 'update_every', 10)
        clip_top_k    = getattr(clip_cfg, 'top_k',        5)

        info_printer(
            f"Initializing CLIP open-vocabulary index "
            f"({clip_model} / {clip_pretrained}) on {clip_device}…",
            0, "Initialization"
        )
        clip_encoder = CLIPEncoder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=clip_device,
        )
        clip_index_path = os.path.join(
            main_cfg.dirs.result_dir, "splatam", "clip_index.pt"
        )
        open_vocab_index = OpenVocabIndex(
            clip_encoder=clip_encoder,
            update_every=clip_update_every,
            top_k=clip_top_k,
        )
        info_printer(
            f"Open-vocabulary CLIP index created. "
            f"Index will be saved alongside validation steps → {clip_index_path}",
            0, "Initialization"
        )
    else:
        info_printer(
            "No [clip] section found in config – open-vocabulary navigation disabled.",
            0, "Initialization"
        )

    ##################################################
    ### Run ActiveLang
    ##################################################
    ## load initial pose and convert from RUB to RDF (splatam)) ##
    c2w_slam = planner.load_init_pose() # RUB
    c2w_slam[:3, 1] *= -1
    c2w_slam[:3, 2] *= -1 # RDF
    c2w_slam_init = c2w_slam.clone() # RDF

    # Make the initial pose available to the planner for goal-pose conversion
    planner.c2w_slam_init = c2w_slam_init

    ## initialize exploration map in slam ##
    T_sim2slam = torch.inverse(c2w_slam_init) # RDF # transformation that takes sim-world points to slam-world-origin (i.e. first camera)
    if main_cfg.slam.enable_active_planning:
        slam.init_exploration_map(T_sim2slam)

    planner.init_data(T_sim2slam)
    if main_cfg.planner.method in [
        "active_gs", "active_gsv2", "active_gs_hybrid", "active_gs_hybrid_v3",
    ]:
        planner.init_local_planner()

    ### attach CLIP index to the planner (if available) ###
    if open_vocab_index is not None:
        planner.set_open_vocab_index(open_vocab_index)

    ### add timer for planning related timing ###
    planner.timer = timer

    # Robot trajectory in Habitat RUB (for top-down path figure).
    exploration_path_poses: list = []

    for i in range(main_cfg.general.num_iter):
    # for i in range(0, main_cfg.general.num_iter, 10):
        ##################################################
        ### update module infomation (e.g. step)
        ##################################################
        update_module_step(i, [sim, slam, planner, visualizer])

        ##################################################
        ### load pose and transform pose
        ##################################################
        if main_cfg.planner.method == "predefined_traj":
            c2w_slam = planner.update_pose(c2w_slam, i).to(c2w_slam.device) # RUB
            c2w_sim = c2w_slam.cpu().numpy().copy() # RUB
            ## convert back to RDF (splatam) ##
            c2w_slam[:3, 1] *= -1 
            c2w_slam[:3, 2] *= -1
        elif main_cfg.planner.method in [
            "active_lang", "active_gs", "active_gsv2",
            "active_gs_hybrid", "active_gs_hybrid_v3",
        ]:
            ## convert back to RUB (habitat) ##
            c2w_sim = c2w_slam.cpu().numpy().copy() # RDF
            c2w_sim[:3, 1] *= -1 
            c2w_sim[:3, 2] *= -1 # RUB
        else:
            raise NotImplementedError

        ## convert to relative pose (w.r.t first pose) ##
        c2w_slam_rel = torch.inverse(c2w_slam_init) @ c2w_slam # RDF
        
        ##################################################
        ### Simulation
        ##################################################
        timer.start("Simulation", "General")
        vis_semantic = getattr(main_cfg.visualizer, 'vis_semantic', False)
        sim_out = sim.simulate(c2w_sim, return_semantic=vis_semantic)
        timer.end("Simulation")
        color = sim_out['color']
        depth = sim_out['depth']
        # if main_cfg.visualizer.vis_rgbd:
        #     visualizer.visualize_rgbd(color, depth, main_cfg.visualizer.vis_rgbd_max_depth)

        ##################################################
        ### Mapping optimization
        ##################################################
        ### get timer state ###
        if main_cfg.slam.enable_active_planning:
            planner_state = f"{planner.planning_state}_{planner.exploration_stage}" if planner.planning_state == "exploration" else planner.planning_state
        else:
            planner_state = "exploration"
        slam_state = f"SLAM_{planner_state}"
        timer.start(slam_state, "General")

        ### slam options ###
        if  main_cfg.slam.enable_active_planning:
            force_map_update = planner.state == "planning" or planner.planning_state == "post_refinement"
            dont_add_kf = planner.state == "planning"
            only_use_global_keyframe = main_cfg.slam.use_global_keyframe and planner.planning_state == "post_refinement"
            slam.seperate_densification_res = not(planner.planning_state == "post_refinement")
        else:
            force_map_update = False
            dont_add_kf = False
            only_use_global_keyframe = False

        keyframes_extra_subdir = None
        if getattr(main_cfg.slam, 'save_keyframes', True) and getattr(
            main_cfg.slam, 'save_keyframes_exploration', True
        ):
            if main_cfg.slam.enable_active_planning and planner.planning_state == "exploration":
                keyframes_extra_subdir = getattr(
                    main_cfg.slam, 'keyframes_exploration_dir', 'keyframes_exploration'
                )

        slam.online_recon_step(
            i, color, depth, c2w_slam_rel,
            force_map_update, dont_add_kf, only_use_global_keyframe,
            keyframes_extra_subdir=keyframes_extra_subdir,
        )
        timer.end(slam_state)

        if main_cfg.planner.method == "predefined_traj":
            exploration_path_poses.append(np.asarray(c2w_sim, dtype=np.float64).copy())

        ##################################################
        ### Submit new keyframes to SAM+CLIP extractor
        # We detect whether a new keyframe was just added by comparing the
        # keyframe list length before and after the SLAM step.  This is
        # cheaper than modifying the SLAM code and avoids duplication.
        ##################################################
        if sam_clip_extractor is not None or _save_kf_poses or _debug_save_kf:
            _prev_kf_count = getattr(info_printer, '_prev_kf_count', 0)
            _curr_kf_count = len(slam.keyframe_list)
            if _curr_kf_count > _prev_kf_count:
                # One or more keyframes were added
                for kf in slam.keyframe_list[_prev_kf_count:]:
                    kf_id = kf['id']
                    # Submit frame to SAM+CLIP extractor
                    if sam_clip_extractor is not None:
                        kf_color_hwc = kf['color'].permute(1, 2, 0)  # (H,W,3)
                        sam_clip_extractor.submit(kf_id, kf_color_hwc)
                    # Store w2c pose for later export
                    if _save_kf_poses:
                        w2c_mat = kf['est_w2c'].detach().cpu().numpy().tolist()
                        _kf_poses[kf_id] = w2c_mat
                    # Debug: save keyframe RGB image to disk
                    if _debug_save_kf:
                        import cv2
                        kf_np = (kf['color'].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        kf_np = np.ascontiguousarray(kf_np)
                        cv2.imwrite(
                            os.path.join(_debug_kf_dir, f"frame_{kf_id:06d}.jpg"),
                            cv2.cvtColor(kf_np, cv2.COLOR_RGB2BGR),
                        )
            info_printer._prev_kf_count = _curr_kf_count

        #################################################
        ## save data for comprehensive visualization
        #################################################
        if main_cfg.visualizer.vis_rgbd:
            _render_out = slam.render(c2w_slam_rel)
            if len(_render_out) == 4:
                im, rastered_depth, _, seen = _render_out
            else:
                im, rastered_depth, _ = _render_out
                seen = None
            im = im.permute(1, 2, 0)
            rastered_depth = rastered_depth[0]
            visualizer.visualize_rgbd_w_render(color, depth, im, rastered_depth, main_cfg.visualizer.vis_rgbd_max_depth)
            if vis_semantic and 'seman' in sim_out and seen is not None and hasattr(slam, 'render_semantic'):
                pred_class_id, _ = slam.render_semantic(c2w_slam_rel, seen)
                visualizer.visualize_semantic(sim_out['seman'], pred_class_id, slam.n_cls)
            visualizer.main(slam, planner, color, depth, im, rastered_depth, c2w_slam)
        
        ##################################################
        ### Update open-vocabulary CLIP index
        # Runs every `update_every` steps (cheap no-op otherwise).
        # Only newly-added keyframes are encoded, so the overhead is
        # proportional to the rate of new keyframes, not total steps.
        ##################################################
        if open_vocab_index is not None:
            open_vocab_index.maybe_update(slam.keyframe_list)

        ##################################################
        ### Active Planning
        ##################################################
        if main_cfg.slam.enable_active_planning:
            if i == 0:
                ### update REFINE POOL ###
                planner.add_refine_pool_cand(slam.keyframe_list)

            ### get timer state ###
            planner_state = f"{planner.planning_state}_{planner.exploration_stage}" if planner.planning_state == "exploration" else planner.planning_state
            timer.start(planner_state, "General")

            c2w_slam_rel = planner.main(
                c2w_slam_rel, 
                slam)
            c2w_slam = c2w_slam_init @ c2w_slam_rel

            timer.end(planner_state)

            ##################################################
            ### Disable SAM+CLIP once exploration ends
            # Checked AFTER planner.main() so we react on the same step the
            # transition happens, not one step late.  We wait for the worker
            # thread to finish its current frame (wait=True) so the GPU is
            # completely free before the heavier 60-iter refinement SLAM starts.
            ##################################################
            if sam_clip_extractor is not None and planner.planning_state != "exploration":
                info_printer(
                    f"Exploration done — cleanly stopping SAM+CLIP extractor "
                    f"(state: {planner.planning_state}).",
                    i + 1, "SAM+CLIP"
                )
                info_printer(
                    "Flushing SAM+CLIP queue to avoid losing pending keyframes...",
                    i + 1, "SAM+CLIP"
                )
                sam_clip_extractor.flush()
                sam_clip_extractor.stop(wait=True, drain=True)
                sam_clip_stats_snapshot = sam_clip_extractor.stats()
                sam_clip_extractor = None

            if planner.planning_state in ["refinement", "post_refinement"]:
                # Refinement only optimises existing Gaussians — disable additions to
                # prevent the Gaussian count from exploding (can grow 5x per 100 steps
                # when add_new_gaussians=True is used with all global keyframes).
                slam.config['mapping']['add_new_gaussians'] = False
                # Free any fragmented CUDA allocations accumulated during exploration
                # so that the 60-iter rasterisation passes have enough contiguous memory.
                torch.cuda.empty_cache()
                if force_map_update:
                    slam.config['mapping']['num_iters'] = main_cfg.slam.refine_map_iter
                else:
                    slam.config['mapping']['num_iters'] = map_iter_og
                    
            # Record pose after planning (RUB / Habitat world).
            _pose_rub = c2w_slam.detach().cpu().numpy().copy()
            _pose_rub[:3, 1] *= -1
            _pose_rub[:3, 2] *= -1
            exploration_path_poses.append(_pose_rub)

            if planner.planning_state == "done":
                break

            ##################################################
            ### Validation during training
            ##################################################
            if hasattr(main_cfg.slam, 'eval_during_training') and main_cfg.slam.eval_during_training:
                eval_freq = getattr(main_cfg.slam, 'eval_during_training_freq', 100)
                if (i + 1) % eval_freq == 0 or i == 0:
                    info_printer(f"Running validation at step {i+1}...", i+1, "Validation")
                    timer.start("Validation", "General")
                    try:
                        # Get dataset length for full evaluation
                        dataset_len = len(slam.dataset_eval)
                        user_max = getattr(main_cfg.slam, 'eval_during_training_max_frames', None)
                        
                        # If user_max is None or >= dataset length, use all frames
                        # Otherwise, use min of processed frames and user_max
                        if user_max is None or int(user_max) >= dataset_len or int(user_max) == -1:
                            # Use all frames for evaluation
                            max_frames = None  # None means use all frames in eval_result
                            info_printer(f"Evaluating on all {dataset_len} frames", i+1, "Validation")
                        else:
                            # Limit to processed frames or user_max, whichever is smaller
                            max_frames = min(i + 1, int(user_max))
                            info_printer(f"Evaluating on {max_frames} frames (limited)", i+1, "Validation")
                        
                        eval_suffix = f"step_{i+1:04d}"
                        slam.eval_result(
                            eval_dir_suffix=eval_suffix,
                            ignore_first_frame=True,
                            save_frames=False,
                            max_frames=max_frames,
                        )
                        if open_vocab_index is not None and len(open_vocab_index) > 0:
                            open_vocab_index.save(clip_index_path)
                        info_printer(f"Validation completed at step {i+1}", i+1, "Validation")
                    except torch.cuda.OutOfMemoryError as e:
                        info_printer(f"Validation OOM at step {i+1}", i+1, "Validation")
                        print(
                            "Validation failed: CUDA out of memory. "
                            "Try reducing eval_during_training_max_frames in config (e.g. 50 or 100) "
                            "or closing other GPU processes."
                        )
                        print(f"OOM error: {e}")
                    except Exception as e:
                        info_printer(f"Validation failed at step {i+1}: {str(e)}", i+1, "Validation")
                        print(f"Warning: Validation error: {e}")
                    timer.end("Validation")

            ### store data for visualization ###
            # if planner.state == "planning" and "exploration" in planner.planning_state:
            #     ### save params and render result ###
            #     # slam.print_and_save_result(f"step_{i:04}", ignore_first_frame=True)

            #     ### save information ###
            #     igs = []
            #     for key, val in planner.explore_pool.items():
            #         igs.append(val['ig'].detach().cpu().numpy())
            #     igs = np.asarray(igs)
            #     eval_dir_suffix = f"step_{i:04}"
            #     eval_dir = slam.eval_dir + "_" + eval_dir_suffix 
            #     os.makedirs(eval_dir, exist_ok=True)
            #     np.save(os.path.join(eval_dir, "information.npy"), igs)



    ##################################################
    ### Flush SAM+CLIP extractor and save poses
    ##################################################
    if sam_clip_extractor is not None:
        info_printer("Waiting for SAM+CLIP extractor to finish...", 0, "SAM+CLIP")
        sam_clip_extractor.flush()
        sam_clip_extractor.stop(wait=True, drain=True)
        sam_clip_stats_snapshot = sam_clip_extractor.stats()
        info_printer("SAM+CLIP extraction complete.", 0, "SAM+CLIP")

    ##################################################
    ### Audit SAM+CLIP feature coverage
    ##################################################
    lang_feat_dir = os.path.join(main_cfg.dirs.result_dir, "language_features")
    if os.path.isdir(lang_feat_dir):
        pat = re.compile(r"^(\d+)_([sf])\.npy$")
        ids_s = set()
        ids_f = set()
        for fn in os.listdir(lang_feat_dir):
            m = pat.match(fn)
            if m is None:
                continue
            fid = int(m.group(1))
            if m.group(2) == "s":
                ids_s.add(fid)
            else:
                ids_f.add(fid)
        ids_both = ids_s & ids_f
        n_poses = len(_kf_poses) if _save_kf_poses else 0
        miss_pose = (set(_kf_poses.keys()) - ids_both) if _save_kf_poses else set()
        info_printer(
            "SAM+CLIP coverage audit: "
            f"poses={n_poses}, s_files={len(ids_s)}, f_files={len(ids_f)}, paired={len(ids_both)}, "
            f"missing_from_poses={len(miss_pose)}",
            0, "SAM+CLIP"
        )
        if sam_clip_stats_snapshot is not None:
            info_printer(
                "SAM+CLIP worker stats: "
                + ", ".join(f"{k}={v}" for k, v in sam_clip_stats_snapshot.items()),
                0, "SAM+CLIP"
            )
    elif _save_clip:
        info_printer(
            f"SAM+CLIP coverage audit skipped: directory not found: {lang_feat_dir}",
            0, "SAM+CLIP"
        )

    if _save_kf_poses and _kf_poses:
        poses_path = os.path.join(main_cfg.dirs.result_dir, "keyframe_poses.json")
        with open(poses_path, 'w') as _f:
            json.dump({str(k): v for k, v in _kf_poses.items()}, _f)
        info_printer(f"Keyframe poses saved → {poses_path} ({len(_kf_poses)} frames)", 0, "Poses")

    if exploration_path_poses:
        _bbox = getattr(main_cfg.slam, "bbox_bound", None)
        _bbox_xy = _bbox[:2] if _bbox is not None else None
        _run_tag = os.path.basename(os.path.normpath(main_cfg.dirs.result_dir))
        _path_png = save_exploration_path_topdown(
            exploration_path_poses,
            main_cfg.dirs.result_dir,
            bbox_xy=_bbox_xy,
            scene_name=main_cfg.general.scene,
            run_tag=_run_tag,
        )
        info_printer(
            f"Exploration path (top-down) saved → {_path_png} ({len(exploration_path_poses)} poses)",
            0,
            "Trajectory",
        )

    ##################################################
    ### Save Final Mesh and Checkpoint
    ##################################################
    slam.print_and_save_result("final", ignore_first_frame=True)

    if open_vocab_index is not None and len(open_vocab_index) > 0:
        open_vocab_index.save(clip_index_path)
        info_printer(
            f"CLIP index saved: {len(open_vocab_index)} keyframes → {clip_index_path}",
            0, "CLIP"
        )

    ##################################################
    ### Runtime Analysis
    ##################################################
    timer.time_analysis(method='mean')
    for _timer_key, _skip, _stride in (
        ("SLAM_exploration_0", 4, 5),
        ("SLAM_exploration_1", 3, 5),
        ("SLAM_exploration", 4, 5),
    ):
        _durations = timer.timers.get(_timer_key, {}).get("duration")
        if _durations:
            print(
                f"per-iter {_timer_key}: ",
                np.mean(_durations[_skip:][::_stride]),
            )
