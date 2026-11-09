"""Explicit Own-robot playback overrides; training defaults remain unchanged."""
import copy


def add_playback_arguments(parser):
    parser.add_argument("--solver_position_iterations", type=int, default=None,
                        help="Own playback only: position solver iterations (1..255).")
    parser.add_argument("--solver_velocity_iterations", type=int, default=None,
                        help="Own playback only: velocity solver iterations (0..255).")
    parser.add_argument("--standstill", action="store_true",
                        help="Own playback only: sample zero linear and yaw velocity commands.")


def validate_playback_arguments(args):
    enabled = (args.solver_position_iterations is not None or
               args.solver_velocity_iterations is not None or args.standstill)
    if enabled and "-Own-" not in (args.task or ""):
        raise ValueError("Solver/standstill playback overrides require an Own robot task")
    if args.solver_position_iterations is not None and not 1 <= args.solver_position_iterations <= 255:
        raise ValueError("Position solver iterations must be in 1..255")
    if args.solver_velocity_iterations is not None and not 0 <= args.solver_velocity_iterations <= 255:
        raise ValueError("Velocity solver iterations must be in 0..255")
    if args.standstill and args.keyboard:
        raise ValueError("--standstill and --keyboard select conflicting command sources")


def apply_playback_overrides(cfg, args):
    validate_playback_arguments(args)
    if args.solver_position_iterations is not None or args.solver_velocity_iterations is not None:
        cfg.robot_cfg = copy.deepcopy(cfg.robot_cfg)
        props = cfg.robot_cfg.spawn.articulation_props
        if args.solver_position_iterations is not None:
            props.solver_position_iteration_count = args.solver_position_iterations
        if args.solver_velocity_iterations is not None:
            props.solver_velocity_iteration_count = args.solver_velocity_iterations
        print(f"[INFO] Playback solver: {props.solver_position_iteration_count}/"
              f"{props.solver_velocity_iteration_count}; training defaults unchanged.")
    if args.standstill:
        cfg.commands = copy.deepcopy(cfg.commands)
        cfg.commands.heading_command = False
        cfg.commands.rel_heading_envs = 0.0
        cfg.commands.rel_standing_envs = 1.0
        cfg.commands.special_modes = {}
        cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
        cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
        print("[INFO] Standstill playback: vx=vy=yaw_rate=0 (policy remains active).")
