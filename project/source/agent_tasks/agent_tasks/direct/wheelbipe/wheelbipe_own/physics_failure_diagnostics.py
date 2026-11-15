"""Record closed-loop solver spikes before the existing hard guard aborts."""
import json
import time
import torch
from agent_world.sim620_compat import torch_data as T
from isaaclab.utils.math import quat_apply


def record_loop_spike(env, error):
    eid = int(error.amax(1).argmax())
    robot = env._native_robot
    view = robot.root_physx_view
    row = dict(time_unix=time.time(), policy_step=env._audit_steps, environment=eid,
               episode_step=int(env.episode_length_buf[eid]),
               closure_m=error[eid].detach().cpu().tolist(),
               body_names=robot.body_names, joint_names=robot.joint_names)
    for name in ['joint_pos', 'joint_vel', 'joint_pos_target', 'joint_vel_target',
                 'applied_torque', 'root_state_w', 'body_pos_w', 'body_quat_w',
                 'body_ang_vel_w', 'body_lin_vel_w', 'body_mass']:
        try:
            row[name] = T(getattr(robot.data, name))[eid].detach().cpu().tolist()
        except Exception as exc:
            row[name + '_unavailable'] = str(exc)
    row['contacts'] = T(env.contact_sensor.data.net_forces_w)[eid].detach().cpu().tolist()
    row['contact_names'] = env.contact_sensor.body_names
    row['loop_body_indices'] = env._audit_ids.cpu().tolist()
    if hasattr(env, '_dr_clock'):
        row.update(push_clock=float(env._dr_clock[eid]), push_deadline=float(env._dr_push_interval[eid]),
                   push_force=env._dr_push[eid].cpu().tolist(), rolling_coefficient=float(env._dr_rr[eid, 0]))
    # Cross-check the cached public poses against raw native PhysX link poses.
    if not env.cfg.experiment_profile_settings.get('per_env_closure_reset', False):
        try:
            names = [str(path).rsplit('/', 1)[-1] for path in view.link_paths[eid]]
            order = [names.index(name) for name in robot.body_names]
            pose = T(view.get_link_transforms())[eid, order].to(env.device)
            bp = pose[:, :3][env._audit_ids]
            cached_q = T(robot.data.body_quat_w)[eid]
            direct_dot = (pose[:, 3:] * cached_q).sum(-1).abs().mean()
            swizzle_dot = (pose[:, [6, 3, 4, 5]] * cached_q).sum(-1).abs().mean()
            bq = (pose[:, 3:] if direct_dot > swizzle_dot else pose[:, [6, 3, 4, 5]])[env._audit_ids]
            row['raw_quaternion_order'] = 'matches_native_public' if direct_dot > swizzle_dot else 'converted_to_native_public'
            points = bp + quat_apply(bq, env._audit_anchors)
            row['raw_physx_closure_m'] = torch.linalg.vector_norm(points[:, 0] - points[:, 1], dim=-1).cpu().tolist()
            row['raw_physx_link_poses'] = pose.cpu().tolist()
        except Exception as exc:
            row['raw_physx_unavailable'] = str(exc)
    dest = env._audit_path.parent / 'loop_spike_events.jsonl'
    with dest.open('a') as stream:
        stream.write(json.dumps(row) + '\n')
