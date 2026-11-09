"""Bounded evaluation: preserve default rewards and capture pre-reset metrics."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Robotics-Wheelbipe-V14-Flat-v0')
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--checkpoint', nargs='*', default=[])
parser.add_argument('--output', required=True)
parser.add_argument('--saved_obs', action='store_true')
parser.add_argument('--quiet_start', action='store_true')
parser.add_argument('--standing', action='store_true')
parser.add_argument('--solver_velocity_iterations', type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import csv
import hashlib
import importlib
import json
import pathlib
import types
import numpy as np
import torch
import yaml
import gymnasium as gym
import agent_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.seed = args.seed
if args.solver_velocity_iterations is not None:
    cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count=args.solver_velocity_iterations
saved = None
if args.saved_obs:
    saved = yaml.load((pathlib.Path(args.checkpoint[0]).parent/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
    # Only explicit sensor/observation semantics, no model/physics/reward substitution.
    for key in ['obs_input_scale_cfg', 'obs_input_clip_cfg']:
        def numeric(value):
            if isinstance(value, dict): return {k:numeric(v) for k,v in value.items()}
            if isinstance(value, list): return [numeric(v) for v in value]
            try: return float(value)
            except (ValueError, TypeError): return value
        setattr(cfg,key,numeric(saved[key]))
    for key in ['ctrl_mode_obs_layout','jump_takeoff_extra_obs_layout']:
        if key in saved: setattr(cfg,key,tuple(saved[key]))
if args.quiet_start:
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque_xyz = None
    cfg.use_predefined_leg_random_start = False
    cfg.leg_length_range = [0.22,0.22]
    cfg.leg_angle_range = [0.,0.]
    cfg.use_joint_vel_random_start = False
    cfg.predefined_reset_ground = {'enabled':False}
    cfg.events.reset_base.params['pose_range'] = {'roll':(0.,0.),'pitch':(0.,0.),'yaw':(0.,0.)}
cfg.play_wheel_material_debug = False
dump_yaml(str(out/'env.yaml'), cfg)
env = gym.make(args.task,cfg=cfg)
u = env.unwrapped
original_reset = u._reset_idx
delay_reset_checks = [0]
def checked_reset(self, env_ids):
    original_reset(env_ids)
    for group in ('obs_delay','act_delay'):
        for buffer in getattr(self,group,{}).values():
            assert torch.all(buffer._circular_buffer._num_pushes[env_ids]==0), 'Delay history crossed episode reset'
    delay_reset_checks[0]+=1
u._reset_idx=types.MethodType(checked_reset,u)
imports = {n:importlib.import_module(n).__file__ for n in ['agent_tasks','agent_world','agent_rl','rsl_rl','isaaclab','isaaclab_rl']}
robot = u.robot
snapshot = {'imports':imports,'task':args.task,'dt':u.physics_dt,'step_dt':u.step_dt,
            'evaluation_arguments':vars(args),
            'command_override':{'velocity':[0.,0.,0.],'height':0.22} if args.standing else None,
            'joint_names':robot.joint_names,'body_names':robot.body_names,
            'action_order':[robot.joint_names[i] for i in u._actuate_idx],
            'masses':robot.root_physx_view.get_masses().cpu().tolist(),
            'inertias':robot.root_physx_view.get_inertias().cpu().tolist(),
            'reset_contacts':[u.contact_sensor.body_names[i] for i in u._reset_contact_link_idx],
            'usd':cfg.robot_cfg.spawn.usd_path}
snapshot['effective_max_wheel_vel']=u.max_wheel_vel
snapshot['joint_limits']=robot.data.joint_pos_limits[0].cpu().tolist()
snapshot['actuators']={name:{'joint_names':act.joint_names,'stiffness':act.stiffness.cpu().tolist(),
                            'damping':act.damping.cpu().tolist(),'effort_limit':act.effort_limit.cpu().tolist(),
                            'velocity_limit':act.velocity_limit.cpu().tolist()} for name,act in robot.actuators.items()}
snapshot['usd_sha256']=hashlib.sha256(pathlib.Path(snapshot['usd']).read_bytes()).hexdigest()
from pxr import Usd, UsdPhysics
stage=u.sim.stage
snapshot['usd_layers']=[layer.identifier for layer in stage.GetUsedLayers()]
snapshot['colliders']=[{'path':str(p.GetPath()),'type':p.GetTypeName(),
                        'attributes':{a.GetName():str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:','physx'))}}
                       for p in stage.Traverse(Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI) and '/env_0/' in str(p.GetPath())]
snapshot['usd_joints']=[{'path':str(p.GetPath()),'type':p.GetTypeName(),
                        'limits':{a.GetName():str(a.Get()) for a in p.GetAttributes() if a.GetName().endswith(('lowerLimit','upperLimit'))}}
                       for p in stage.Traverse() if p.IsA(UsdPhysics.Joint) and '/env_0/' in str(p.GetPath())]
(out/'runtime.json').write_text(json.dumps(snapshot,indent=2))
capture = {}
original_dones = u._get_dones
def audited_dones(self):
    terminated, timeout = original_dones()
    roll,pitch,_ = euler_xyz_from_quat(robot.data.root_quat_w)
    contact_forces = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w[:,self._reset_contact_link_idx],dim=-1)
    contact = contact_forces.amax(-1)>1.
    height = robot.data.root_pos_w[:,2]-self.terrain.env_origins[:,2]
    capture.update(terminated=terminated.clone(),timeout=timeout.clone(),
                   length=self.episode_length_buf.clone(),roll=wrap_to_pi(roll).clone(),pitch=wrap_to_pi(pitch).clone(),
                   contact=contact.clone(),contact_forces=contact_forces.clone(),numerical=self._numerical_safety_reset_buf.clone(),
                   height_error=(height-self.height_cmd).clone(),
                   velocity_error=(robot.data.root_lin_vel_b[:,:2]-self.command[:,:2]).clone(),
                   yaw_error=(robot.data.root_ang_vel_b[:,2]-self.command[:,2]).clone())
    return terminated,timeout
u._get_dones = types.MethodType(audited_dones,u)
results=[]
cases = args.checkpoint or ['random']
for checkpoint in cases:
    torch.manual_seed(args.seed)
    obs,_ = env.reset(seed=args.seed)
    if args.standing:
        # Command generator is overridden on every observation, including autoresets.
        orig_obs = u._get_observations
        def standing_obs(self):
            self.command_generator.command[:,:3]=0.
            self.height_cmd.fill_(0.22)
            return orig_obs()
        u._get_observations=types.MethodType(standing_obs,u)
        obs=u._get_observations()
    if checkpoint=='random':
        dims=[obs['policy'].shape[-1],256,128,64,6]
        actor=torch.nn.Sequential(*[layer for i in range(len(dims)-1) for layer in ([torch.nn.Linear(dims[i],dims[i+1]),torch.nn.ELU()] if i<len(dims)-2 else [torch.nn.Linear(dims[i],dims[i+1])])]).to(u.device)
    else:
        d=torch.load(checkpoint,map_location=u.device,weights_only=False)
        sd={k[6:]:v for k,v in d['model_state_dict'].items() if k.startswith('actor.')}
        weights=[v for k,v in sd.items() if k.endswith('.weight')]
        dims=[weights[0].shape[1]]+[v.shape[0] for v in weights]
        actor=torch.nn.Sequential(*[layer for i in range(len(dims)-1) for layer in ([torch.nn.Linear(dims[i],dims[i+1]),torch.nn.ELU()] if i<len(dims)-2 else [torch.nn.Linear(dims[i],dims[i+1])])]).to(u.device)
        actor.load_state_dict(sd,strict=True)
        if 'obs_norm_state_dict' in d: raise RuntimeError('Normalized checkpoint requires its normalizer')
    actor.eval()
    assert dims[0]==obs['policy'].shape[-1],(dims,obs['policy'].shape)
    name=pathlib.Path(checkpoint).stem+'_'+(pathlib.Path(checkpoint).parent.name if checkpoint!='random' else '')
    rows=[];lengths=[];terms={};counts={'terminated':0,'timeout':0,'contact_at_termination':0,'roll_at_termination':0,'pitch_at_termination':0,'numerical':0}
    all_errors=[];upright=0;finite=True;contact_counts=torch.zeros(len(u._reset_contact_link_idx),device=u.device)
    tilts=[];heights=[];contact_steps=0
    first_seen=torch.zeros(u.num_envs,dtype=torch.bool,device=u.device)
    first_lengths=torch.full((u.num_envs,),float('nan'),device=u.device)
    first_counts={'terminated':0,'timeout':0,'timeout_without_termination':0}
    with torch.no_grad():
        for step in range(args.steps):
            action=actor(obs['policy'])
            finite=finite and bool(torch.isfinite(obs['policy']).all() and torch.isfinite(action).all())
            obs,reward,terminated,timeout,info=env.step(action)
            done=capture['terminated']|capture['timeout']
            finite=finite and bool(torch.isfinite(reward).all() and torch.isfinite(robot.data.root_state_w).all())
            first_done=done&~first_seen
            first_lengths[first_done]=capture['length'][first_done]*u.step_dt
            for key in ('terminated','timeout'):
                first_counts[key]+=int((capture[key]&first_done).sum())
            first_counts['timeout_without_termination']+=int((capture['timeout']&~capture['terminated']&first_done).sum())
            first_seen|=done
            tilts.append(torch.stack([capture['roll'].abs(),capture['pitch'].abs()],-1).cpu().numpy())
            heights.append(capture['height_error'].cpu().numpy())
            contact_steps+=int(capture['contact'].sum())
            lengths.extend((capture['length'][done]*u.step_dt).cpu().tolist())
            counts['terminated']+=int(capture['terminated'].sum())
            counts['timeout']+=int(capture['timeout'].sum())
            contact_counts+=((capture['contact_forces']>1.)&capture['terminated'].unsqueeze(-1)).sum(0)
            for key,mask in [('contact_at_termination',capture['contact']),('roll_at_termination',capture['roll'].abs()>np.deg2rad(40)),('pitch_at_termination',capture['pitch'].abs()>np.deg2rad(40)),('numerical',capture['numerical'])]:
                counts[key]+=int((mask&capture['terminated']).sum())
            upright+=int(((capture['roll'].abs()<np.deg2rad(10))&(capture['pitch'].abs()<np.deg2rad(10))).sum())
            errors=[capture['velocity_error'].square().sum(-1).mean().item(),capture['height_error'].square().mean().item(),capture['yaw_error'].square().mean().item()]
            all_errors.append(errors)
            for k,v in u._last_reward_terms.items():
                if k in cfg.rewards: terms[k]=terms.get(k,0.)+v.mean().item()
            rows.append({'step':step,'reward_mean':reward.mean().item(),'height_rmse':errors[1]**0.5,'velocity_rmse':errors[0]**0.5,'roll_abs_mean_deg':capture['roll'].abs().mean().item()*180/np.pi,'pitch_abs_mean_deg':capture['pitch'].abs().mean().item()*180/np.pi,'terminated':int(capture['terminated'].sum()),'timeout':int(capture['timeout'].sum())})
            if (step+1)%250==0: print('AUDIT_PROGRESS',name,step+1,counts,flush=True)
    with (out/(name+'.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    result={'checkpoint':str(pathlib.Path(checkpoint).resolve()) if checkpoint!='random' else checkpoint,'actor_dims':dims,'steps':args.steps,'num_envs':u.num_envs,'counts':counts,'completed_episode_mean_s':float(np.mean(lengths)) if lengths else None,'completed_episode_median_s':float(np.median(lengths)) if lengths else None,'censored_episode_ages_s':(u.episode_length_buf*u.step_dt).cpu().tolist(),'upright_fraction_10deg':upright/(args.steps*u.num_envs),'rmse_velocity_height_yaw':np.sqrt(np.mean(all_errors,axis=0)).tolist(),'finite':finite,'reward_contribution_per_s':{k:v/(args.steps*u.step_dt) for k,v in terms.items()}}
    results.append(result)
    result['seed']=args.seed
    result['first_episode_counts']={**first_counts,'censored':int((~first_seen).sum())}
    result['first_episode_lengths_s']=[v if np.isfinite(v) else None for v in first_lengths.cpu().tolist()]
    result['first_episode_20s_completion_fraction']=first_counts['timeout_without_termination']/u.num_envs
    result['completed_episode_length_percentiles_s']=np.percentile(lengths,[10,50,90,95]).tolist() if lengths else None
    result['abs_roll_pitch_p95_deg']=np.rad2deg(np.percentile(np.concatenate(tilts),95,axis=0)).tolist()
    result['mean_height_error_m']=float(np.concatenate(heights).mean())
    result['undesired_contact_step_fraction']=contact_steps/(args.steps*u.num_envs)
    result['checkpoint_sha256']=hashlib.sha256(pathlib.Path(checkpoint).read_bytes()).hexdigest() if checkpoint!='random' else None
    result['delay_reset_checks']=delay_reset_checks[0]
    result['contact_links_at_termination']={u.contact_sensor.body_names[i]:int(v) for i,v in zip(u._reset_contact_link_idx,contact_counts.cpu().tolist()) if v}
    print('AUDIT_RESULT',json.dumps(result),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    if args.standing: u._get_observations=orig_obs
env.close()
app.close()
