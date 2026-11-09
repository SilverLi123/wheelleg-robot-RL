"""Checkpoint architecture and observation checks; never replace rewards implicitly."""
import hashlib
import importlib
import json
import importlib.metadata
import sys
from pathlib import Path
import torch
import yaml

OBS_KEYS = ('obs_input_scale_enabled', 'obs_input_scale_streams', 'obs_input_scale_cfg',
            'obs_input_clip_cfg', 'joint_pos_obs_encoding', 'leg_action_encoding',
            'ctrl_mode_obs_layout', 'use_frame_stack', 'num_obs_hist',
            'mute_wheel_pos_obs', 'use_wheel_vel_control', 'leg_action_scale',
            'wheel_vel_action_scale', 'max_wheel_vel', 'use_obs_delay', 'obs_delay_cfg',
            'use_act_delay', 'act_delay_cfg', 'use_absolute_height',
            'use_leg_length_as_height', 'links_length', 'alpha_offset',
            'ordered_leg_joint_names', 'gimbal_pitch_target_pos')


def verify_resume(runner, checkpoint_path):
    """Fail if a load silently omits optimizer state or the saved iteration."""
    saved = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    if 'optimizer_state_dict' not in saved or 'iter' not in saved:
        raise ValueError('Continuation requires a training checkpoint with optimizer and iteration')
    if runner.current_learning_iteration != saved['iter']:
        raise RuntimeError('Runner did not restore the checkpoint iteration')
    def compare(left, right, path='optimizer'):
        if torch.is_tensor(left):
            if not torch.is_tensor(right) or not torch.equal(left.cpu(), right.cpu()):
                raise RuntimeError('Resume tensor mismatch: ' + path)
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise RuntimeError('Resume key mismatch: ' + path)
            for k in left:
                compare(left[k], right[k], path + '.' + str(k))
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                raise RuntimeError('Resume length mismatch: ' + path)
            for i, (a, b) in enumerate(zip(left, right)):
                compare(a, b, path + '.' + str(i))
        elif left != right:
            raise RuntimeError('Resume value mismatch: ' + path)
    current_optimizer = runner.alg.optimizer.state_dict()
    compare(saved['optimizer_state_dict'], current_optimizer)
    compare(saved['model_state_dict'], runner.alg.policy.state_dict(), 'model')
    return {'checkpoint_iteration': int(saved['iter']), 'optimizer_exact_match': True,
            'optimizer_state_restored': True, 'optimizer_saved_values_exact': True,
            'optimizer_state_entries': len(current_optimizer['state']),
            'model_exact_match': True,
            'next_iteration': int(saved['iter']) + 1,
            'checkpoint_sha256': hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()}

def _plain(value):
    if isinstance(value, dict): return {k:_plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_plain(v) for v in value]
    if isinstance(value,str):
        if value.lower() in ('true','false'): return value.lower()=='true'
        if value=='null': return None
        try: return float(value)
        except ValueError: return value
    return value

def prepare_checkpoint(env_cfg, agent_cfg, checkpoint, architecture=False, observations=False):
    if not checkpoint: return
    path=Path(checkpoint).resolve()
    if hasattr(env_cfg, 'own_physics_hz'):
        manifest_path=path.parent/'params/launch.json'
        if not manifest_path.exists():
            raise ValueError('Own robot checkpoint requires its original launch manifest')
        manifest=json.loads(manifest_path.read_text())
        if '-Own-' not in manifest.get('task',''):
            raise ValueError('Reference checkpoint cannot initialize the Own robot by default')
        root=Path(__file__).resolve().parents[2]
        recorded=manifest.get('own_asset_sha256',{})
        for p in (root/'source/agent_world/agent_world/assets/usd_files/wheelbipe_own').glob('*'):
            if p.is_file() and recorded.get(str(p.relative_to(root))) != hashlib.sha256(p.read_bytes()).hexdigest():
                raise ValueError('Own robot asset/reset data differ from checkpoint launch: '+p.name)
    data=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(data,dict) or 'model_state_dict' not in data:
        raise ValueError('Use model_N.pt training checkpoint, not exported/policy.pt TorchScript.')
    state=data['model_state_dict']
    if architecture:
        if agent_cfg.policy.class_name != 'ActorCritic':
            raise ValueError('--checkpoint_architecture currently supports feedforward PPO only')
        for network in ('actor','critic'):
            weights=[state[k] for k in sorted(state) if k.startswith(network+'.') and k.endswith('.weight')]
            setattr(agent_cfg.policy,network+'_hidden_dims',[v.shape[0] for v in weights[:-1]])
        agent_path=path.parent/'params/agent.yaml'
        if agent_path.exists():
            saved_agent=yaml.safe_load(agent_path.read_text())
            agent_cfg.policy.activation=saved_agent['policy']['activation']
            if saved_agent.get('empirical_normalization',False)!=agent_cfg.empirical_normalization:
                raise ValueError('Checkpoint empirical normalization differs; select a matching runner configuration.')
    params=path.parent/'params/env.yaml'
    if not params.exists():
        print('[WARNING] No saved env.yaml; observation semantics cannot be verified.')
        return
    saved=_plain(yaml.load(params.read_text(),Loader=yaml.BaseLoader))
    differences={key:{'saved':saved[key],'current':_plain(getattr(env_cfg,key,None))}
                 for key in OBS_KEYS if key in saved and saved[key]!=_plain(getattr(env_cfg,key,None))}
    if observations:
        # Explicit opt-in to sensor preprocessing only. Physics and rewards stay selected-task values.
        allowed={'obs_input_scale_enabled','obs_input_scale_streams','obs_input_scale_cfg',
                 'obs_input_clip_cfg','ctrl_mode_obs_layout'}
        for key in allowed:
            if key in saved: setattr(env_cfg,key,saved[key])
        differences={key:value for key,value in differences.items() if key not in allowed}
    if differences:
        raise ValueError('Checkpoint observation/action semantics differ: '+json.dumps(differences,ensure_ascii=False)+
                         '. For saved scaling/layout, explicitly pass --checkpoint_observations; other differences require a matching task.')
    print('[INFO] Checkpoint observation/action configuration matches compared fields. USD content/training code provenance is not proven by YAML.')

def write_launch_manifest(directory, task, checkpoint=None, mode='play'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    data={'task':task,'mode':mode,'checkpoint':str(Path(checkpoint).resolve()) if checkpoint else None,
          'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
          'imports':{n:importlib.import_module(n).__file__ for n in ('agent_tasks','agent_world','agent_rl','rsl_rl','isaaclab','isaaclab_rl')}}
    root=Path(__file__).resolve().parents[2]
    for name in ('agent_tasks','agent_world','agent_rl'):
        if not Path(data['imports'][name]).resolve().is_relative_to(root):
            raise RuntimeError(f'{name} loaded outside writable project: '+data['imports'][name])
    data['project_root']=str(root)
    data['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in list((root/'source').rglob('*.py'))+list((root/'scripts').rglob('*.py'))+list((root/'scripts').rglob('*.sh'))}
    data['usd_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (root/'source/agent_world').rglob('*.usd')}
    data['own_asset_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root/'source/agent_world/agent_world/assets/usd_files/wheelbipe_own').glob('*') if p.is_file()}
    data['installed_versions']={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    data['python']=sys.version
    data['executable']=sys.executable
    data['original_argv']=sys.orig_argv if hasattr(sys,'orig_argv') else sys.argv
    data['torch_cuda']=torch.version.cuda
    (directory/'launch.json').write_text(json.dumps(data,indent=2))
    print('[INFO] Launch manifest: '+str(directory/'launch.json'))
