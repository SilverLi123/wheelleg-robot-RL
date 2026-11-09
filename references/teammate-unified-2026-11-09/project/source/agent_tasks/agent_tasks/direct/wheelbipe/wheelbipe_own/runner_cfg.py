from isaaclab.utils import configclass
from agent_tasks.direct.wheelbipe.agents.rsl_rl_ppo_cfg import WheelbipeV14FlatPPORunnerCfg


@configclass
class WheelbipeOwnPPORunnerCfg(WheelbipeV14FlatPPORunnerCfg):
    experiment_name = "wheelbipe_own_flat"
    max_iterations = 200
    save_interval = 25
    # Lab2.3/RSL3 uses TensorDict observation groups.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
