import gymnasium as gym

for suffix, cfg in (("v0", "WheelbipeOwnFlatEnvCfg"), ("Play-v0", "WheelbipeOwnFlatPlayEnvCfg")):
    gym.register(id=f"Robotics-Wheelbipe-Own-Flat-{suffix}",
        entry_point=f"{__name__}.env:WheelbipeOwnEnv", disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{cfg}",
                "rsl_rl_cfg_entry_point": f"{__name__}.runner_cfg:WheelbipeOwnPPORunnerCfg"})

# Explicit experimental task; original Own tasks remain available unchanged.
gym.register(id='Robotics-Wheelbipe-Own-Explore-v0',
    entry_point=f'{__name__}.experiment:WheelbipeOwnExperimentEnv', disable_env_checker=True,
    kwargs={'env_cfg_entry_point': f'{__name__}.experiment:WheelbipeOwnExperimentCfg',
            'rsl_rl_cfg_entry_point': f'{__name__}.experiment:WheelbipeOwnExperimentRunnerCfg'})
