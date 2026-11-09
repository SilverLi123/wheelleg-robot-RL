"""Continuous flat-to-rough tiles using the repository's Isaac Lab terrain families."""
import copy
import numpy as np

from isaaclab.terrains import (
    TerrainGeneratorCfg, MeshPlaneTerrainCfg, HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg, HfWaveTerrainCfg, HfRandomUniformTerrainCfg,
)
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh


@height_field_to_mesh
def progressive_rough(difficulty, cfg):
    """The upstream random_uniform_terrain ignores difficulty; scale it explicitly."""
    c = copy.deepcopy(cfg)
    amplitude = .003 + .022 * difficulty
    c.noise_range = (-amplitude, amplitude)
    heights, _ = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, c)
    return _flat_transition(heights, cfg)


@height_field_to_mesh
def progressive_wave(difficulty, cfg):
    heights, _ = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    return _flat_transition(heights, cfg)


def _flat_transition(heights, cfg):
    # A flat central area and a 1 m smooth transition: every outward traversal
    # encounters flat ground followed by undulations, without a height jump.
    x = (np.arange(heights.shape[0]) - heights.shape[0] // 2) * cfg.horizontal_scale
    y = (np.arange(heights.shape[1]) - heights.shape[1] // 2) * cfg.horizontal_scale
    radius = np.maximum(np.abs(x[:, None]), np.abs(y[None, :]))
    blend = np.clip((radius - .7) / 1., 0., 1.)
    blend = blend * blend * (3. - 2. * blend)
    heights = np.rint(heights * blend).astype(np.int16)
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.])
    return heights, origin


FAMILIES = ('flat', 'wave', 'rough', 'slope_up', 'slope_down')
# 20 columns make these proportions exact in the generator.
COLUMN_FAMILY = (0,) * 6 + (1,) * 4 + (2,) * 4 + (3,) * 3 + (4,) * 3


def terrain_generator(seed=20261108):
    return TerrainGeneratorCfg(
        seed=seed, size=(16., 12.), border_width=10., num_rows=6, num_cols=20,
        curriculum=True, horizontal_scale=.1, vertical_scale=.001,
        use_cache=False,
        sub_terrains={
            'flat': MeshPlaneTerrainCfg(proportion=.30),
            'wave': HfWaveTerrainCfg(proportion=.20, function=progressive_wave,
                amplitude_range=(.003, .04), num_waves=12, border_width=.3),
            'rough': HfRandomUniformTerrainCfg(proportion=.20, function=progressive_rough,
                noise_range=(-.025, .025), noise_step=.001,
                downsampled_scale=.3, border_width=.3),
            'slope_up': HfPyramidSlopedTerrainCfg(proportion=.15,
                slope_range=(.03, .25), platform_width=1.4, border_width=.3),
            'slope_down': HfInvertedPyramidSlopedTerrainCfg(proportion=.15,
                slope_range=(.03, .25), platform_width=1.4, border_width=.3),
        })
