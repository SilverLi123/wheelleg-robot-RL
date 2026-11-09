"""Aperiodic terrain replacement with unchanged family mixture and tile geometry."""
import numpy as np

from isaaclab.terrains import (
    TerrainGeneratorCfg, MeshPlaneTerrainCfg, HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg, HfWaveTerrainCfg, HfRandomUniformTerrainCfg,
)
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from .unified_irregular_fields import irregular_wave, irregular_rough

FAMILIES = ('flat', 'wave', 'rough', 'slope_up', 'slope_down')
COLUMN_FAMILY = (0,) * 6 + (1,) * 4 + (2,) * 4 + (3,) * 3 + (4,) * 3


def _shape(cfg):
    return tuple(int(v / cfg.horizontal_scale) for v in cfg.size)


def _flat_transition(heights, cfg):
    x = (np.arange(heights.shape[0]) - heights.shape[0] // 2) * cfg.horizontal_scale
    y = (np.arange(heights.shape[1]) - heights.shape[1] // 2) * cfg.horizontal_scale
    radius = np.maximum(np.abs(x[:, None]), np.abs(y[None, :]))
    blend = np.clip((radius - .7), 0., 1.)
    blend = blend * blend * (3. - 2. * blend)
    return np.rint(heights * blend).astype(np.int16), np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.])


@height_field_to_mesh
def progressive_wave(difficulty, cfg):
    return _flat_transition(irregular_wave(_shape(cfg), cfg.horizontal_scale,
        cfg.vertical_scale, difficulty, cfg.seed), cfg)


@height_field_to_mesh
def progressive_rough(difficulty, cfg):
    return _flat_transition(irregular_rough(_shape(cfg), cfg.horizontal_scale,
        cfg.vertical_scale, difficulty, cfg.seed), cfg)


@height_field_to_mesh
def textured_slope(difficulty, cfg):
    heights, origin = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    noise = irregular_rough(heights.shape, cfg.horizontal_scale, cfg.vertical_scale,
        difficulty, cfg.seed + 191)
    noise, _ = _flat_transition(noise, cfg)
    # Preserve the nominal slope and add only 10% of the rough-family relief.
    return heights + np.rint(noise * .1).astype(np.int16), origin


def terrain_generator(seed=20261108):
    return TerrainGeneratorCfg(
        seed=seed, size=(16., 12.), border_width=10., num_rows=6, num_cols=20,
        curriculum=True, horizontal_scale=.1, vertical_scale=.001, use_cache=False,
        sub_terrains={
            'flat': MeshPlaneTerrainCfg(proportion=.30),
            'wave': HfWaveTerrainCfg(proportion=.20, function=progressive_wave,
                amplitude_range=(.003, .04), num_waves=12, border_width=.3),
            'rough': HfRandomUniformTerrainCfg(proportion=.20, function=progressive_rough,
                noise_range=(-.025, .025), noise_step=.001, downsampled_scale=.3, border_width=.3),
            'slope_up': HfPyramidSlopedTerrainCfg(proportion=.15, function=textured_slope,
                slope_range=(.03, .25), platform_width=1.4, border_width=.3),
            'slope_down': HfInvertedPyramidSlopedTerrainCfg(proportion=.15, function=textured_slope,
                slope_range=(.03, .25), platform_width=1.4, border_width=.3),
        })
