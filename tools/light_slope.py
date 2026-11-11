"""在 pi 上运行一组 V14-6 冻结策略评估；不导入训练预设。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


CASES = ("flat", "up", "down")
START_X = -6.0  # 单块地形以世界原点居中；复用平地的 +X 初始朝向。
GRADE = .03
MIN_FORWARD_M = .05
MIN_FORWARD_X_COMPONENT = .95
SLOPE_CONFIG_CLASS = {
    "up": "HfPyramidSlopedTerrainCfg",
    # Isaac Lab 3.0 的 inverted 版本沿用同一个底层生成函数，
    # 但配置中的 inverted=True 会让沿 +X 的高度变化为负。
    "down": "HfInvertedPyramidSlopedTerrainCfg",
}
TEAM = Path("/data/wheelbipe-v14-6-training-20261108")
CHECKPOINT = TEAM / "logs/rsl_rl/wheelbipe_v14_6_dr_20261108/2026-11-07_18-03-53_flat_seed42_long/model_2000.pt"
CHECKPOINT_SHA = "6d321ea1b17ae240719f492bcde40f25e849ff1911399bd9e674023df0cec867"
RUNTIME = Path("/data/isaac620-migration-20261103/env620/bin/python")


def checked_grades(case, heights, spacing):
    """沿行进方向计算有符号坡度（高差 / 水平距离）；拒绝平台或反向。"""
    expected = {"flat": 0.0, "up": GRADE, "down": -GRADE}[case]
    grades = [(b - a) / spacing for a, b in zip(heights, heights[1:])]
    if not grades or any(not math.isclose(g, expected, abs_tol=.008) for g in grades):
        raise ValueError(f"{case}: collision mesh grades {grades}; expected {expected:+.3f}")
    return grades


def pitch_degrees(quat_wxyz):
    """把 Isaac Lab 的 (w, x, y, z) 四元数转换为世界坐标俯仰角。"""
    w, x, y, z = quat_wxyz
    s = 2 * (w * y - z * x)
    return math.degrees(math.asin(max(-1., min(1., s))))


def forward_x_component(quat_wxyz):
    """计算机身局部 +X 轴投影到世界 +X 轴后的分量。"""
    w, x, y, z = quat_wxyz
    del w, x  # 只有 y、z 影响局部 X 轴旋转方向。
    return 1. - 2. * (y * y + z * z)


def root_quat_wxyz(root_state_xyzw):
    """把 Isaac Lab root_state_w 的 [x,y,z,w] 转为计算函数的 [w,x,y,z]。"""
    x, y, z, w = root_state_xyzw[3:7]
    return [w, x, y, z]


def pitch_from_root_state(root_state_xyzw):
    """从 Isaac Lab 完整 root state 中读取并计算俯仰角。"""
    return pitch_degrees(root_quat_wxyz(root_state_xyzw))


def zeroed_slope(difficulty, cfg):
    """调用队友所用的坡面生成函数，只平移地形，让出发点地面 z=0。"""
    import numpy as np
    from isaaclab.terrains.height_field import hf_terrains

    meshes, origin = hf_terrains.pyramid_sloped_terrain(difficulty, cfg)
    start_xy = (cfg.size[0] / 2 + START_X, cfg.size[1] / 2)
    points = meshes[0].vertices
    point = points[np.linalg.norm(points[:, :2] - start_xy, axis=1).argmin()]
    if not np.allclose(point[:2], start_xy, atol=1e-5):
        raise ValueError("Slope start must lie on a terrain vertex")
    dz = -float(point[2])
    for mesh in meshes:
        mesh.apply_translation((0., 0., dz))
    origin[2] += dz
    return meshes, origin


def configure(cfg, case):
    """只替换地面几何，保留原 TerrainImporter 的摩擦和碰撞材料。"""
    if case == "flat":
        return
    from isaaclab.terrains import (
        HfInvertedPyramidSlopedTerrainCfg, HfPyramidSlopedTerrainCfg, TerrainGeneratorCfg,
    )

    terrain_cfg_classes = {
        "up": HfPyramidSlopedTerrainCfg,
        "down": HfInvertedPyramidSlopedTerrainCfg,
    }
    piece = terrain_cfg_classes[case]
    terrain = cfg.terrain.copy()
    terrain.terrain_type = "generator"
    terrain.max_init_terrain_level = 0
    terrain.terrain_generator = TerrainGeneratorCfg(
        seed=cfg.seed, size=(16., 12.), num_rows=1, num_cols=1, border_width=1.,
        horizontal_scale=.1, vertical_scale=.001, use_cache=False,
        sub_terrains={case: piece(proportion=1., function=zeroed_slope,
                                 slope_range=(GRADE, GRADE), platform_width=1.4, border_width=.3)},
    )
    cfg.terrain = terrain
    cfg.scene.terrain = terrain.copy()


def collision_heights(env, xy):
    """对 USD 碰撞网格向下发射射线；使用世界坐标和仿真采用的三角形。"""
    import numpy as np
    import torch
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import Usd, UsdGeom, UsdPhysics

    path, = env.terrain.terrain_prim_paths
    prims = [p for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath(path))
             if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
    if len(prims) != 1:
        raise ValueError(f"Expected one terrain collision mesh under {path}, found {len(prims)}")
    mesh = UsdGeom.Mesh(prims[0])
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    transform = np.asarray(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    vertices = (np.c_[points, np.ones(len(points))] @ transform)[:, :3].astype(np.float32)
    if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
        raise ValueError("Expected a triangulated terrain collision mesh")
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    query_mesh = convert_to_warp_mesh(vertices, faces, device=env.device)
    starts = torch.tensor([(x, y, 20.) for x, y in xy], device=env.device, dtype=torch.float32)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1
    hits = raycast_mesh(starts, directions, query_mesh, max_dist=40.)[0]
    if not torch.isfinite(hits).all():
        raise ValueError("A downward ray missed the terrain")
    return hits[:, 2].tolist()


def prepare_start(env, case):
    origin = env.terrain.env_origins[0]
    if case == "flat":
        return {"case": case, "start_xyz_m": origin.tolist(), "source": "original flat plane",
                "slope_status": "NOT_APPLICABLE", "initial_pose_status": "PENDING",
                "direction_status": "PENDING", "status": "PENDING"}

    # +0.50 m/s × 6 s = 3 m；每半米检查一次，避免把中心平台算作坡面。
    xy = [(START_X + .5 * i, 0.) for i in range(7)]
    heights = collision_heights(env, xy)
    grades = checked_grades(case, heights, .5)
    if abs(heights[0]) > .001:
        raise ValueError(f"Ground at slope start was not zeroed: {heights[0]:.4f} m")
    origin[0], origin[1] = xy[0]
    # 原复位仍负责选闭链姿态、清零速度；加几毫米余量，避免水平起步时轮胎穿地。
    origin[2] = heights[0] + .006
    return {"case": case, "start_xyz_m": origin.tolist(),
            "route_ground_heights_m": heights, "route_grades": grades,
            "slope_config_class": SLOPE_CONFIG_CLASS[case],
            "slope_status": "PENDING", "initial_pose_status": "PENDING",
            "direction_status": "PENDING", "status": "PENDING"}


def validate_reset_state(env, trial):
    """确认 reset 后仍从预期起点、零速度和世界 +X 方向起步。"""
    from agent_world.sim620_compat import torch_data

    state = torch_data(env._native_robot.data.root_state_w)[0].tolist()
    forward_x = forward_x_component(root_quat_wxyz(state))
    trial["initial_root_state_w"] = state
    trial["initial_forward_x"] = forward_x
    if forward_x < MIN_FORWARD_X_COMPONENT:
        trial["initial_pose_status"] = "FAIL"
        trial["status"] = "FAIL"
        raise ValueError(f"Initial pose does not face world +X: forward_x={forward_x:.4f}")
    if math.dist(state[:2], trial["start_xyz_m"][:2]) > .01:
        trial["initial_pose_status"] = "FAIL"
        trial["status"] = "FAIL"
        raise ValueError(f"Reset moved the robot away from the configured start: {state[:3]}")
    if math.dist(state[7:13], [0] * 6) > .01:
        trial["initial_pose_status"] = "FAIL"
        trial["status"] = "FAIL"
        raise ValueError(f"Reset did not clear root velocity: {state[7:13]}")
    trial["initial_pose_status"] = "PASS"


def add_metrics(env, path, trial):
    with path.open() as log:
        rows = [json.loads(line) for line in log]
    states = [row["root_state"] for row in rows]
    # 平地复位使用单位四元数；验证实际的起步姿态和静止条件没有被覆盖。
    initial = states[0]
    if (math.dist(initial[:2], trial["start_xyz_m"][:2]) > .01
            or math.dist(initial[3:6], [0, 0, 0]) > .01
            or abs(abs(initial[6]) - 1) > .01
            or math.dist(initial[7:13], [0] * 6) > .01):
        raise ValueError(f"Unexpected initial pose or velocity: {initial}")
    if any(not math.isclose(row["command"][0], .5, abs_tol=1e-6)
           or not math.isclose(row["command"][2], 0., abs_tol=1e-6) for row in rows):
        raise ValueError("The actual commands differ from the declared +0.50 m/s straight run")
    pitch = [pitch_from_root_state(state) for state in states]
    trial["world_pitch_range_deg"] = [min(pitch), max(pitch)]
    # 首次重置之后会瞬移回起点，轨迹指标只统计第一次尝试。
    stop = next((i + 1 for i, row in enumerate(rows) if row["reset"]), len(rows))
    xy = [state[:2] for state in states[:stop]]
    forward_m = xy[-1][0] - xy[0][0]
    trial["first_attempt_forward_m"] = forward_m
    trial["direction_status"] = "PASS" if forward_m > MIN_FORWARD_M else "FAIL"
    if trial["case"] != "flat":
        heights = collision_heights(env, xy)
        trial["first_attempt_ground_elevation_change_m"] = heights[-1] - heights[0]
        # 没走够 0.5 m 是可用的失败结果；不能因为地块名字就宣称上坡成功。
        signed_change = (1 if trial["case"] == "up" else -1) * (heights[-1] - heights[0])
        trial["measured_slope_travel"] = trial["first_attempt_forward_m"] >= .5 and signed_change >= .01
        trial["slope_status"] = "PASS" if trial["measured_slope_travel"] else "FAIL"
    else:
        trial["first_attempt_ground_elevation_change_m"] = 0.
        trial["measured_slope_travel"] = False
        trial["slope_status"] = "NOT_APPLICABLE"
    trial["status"] = "PASS" if (
        trial["initial_pose_status"] == "PASS"
        and trial["direction_status"] == "PASS"
        and trial["slope_status"] in ("PASS", "NOT_APPLICABLE")
    ) else "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("gpu", type=int, help="pi 上 nvidia-smi 显示的 GPU 编号")
    args = parser.parse_args()
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        parser.error("model_2000.pt SHA-256 does not match the verified baseline")
    project = TEAM / "project"
    tools = Path(__file__).resolve().parent
    gpu_info = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    uuid, utilization, memory = (item.strip() for item in gpu_info.split(","))
    if int(utilization) != 0 or int(memory) > 20:
        parser.error(f"GPU {args.gpu} is busy: {gpu_info}")
    out = Path(tempfile.mkdtemp(prefix=f"v14_6_light_slope_{args.case}_", dir="/data/wheeled-legged/logs"))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, HEADLESS="1", OMNI_KIT_ACCEPT_EULA="yes",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               WHEELBIPE_STAGE="3", WHEELBIPE_INTEGRATION_SUBSTEPS="2",
               PYTHONPATH=os.pathsep.join(str(project / p) for p in
                                          ("source/agent_tasks", "source/agent_world", "source/agent_rl", "")))
    for key, name in (("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "data"),
                      ("XDG_CONFIG_HOME", "config"), ("TMPDIR", "tmp"),
                      ("WHEELBIPE_DIAGNOSTICS_DIR", "diagnostics")):
        (out / name).mkdir()
        env[key] = str(out / name)
    (out / "kit").mkdir()
    print(f"输出目录：{out}", flush=True)
    return subprocess.call([
        str(RUNTIME), "-B", str(tools / "record.py"), "--light-slope-case", args.case,
        "--profile", "flat_seed42", "--checkpoint", str(CHECKPOINT), "--output", str(out),
        "--fixed-command", "0.5", "0", "--steps", "300", "--seed", "20261108", "--height", "0.255",
        "--record", "--device", "cuda:0", "--viz", "kit", "--livestream", "0",
        f"--kit_args=--portable-root={out / 'kit'}",
    ], cwd=out, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
