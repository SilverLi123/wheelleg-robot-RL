"""Joint-frame FK and constraint-consistent numerical IK, in native SI coordinates.

No reference-triangle assumptions, saved-angle offsets, clamp(arccos), or PhysX.
USD: T_child = T_parent @ frame0 @ motion(q) @ inverse(frame1).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "source/agent_world/agent_world/assets/usd_files/wheelbipe_own"


class Mechanism:
    def __init__(self, path=ASSET / "mechanism.json"):
        self.model = json.loads(Path(path).read_text())
        self.joints = self.model["joints"]
        self.names = list(self.joints)
        self.saved = {n: j["saved_q"] for n, j in self.joints.items()}
        self.f0 = {n: np.array(j["frame0"]) for n, j in self.joints.items()}
        self.inv1 = {n: np.linalg.inv(j["frame1"]) for n, j in self.joints.items()}

    def fk(self, q, side=None):
        poses = {"base_link": np.eye(4)}
        remaining = {n: j for n, j in self.joints.items() if side is None or n.startswith(side + "_")}
        while remaining:
            progressed = False
            for n, j in list(remaining.items()):
                if j["parent"] not in poses:
                    continue
                axis = np.eye(3)["XYZ".index(j["axis"])]
                motion = np.eye(4)
                if j["unit"] == "m":
                    motion[:3, 3] = axis * q[n]
                else:
                    motion[:3, :3] = Rotation.from_rotvec(axis * q[n]).as_matrix()
                poses[j["child"]] = poses[j["parent"]] @ self.f0[n] @ motion @ self.inv1[n]
                del remaining[n]
                progressed = True
            if not progressed:
                raise ValueError("Joint graph is disconnected or cyclic")
        return poses

    def residual(self, q, side=None):
        p = self.fk(q, side)
        return np.concatenate([
            (p[j["parent"]] @ np.array(j["frame0"]))[:3, 3]
            - (p[j["child"]] @ np.array(j["frame1"]))[:3, 3]
            for j in self.model["loops"] if side is None or j["name"].startswith(side + "_")])

    def solve(self, side, front, rear, seed=None, endpoint=None):
        q = dict(self.saved if seed is None else seed)
        q[side + "_front1_joint"], q[side + "_rear1_joint"] = front, rear
        suffixes = ["front2", "front3", "front4", "rear2", "spring1", "spring2"]
        names = [side + "_" + x + "_joint" for x in suffixes]
        if endpoint is not None:
            names += [side + "_front1_joint", side + "_rear1_joint"]
        def unpack(x):
            return dict(q, **dict(zip(names, x)))
        def residual(x):
            candidate = unpack(x)
            r = self.residual(candidate, side)
            if endpoint is not None:
                r = np.r_[r, (self.fk(candidate, side)[side + "_wheel_link"][:3, 3] - endpoint)[[0, 2]]]
            return r
        low, high = [], []
        for n in names:
            j = self.joints[n]
            low.append(j["lower"] if j["lower"] is not None else -2 * math.pi)
            high.append(j["upper"] if j["upper"] is not None else 2 * math.pi)
        result = least_squares(residual, [q[n] for n in names], bounds=(low, high),
                               xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=120)
        q = unpack(result.x)
        err = np.max(np.linalg.norm(self.residual(q, side).reshape(-1, 3), axis=1))
        spring = q[side + "_spring2_joint"]
        if not result.success or err > 2e-5 or not 0 <= spring <= 0.042:
            raise ValueError(f"Unreachable {side}: closure={err:.6g}m, spring={spring:.6g}m")
        if endpoint is not None and np.linalg.norm(residual(result.x)[-2:]) > 2e-5:
            raise ValueError("Requested wheel endpoint is unreachable")
        return q, err


def build_bank(count=1000, seed=20261028):
    m = Mechanism()
    rng = np.random.default_rng(seed)
    banks, summaries, wheel_endpoints = {}, {}, {}
    for side in ("left", "right"):
        f, r = (m.saved[side + "_" + x + "1_joint"] for x in ("front", "rear"))
        nominal, nominal_err = m.solve(side, f, r)
        rows, endpoints, errors = [nominal], [m.fk(nominal, side)[side + "_wheel_link"][:3, 3]], [nominal_err]
        rejected = 0
        # Scan each motor independently; retain only closed poses in the physical spring band.
        for _ in range(count * 5):
            if len(rows) >= count:
                break
            try:
                q, err = m.solve(side, f + rng.uniform(-0.30, 0.30), r + rng.uniform(-0.30, 0.30))
            except ValueError:
                rejected += 1
                continue
            rows.append(q)
            endpoints.append(m.fk(q, side)[side + "_wheel_link"][:3, 3])
            errors.append(err)
        if len(rows) != count:
            raise RuntimeError(f"Not enough reachable {side} poses")
        banks[side] = [[q[n] for n in m.names] for q in rows]
        wheel_endpoints[side] = [x.tolist() for x in endpoints]
        ep = np.array(endpoints)
        length = np.linalg.norm(ep[:, [0, 2]], axis=1)
        angle = np.arctan2(-ep[:, 0], -ep[:, 2])
        summaries[side] = {"count": len(rows), "rejected": rejected, "max_closure_m": max(errors),
            "spring_range_m": [min(q[side + "_spring2_joint"] for q in rows), max(q[side + "_spring2_joint"] for q in rows)],
            "leg_length_range_m": [float(length.min()), float(length.max())],
            "leg_angle_range_rad": [float(angle.min()), float(angle.max())],
            "level_base_ground_height_range_m": [float(-ep[:, 2].max() + 0.0625), float(-ep[:, 2].min() + 0.0625)],
            "nominal_wheel_in_base_m": endpoints[0].tolist(), "nominal": nominal}
    data = {"names": m.names, "seed": seed, "native_SI": True, "banks": banks, "wheel_endpoints": wheel_endpoints,
            "summary": summaries, "status": "OFFLINE geometry only; GPU reset validation pending"}
    (ASSET / "reset_bank.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    (ASSET / "reachability.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({s: {k: v for k, v in x.items() if k != "nominal"} for s, x in summaries.items()}, indent=2))


if __name__ == "__main__":
    build_bank()
