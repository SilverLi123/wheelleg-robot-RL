"""在已启动的 Isaac Sim 的 Script Editor 中打开 V14--6 检查场景。

输入：默认的交付 USD 路径；输出：已打开且停止的场景，以及模型元信息。
这一步不新建 Isaac Sim，不控制关节、不步进物理，也不保存或覆盖 USD。
"""

import asyncio
import sys
from pathlib import Path


# 共享环境不做 pip install -e；只让当前 Python 进程找到本项目的 source。
SOURCE_DIR = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import wheelleg_lab

# Script Editor 会保留已导入模块。若加载的是其他同名包，明确报错，避免串用源码。
if Path(wheelleg_lab.__file__).resolve().parent != SOURCE_DIR / "wheelleg_lab":
    raise RuntimeError("发现其他路径的 wheelleg_lab 包，请在本人的新 Isaac Sim 会话中运行。")

from wheelleg_lab.assets import INSPECTION_SCENE_USD


async def open_robot(usd_path: Path = INSPECTION_SCENE_USD) -> None:
    """打开完整检查场景；可传入另一个明确指定的 USD 文件进行检查。"""
    usd_path = Path(usd_path).expanduser().resolve()
    # 在切换场景前检查文件，错误路径不会触发场景打开操作。
    if not usd_path.is_file():
        raise FileNotFoundError(f"找不到模型，请检查本地文件或手动同步结果：{usd_path}")

    # 这些模块由已经启动的 Isaac Sim 提供，普通 Mac Python 不能替代它们。
    # 放在函数内导入，让模型路径和函数定义本身可以在本地只读检查。
    import omni.timeline
    import omni.usd
    from pxr import UsdGeom

    timeline = omni.timeline.get_timeline_interface()
    context = omni.usd.get_context()
    timeline.stop()
    print(f"[wheelleg_lab] 正在打开：{usd_path}")
    try:
        # await 把等待交给 Isaac Sim 的事件循环，加载期间仍可处理界面事件。
        # 它会替换当前打开的场景，因此应在本人已处理未保存改动的会话中运行。
        success, error = await context.open_stage_async(str(usd_path))
    finally:
        # 加载成功或抛出异常都执行，确保本脚本不留下正在播放的时间线。
        timeline.stop()
    if not success:
        raise RuntimeError(f"USD 场景打开失败：{error}")

    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("加载结束，但没有取得 USD 场景。")
    default_prim = stage.GetDefaultPrim()
    print(f"[wheelleg_lab] 场景文件：{stage.GetRootLayer().identifier}")
    print(f"[wheelleg_lab] 默认根节点：{default_prim.GetPath() if default_prim else '未设置'}")
    print(f"[wheelleg_lab] 长度单位：1 场景单位 = {UsdGeom.GetStageMetersPerUnit(stage)} 米")
    print(f"[wheelleg_lab] 向上坐标轴：{UsdGeom.GetStageUpAxis(stage)}")
    print("[wheelleg_lab] 时间线已停止；模型能打开还不代表已验证站立或控制。")


if __name__ == "__main__":
    # 在 Script Editor 通过 runpy.run_path(..., run_name="__main__") 执行此入口。
    # 普通 Python 误运行时只给提示，不尝试下载依赖或启动另一个模拟器。
    try:
        import omni.kit.app
    except ImportError:
        raise RuntimeError("请在已启动的 Isaac Sim Script Editor 中运行本脚本。") from None
    if not omni.kit.app.get_app().is_running():
        raise RuntimeError("请先启动 Isaac Sim，再运行本脚本。")

    # Isaac Sim 已有事件循环，使用 ensure_future 排队执行，而不是 asyncio.run 新建循环。
    # 保留任务引用，便于在 Script Editor 中查看 OPEN_ROBOT_TASK 的完成状态或异常。
    OPEN_ROBOT_TASK = asyncio.ensure_future(open_robot())
