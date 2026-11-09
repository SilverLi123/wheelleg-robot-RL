"""集中记录模型文件的位置，不创建机器人，也不设置物理或电机参数。"""

from pathlib import Path


# 当前文件位于 <项目>/source/wheelleg_lab/assets.py。
# parents[0] 是 wheelleg_lab，parents[1] 是 source，parents[2] 才是项目根目录。
# 根据文件自身定位，换电脑、手动同步到服务器或切换终端目录后都无需改路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "assets" / "robot" / "v14_6"

# 机器人资产：无地面、无气弹簧驱动，留给后续自己的场景搭建使用。
# 它不是队友训练时处理过轮胎接触等设置的派生 USD。
ROBOT_USD = MODEL_DIR / "wheelbipev14--6_robot.usdc"

# 检查场景：交付文件已经带地面、灯光和原生气弹簧驱动，适合先打开观察。
# 本阶段只加载并暂停，不能再额外叠加训练代码里的气弹簧施力。
INSPECTION_SCENE_USD = MODEL_DIR / "wheelbipev14--6_gas_400_580.usdc"
