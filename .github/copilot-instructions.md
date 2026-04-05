# Project Guidelines

## Code Style
- Keep edits minimal and consistent with existing patterns in nearby files.
- Preserve CLI argument order and default values in root shell entrypoints unless a task explicitly requires changing behavior.
- In realtime control paths under deploy_real/, avoid adding blocking operations inside control loops.

## Architecture
- legged_gym/ contains RL environments and training logic; root scripts (for example train.sh and to_onnx.sh) are thin wrappers around legged_gym/legged_gym/scripts/.
- deploy_real/ contains runtime servers:
  - server_motion_lib.py is a high-level motion source.
  - server_low_level_g1_sim.py and server_low_level_g1_real.py are low-level controller servers using ONNX policies.
- Redis is the bridge between high-level motion streaming and low-level control.
- GMR/ is a separate retargeting component with its own dependency constraints.

## Build and Test
- Use the environment intended for the target component before running Python:
  - For TWIST2 training/deployment code, run: source .venv-twist2/bin/activate
  - For GMR retargeting code, run: source .venv-gmr/bin/activate
- Common project commands:
  - bash train.sh <experiment_id> <device>
  - bash to_onnx.sh <checkpoint_path>
  - bash run_motion_server.sh [robot] [motion_file]
  - bash sim2sim.sh [robot] [onnx_path]
  - bash teleop.sh
- For quick Python syntax checks after edits, prefer py_compile in the correct environment.
- Do not run long-running GPU or hardware-dependent commands unless the task explicitly asks for them.

## Conventions
- The project intentionally uses two Python environments because isaacgym and mujoco/GMR require different Python versions.
- Many workflows require a running Redis server; remote teleop may require Redis network configuration.
- Link to existing docs instead of duplicating operational details:
  - README.md (installation and end-to-end usage)
  - doc/TELEOP.md (teleop pipeline and controls)
  - doc/unitree_g1.md and doc/unitree_g1.zh.md (physical robot setup)
  - GMR/README.md and GMR/DOC.md (retargeting module)
  - memories/repo/twist2-notes.md (repository-specific ONNX/export gotchas)