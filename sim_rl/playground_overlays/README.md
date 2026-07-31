# MuJoCo Playground overlays

Files here are copied on top of the `sim_rl/mujoco_playground` submodule
checkout by `scripts/apply_playground_overlays.sh` (also invoked from
setup + train).

They exist because Aero Hand Open customizations (notably
`AeroCubeRotateZAxis38mm`) are not published on
`google-deepmind/mujoco_playground`, so a submodule gitlink alone is not
fetchable on other machines.

Re-export after editing the submodule:

```bash
cd sim_rl/mujoco_playground
git archive HEAD \
  learning/train_jax_ppo.py \
  mujoco_playground/_src/mjx_env.py \
  mujoco_playground/_src/manipulation/__init__.py \
  mujoco_playground/_src/manipulation/aero_hand/aero_hand_constants.py \
  mujoco_playground/_src/manipulation/aero_hand/rotate_z.py \
  mujoco_playground/_src/manipulation/aero_hand/xmls/reorientation_cube_38mm.xml \
  mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_cube_38mm.xml \
  mujoco_playground/config/manipulation_params.py \
| tar -x -C ../playground_overlays
```
