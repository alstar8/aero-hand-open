"""Parse a run_ppo_on_hand.py --debug_every=1 log into a CSV comparable to
sim_rollout_log.csv (same action_i / motor_target_i columns, plus real-only
act_cmd_i (deg) and actuations_i (deg) columns)."""
import csv
import pathlib
import re
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

IN_LOG = sys.argv[1] if len(sys.argv) > 1 else str(_REPO_ROOT / "logs" / "real_full_log.log")
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else str(_REPO_ROOT / "logs" / "real_rollout_log.csv")

LINE_RE = re.compile(
    r"step=(?P<step>\d+)\s+act=\[(?P<act>[^\]]+)\]\s+"
    r"motor_targets=\[(?P<mt>[^\]]+)\]\s+act_cmd=\[(?P<cmd>[^\]]+)\]\s+"
    r"actuations=\[(?P<act2>[^\]]+)\]"
)


def parse_floats(s):
  return [float(x) for x in s.split()]


def main():
  rows = []
  with open(IN_LOG) as f:
    for line in f:
      m = LINE_RE.search(line)
      if not m:
        continue
      step = int(m.group("step"))
      act = parse_floats(m.group("act"))
      mt = parse_floats(m.group("mt"))
      cmd = parse_floats(m.group("cmd"))
      actu = parse_floats(m.group("act2"))
      rows.append((step, act, mt, cmd, actu))

  header = (
      ["step"]
      + [f"action_{i}" for i in range(7)]
      + [f"motor_target_{i}" for i in range(7)]
      + [f"act_cmd_deg_{i}" for i in range(7)]
      + [f"actuations_deg_{i}" for i in range(7)]
  )
  pathlib.Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
  with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for step, act, mt, cmd, actu in rows:
      writer.writerow([step] + act + mt + cmd + actu)

  print(f"Parsed {len(rows)} steps -> {OUT_CSV}")


if __name__ == "__main__":
  main()
