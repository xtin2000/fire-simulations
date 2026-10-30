#!/bin/zsh
# Build, run and export one burn scenario by index (called by run_burn_dataset.sh).
cd "${0:A:h}/.." || exit 1
id=$(printf "burn%05d" "$1")
[[ -f burn_dataset/$id.npz ]] && exit 0
if ! .venv/bin/python -m fire_sim.burn_scenarios --start "$1" --count 1 > /dev/null; then
  echo "$(date +%T) $id FAILED-generate"
  exit 0
fi
(cd burn_scenarios/$id && fds $id.fds > run.log 2>&1)
if tail -1 burn_scenarios/$id/run.log | grep -q successfully \
    && .venv/bin/python -m fire_sim.burn_export burn_scenarios/$id; then
  echo "$(date +%T) $id ok"
else
  echo "$(date +%T) $id FAILED"
fi
