#!/bin/bash
export OMP_NUM_THREADS=1

host=${1:-localhost}
port=${2:-60000}

PYTHON="./.venv/bin/python"

for i in {1..7}; do
  $PYTHON run_player.py --host $host --port $port -n $i -t MujocoCodebase &
done
