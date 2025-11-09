#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python3}"
APP="${APP:-./konfig_2practice.py}"

cases=(
  "configs/simple_dag_3.xml"
  "configs/diamond_3.xml"
  "configs/with_cycle_3.xml"
  "configs/self_loop_3.xml"
  "configs/filter_3.xml"
  "configs/disconnected_3.xml"
)

for cfg in "${cases[@]}"; do
  echo
  echo "==================== $cfg ===================="
  $PY "$APP" --config "$cfg" | tee "out_${cfg##*/}.log"
done

echo
echo "Логи сохранены: out_3_*.log"
