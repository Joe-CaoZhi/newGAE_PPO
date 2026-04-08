#!/bin/bash
# Monitor ICML experiment progress
while true; do
    N=$(find /Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment -name "*.json" -not -name "icml_summary.json" | wc -l)
    PROCS=$(ps aux | grep "run_icml_experiment" | grep -v grep | wc -l)
    echo "$(date '+%H:%M:%S')  completed=$N/60  procs=$PROCS"
    if [ "$N" -ge 60 ] || [ "$PROCS" -eq 0 ]; then
        echo "DONE!"
        break
    fi
    sleep 60
done

