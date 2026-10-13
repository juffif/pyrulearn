#!/usr/bin/env bash
# Re-run every demo's full version, one after the other, from the repository root:
#
#     tools/run_all_demos.sh            # all demos
#     tools/run_all_demos.sh representations dense_vs_modular    # just these
#
# Sequential on purpose: most reports are about fit times, which concurrent load skews.
# Run it on an otherwise idle machine, in tmux or with nohup. Results are cached per demo
# (demos/_*_cache), so an interrupted run continues where it stopped -- but the cache key
# doesn't contain the code version, so delete the caches (`rm -rf demos/_*_cache`) before
# a run on new code. A demo that fails doesn't stop the others; a summary is printed at
# the end. Logs: demos/_logs/<demo>.log.
#
# External tools, where a demo needs them (otherwise it records the learner as failed):
#     WEKA_JAR, WEKA_JAVA  Weka (heuristic_comparison, numeric_discretization, ripper_comparison,
#                          seco_learners_comparison, native_vs_interfaced)
#     LORD_DIR             a checkout of github.com/vqphuynh/LORD with compiled classes in bin/
#                          (seco_learners_comparison); or LORD_CLASSPATH / LORD_JAVA
#     PYRULEARN_DATA_HOME  where the OpenML datasets are cached
set -u
cd "$(dirname "$0")/.."

# the full comparisons, most important first; covering and heuristic_isometrics have no
# quick/full distinction
ALL="representations dense_vs_modular seco_learners_comparison heuristic_comparison \
ripper_comparison numeric_discretization multiclass_decomposition native_vs_interfaced \
pool_distillers covering_boosting covering heuristic_isometrics"
NO_FLAG=" covering heuristic_isometrics "

DEMOS="${*:-$ALL}"
PYTHON="${PYTHON:-python}"
mkdir -p demos/_logs
failed=""
for demo in $DEMOS; do
    flag="--full"
    case "$NO_FLAG" in *" $demo "*) flag="" ;; esac
    echo "=== $demo $flag ($(date '+%F %T'))"
    if PYTHONUTF8=1 "$PYTHON" "demos/$demo.py" $flag > "demos/_logs/$demo.log" 2>&1; then
        echo "    done ($(date '+%T'))"
    else
        echo "    FAILED -- see demos/_logs/$demo.log"
        failed="$failed $demo"
    fi
done
echo
if [ -n "$failed" ]; then
    echo "failed:$failed"
    exit 1
fi
echo "all demos finished"
