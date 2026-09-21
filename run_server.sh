#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"42 43 44 45 46"}
BASE=${BASE:-DeepPavlov/rubert-base-cased}
LARGE=${LARGE:-ai-forever/ruRoberta-large}
LOG=logs/server.log
FAILED=0

usage() {
    cat <<TXT
Прогоны на сервере с 4090. Каждый этап перезапускаем: готовые прогоны пропускаются.

    ./run_server.sh setup      venv и зависимости, один раз
    ./run_server.sh seeds      5 seeds x 2 модели x {none, grade}   20 прогонов  ~55 мин
    ./run_server.sh group      групповой сплит по фильму            10 прогонов  ~30 мин
    ./run_server.sh heads      CORN и веса классов                  20 прогонов  ~55 мин
    ./run_server.sh ladder     нижние ступени лестницы              30 прогонов  ~85 мин
    ./run_server.sh probe      линейный пробинг, без дообучения                  ~10 мин
    ./run_server.sh summary    все сводные таблицы
    ./run_server.sh all        seeds, group, heads, ladder, probe, summary

Этапы независимы и идут по убыванию важности: seeds закрывает главный пробел
(у ruRoberta пока один прогон, её разброс неизвестен), остальное — сверх того.

Переменные окружения: PY, SEEDS, BASE, LARGE.
TXT
}

check_gpu() {
    "$PY" - <<'PYCODE'
import sys, torch
if not torch.cuda.is_available():
    print("CUDA недоступна: torch видит только CPU.")
    print("Прогоны займут часы вместо минут. Переустановите torch с поддержкой CUDA.")
    sys.exit(1)
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {name}, {total:.0f} GB")
PYCODE
}

run() {
    echo "  → $*"
    if ! "$PY" src/train.py --skip-existing "$@" >> "$LOG" 2>&1; then
        echo "     ПРОВАЛ, хвост лога:"
        tail -5 "$LOG" | sed 's/^/     /'
        FAILED=$((FAILED + 1))
    fi
}

stage_setup() {
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    check_gpu
}

stage_seeds() {
    echo "=== seeds: разброс по инициализации, обе модели"
    for model in "$BASE" "$LARGE"; do
        for seed in $SEEDS; do
            for redact in none grade; do
                run --model "$model" --seed "$seed" --redact "$redact"
            done
        done
    done
}

stage_group() {
    echo "=== group: групповой сплит по фильму, пересечение train/test убрано"
    for model in "$BASE" "$LARGE"; do
        for seed in $SEEDS; do
            run --model "$model" --seed "$seed" --group movie_name
        done
    done
}

stage_heads() {
    echo "=== heads: ординальная голова и веса классов"
    for model in "$BASE" "$LARGE"; do
        for seed in $SEEDS; do
            run --model "$model" --seed "$seed" --head corn
            run --model "$model" --seed "$seed" --class-weights balanced
        done
    done
}

stage_ladder() {
    echo "=== ladder: нижние ступени, дообучением"
    for model in "$BASE" "$LARGE"; do
        for seed in $SEEDS; do
            for redact in last-sentence tail15 first-half; do
                run --model "$model" --seed "$seed" --redact "$redact"
            done
        done
    done
}

stage_probe() {
    echo "=== probe: линейный пробинг замороженных представлений"
    echo "перезапишет analysis/probe_*.csv числами с CUDA"
    "$PY" src/probe.py --model "$BASE"
    "$PY" src/probe.py --model "$LARGE"
}

delta() {
    if "$PY" src/aggregate.py --delta "$1" "$2" > /dev/null 2>&1; then
        echo "--- $1  минус  $2"
        "$PY" src/aggregate.py --delta "$1" "$2"
        echo
    fi
}

stage_summary() {
    echo "=== сводка"
    "$PY" src/aggregate.py
    echo
    echo "=== дельты по совпадающим seed'ам"
    delta rubert-base-cased rubert-base-cased_nograde
    delta ruRoberta-large ruRoberta-large_nograde
    delta ruRoberta-large rubert-base-cased
    delta ruRoberta-large_nograde rubert-base-cased_nograde
    delta rubert-base-cased_corn rubert-base-cased
    delta ruRoberta-large_corn ruRoberta-large
    delta rubert-base-cased_balanced rubert-base-cased
    delta ruRoberta-large_balanced ruRoberta-large
    echo "=== групповой сплит: тестовая выборка другая, парным сравнение не является"
    delta rubert-base-cased_group-movie rubert-base-cased
    delta ruRoberta-large_group-movie ruRoberta-large
    echo "=== калибровка"
    for run_dir in runs/*/; do
        [ -f "$run_dir/val_predictions.csv" ] || continue
        echo "--- $(basename "$run_dir")"
        "$PY" src/evaluate.py "$run_dir/predictions.csv" --calibrate
    done
}

mkdir -p logs
case "${1:-help}" in
    setup)   stage_setup ;;
    seeds)   check_gpu; stage_seeds ;;
    group)   check_gpu; stage_group ;;
    heads)   check_gpu; stage_heads ;;
    ladder)  check_gpu; stage_ladder ;;
    probe)   check_gpu; stage_probe ;;
    summary) stage_summary ;;
    all)     check_gpu; stage_seeds; stage_group; stage_heads; stage_ladder; stage_probe; stage_summary ;;
    *)       usage; exit 0 ;;
esac

if [ "$FAILED" -gt 0 ]; then
    echo
    echo "провалов: $FAILED, подробности в $LOG"
    exit 1
fi
