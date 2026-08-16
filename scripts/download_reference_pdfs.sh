#!/usr/bin/env bash
# Download reference PDFs listed in reference/manifest.csv (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/reference/manifest.csv"
while IFS=, read -r level sublevel filename arxiv_id title; do
  [[ "$level" == "level" ]] && continue
  [[ -z "$arxiv_id" ]] && continue
  dir="$ROOT/reference"
  case "$level" in
    L0) dir="$dir/L0-start-here/papers" ;;
    L1)
      case "$sublevel" in
        compiler-autotuning) dir="$dir/L1-foundations/compiler-autotuning" ;;
        dsls) dir="$dir/L1-foundations/dsls" ;;
        llm-kernels) dir="$dir/L1-foundations/llm-kernels" ;;
      esac
      ;;
    L2) dir="$dir/L2-benchmarks/papers" ;;
    L3) dir="$dir/L3-llm-kernel-models/papers" ;;
    L4)
      case "$sublevel" in
        iterative-agents) dir="$dir/L4-agentic-search/iterative-agents" ;;
      esac
      ;;
    L5)
      case "$sublevel" in
        multi-agent) dir="$dir/L5-advanced-topics/multi-agent" ;;
        memory-and-retrieval) dir="$dir/L5-advanced-topics/memory-and-retrieval" ;;
      esac
      ;;
    PBT) dir="$dir/PBT-property-based-testing/papers" ;;
  esac
  mkdir -p "$dir"
  out="$dir/$filename"
  if [[ -f "$out" && -s "$out" ]]; then
    echo "skip $out"
    continue
  fi
  echo "fetch $arxiv_id -> $out"
  curl -fsSL -o "$out" "https://arxiv.org/pdf/${arxiv_id}.pdf"
  sleep 0.2
done < "$MANIFEST"
