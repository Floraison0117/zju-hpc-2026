#!/usr/bin/env bash
# metadata_store.sh — 搜索树元数据 CRUD（KernelEvolve metadata + object store）
# 用法：
#   bash assets/lab4/opt/search/metadata_store.sh add <id> <parent> <F> <is_buggy> <lever>
#   bash assets/lab4/opt/search/metadata_store.sh list
#   bash assets/lab4/opt/search/metadata_store.sh best [N]
#   bash assets/lab4/opt/search/metadata_store.sh lineage <id>
set -euo pipefail
TREE="assets/lab4/opt/search/tree"
mkdir -p "$TREE"
CMD="${1:?add|list|best|lineage required}"

case "$CMD" in
add)
  ID="${2:?id}"; PAR="${3:-root}"; F="${4:?F}"; BUG="${5:?is_buggy}"; LEV="${6:?lever}"
  cat > "$TREE/$ID.md" <<EOF
- id: $ID
- parent: $PAR
- F: $F
- is_buggy: $BUG
- lever: $LEV
- ts: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
  echo "added $TREE/$ID.md"
  ;;
list)
  for f in "$TREE"/*.md; do
    [ -f "$f" ] || continue
    grep -hE '^- (id|F|is_buggy|lever):' "$f" | tr '\n' ' ' | sed 's/- //g; s/: /:/g'
    echo
  done
  ;;
best)
  N="${2:-5}"
  for f in "$TREE"/*.md; do
    [ -f "$f" ] || continue
    F=$(grep -oP '^- F: \K[\d.]+' "$f" 2>/dev/null || echo 0)
    BUG=$(grep -oP '^- is_buggy: \K[01]' "$f" 2>/dev/null || echo 1)
    [ "$BUG" = 0 ] && echo "$F $f"
  done | sort -rn | head -"$N"
  ;;
lineage)
  ID="${2:?id required}"
  cur="$TREE/$ID.md"
  while [ -f "$cur" ]; do
    grep -hE '^- (id|parent|lever|F):' "$cur" | tr '\n' ' '; echo
    PAR=$(grep -oP '^- parent: \K\S+' "$cur" 2>/dev/null || echo root)
    [ "$PAR" = root ] && break
    cur="$TREE/$PAR.md"
  done
  ;;
*) echo "unknown cmd $CMD" >&2; exit 2 ;;
esac
