cd /c/Users/Swastik/Desktop/CRAG
echo "=== knn audit chain (Phase A pass 1) ==="
for f in scratchpad/_l1kn/log_run_squad.txt scratchpad/_l1kn/log_run_webqsp.txt scratchpad/_l1kn/log_run_hotpot.txt; do
  [ -f "$f" ] && echo "  $(basename $f): $(tail -1 $f)"
done
ls scratchpad/_l1kn/run_*.json 2>/dev/null | sed 's/^/  have /'
echo "=== A2 (Phase A pass 2) ==="; tail -2 scratchpad/_l1ep/log_a2_chain.txt 2>/dev/null
for d in musique_clean squad_clean metaqa 2wiki_clean webqsp hotpotqa_clean; do
  [ -f scratchpad/_l1ep/log_a2_$d.txt ] && echo "  $d: $(tail -1 scratchpad/_l1ep/log_a2_$d.txt)"
done
echo "=== master (Phase B+C) ==="; tail -3 scratchpad/_l1ep/log_master_small.txt 2>/dev/null
tail -2 scratchpad/_l1ep/log_master_big.txt 2>/dev/null
echo "=== mem/cpu ==="
powershell -Command "Get-CimInstance Win32_OperatingSystem | ForEach-Object { 'freeGB ' + [math]::Round(\$_.FreePhysicalMemory/1MB,1) }; (Get-Process python -ErrorAction SilentlyContinue).Count"
