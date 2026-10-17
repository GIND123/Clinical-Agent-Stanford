#!/usr/bin/env bash
# Run at boot from the user's crontab (@reboot): restart the sync daemon, and any GPU queue that has not finished,
# while the approved GPU window lasts (to Mon Oct 19, 10:30 IST). Queues skip finished steps (.done markers) and
# training resumes from its newest checkpoint, so a reboot costs at most the steps since the last checkpoint.
# After the deadline it only restarts the sync daemon.
cd "$(dirname "$0")/.."
sleep 90  # let the NVIDIA driver and the network come up
mkdir -p outputs/logs
echo "$(date -Is) resume_after_reboot" >> outputs/logs/resume.log
pgrep -f "python scripts/sync_daemon.py" >/dev/null || (setsid nohup bash -c \
  'source scripts/env_gpu.sh && CUDA_VISIBLE_DEVICES= exec python scripts/sync_daemon.py --interval 900' \
  >> outputs/logs/sync.log 2>&1 < /dev/null &)
if [ "$(date +%s)" -lt "$(date -d "2026-10-19 10:30 +0530" +%s)" ]; then
  for n in 10 11; do
    grep -q "queue v$n finished" "outputs/logs/queue_v$n.log" 2>/dev/null && continue
    pgrep -f "scripts/queue_v$n.sh" >/dev/null && continue
    (setsid nohup bash "scripts/queue_v$n.sh" >> "outputs/logs/queue_v$n.log" 2>&1 < /dev/null &)
    echo "$(date -Is) restarted queue_v$n" >> outputs/logs/resume.log
    sleep 2
  done
fi
