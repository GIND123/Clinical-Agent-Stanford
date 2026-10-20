#!/usr/bin/env bash
# Run at boot from the user's crontab (@reboot): restart the sync daemon, and any GPU queue that has not finished,
# while the approved GPU window lasts (to Wed Oct 21, 18:00 IST; the user asked on Oct 17 to finish every run). Queues skip finished steps (.done markers) and
# training resumes from its newest checkpoint, so a reboot costs at most the steps since the last checkpoint.
# After the deadline it only restarts the sync daemon.
cd "$(dirname "$0")/.."
sleep 90  # let the NVIDIA driver and the network come up
mkdir -p outputs/logs
echo "$(date -Is) resume_after_reboot" >> outputs/logs/resume.log
pgrep -f "python scripts/sync_daemon.py" >/dev/null || (setsid nohup bash -c \
  'source scripts/env_gpu.sh && CUDA_VISIBLE_DEVICES= exec python scripts/sync_daemon.py --interval 900' \
  >> outputs/logs/sync.log 2>&1 < /dev/null &)
# Crash-loop guard: the machine has been resetting with firmware-logged hardware errors, several times within
# minutes of GPU work restarting (Oct 17, 22:10-22:30). If a resume already ran in the last 45 minutes, restart only
# the sync daemon and leave the GPU idle; otherwise wait 5 minutes before putting the GPU back under load.
now=$(date +%s)
recent=$(grep "resume_after_reboot" outputs/logs/resume.log | tail -2 | head -1 | cut -c1-25)
if [ -n "$recent" ] && [ $(( now - $(date -d "$recent" +%s) )) -lt 2700 ]; then
  echo "$(date -Is) crash-loop guard: another resume within 45 min; GPU queues not restarted" >> outputs/logs/resume.log
  exit 0
fi
sleep 300
if [ "$(date +%s)" -lt "$(date -d "2026-10-21 18:00 +0530" +%s)" ]; then
  for n in 10 11; do
    grep -q "queue v$n finished" "outputs/logs/queue_v$n.log" 2>/dev/null && continue
    pgrep -f "scripts/queue_v$n.sh" >/dev/null && continue
    (setsid nohup bash "scripts/queue_v$n.sh" >> "outputs/logs/queue_v$n.log" 2>&1 < /dev/null &)
    echo "$(date -Is) restarted queue_v$n" >> outputs/logs/resume.log
    sleep 2
  done
fi
