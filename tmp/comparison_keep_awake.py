"""Temporary idle-sleep inhibition while this explicitly resumed batch is active.

Does not change the power plan or keep the display on. Manual sleep is unaffected.
The execution-state request belongs to this process and ends when it exits.
"""
from pathlib import Path
import ctypes
import json
import time

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    output=root/'experiments/artifacts/comparison_v3'
    request=ctypes.windll.kernel32.SetThreadExecutionState
    if not request(0x80000001):raise RuntimeError('Unable to request temporary system execution state')
    print('[batch power] idle sleep inhibited for this active batch; display policy unchanged',flush=True)
    try:
        while True:
            pause=output/'pause_status.json'
            if pause.exists() and json.loads(pause.read_text('utf-8')).get('status')=='paused_by_user':break
            verified=output/'verification.json'
            if verified.exists() and json.loads(verified.read_text('utf-8')).get('complete'):break
            time.sleep(15)
    finally:
        request(0x80000000)
        print('[batch power] temporary idle-sleep request released',flush=True)
