"""Yield our process when another compute job appears on the selected GPU."""
import os
import subprocess
import time

class ResourceGuard:
    def __init__(self, budget, gpu_uuid):
        self.budget = budget
        self.gpu_uuid = gpu_uuid
        self.last_check = float('-inf')

    def __getattr__(self, name):
        return getattr(self.budget, name)

    def check(self, reserve=5):
        self.budget.check(reserve)
        if time.monotonic() - self.last_check < 5:
            return
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise RuntimeError('Cannot verify shared GPU process state')
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split(',')]
            if len(fields) != 2:
                raise RuntimeError('Unrecognized shared GPU process state')
            uuid, pid = fields
            if uuid == self.gpu_uuid and pid != str(os.getpid()):
                self.budget.emit('resource_busy', reason='Another compute process appeared; yielding our GPU job')
                raise RuntimeError('RESOURCE_BUSY: another compute process is active')
        self.last_check = time.monotonic()
