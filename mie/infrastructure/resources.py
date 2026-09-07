import resource
import time
import tracemalloc


class ResourceTracker:
    """Runtime / peak memory / CPU time for an experiment run."""
    def __enter__(self):
        tracemalloc.start()
        self.t0 = time.time()
        self.c0 = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        return self
    def __exit__(self, *a):
        self.runtime_s = round(time.time() - self.t0, 3)
        self.peak_mb = round(tracemalloc.get_traced_memory()[1] / 1e6, 1)
        tracemalloc.stop()
        self.cpu_s = round(resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.c0, 3)
    def as_dict(self):
        return {"runtime_s": self.runtime_s, "peak_mem_mb": self.peak_mb, "cpu_s": self.cpu_s}
