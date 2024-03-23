'''
parallel run example:
def slow_power(x, p):
    time.sleep(1)
    return x ** p
from joblib import Parallel, delayed
number_of_cpu = joblib.cpu_count()
delayed_funcs = [delayed(slow_power)(i, 5) for i in range(10)]
parallel_pool = Parallel(n_jobs=number_of_cpu)
parallel_pool(delayed_funcs)
+> equivalent to:
def delayer(gunc):
    gunc = delayed(gunc)
    def wrapper(*args, **kws):
        return gunc(*args, **kws)
    return wrapper
def parallel_runner(jobs: iterator, n_jobs = -1):
    return Parallel(n_jobs=n_jobs)(jobs)
@ delayer
def slow_power2(x, p):
    time.sleep(1)
    return x + p, x - p
jobs = (slow_power2(i, 5) for i in range(20))
parallel_runner(jobs)
'''
from joblib import delayed, Parallel


def delayer(gunc):
    # the decorator for delay running funcs, see examples above
    gunc = delayed(gunc)

    def wrapper(*args, **kws):
        return gunc(*args, **kws)

    return wrapper


def parallel_run(jobs, n_jobs: int = -1) -> list:
    """
    :param jobs: iterator of jobs to run
    :param n_jobs: number of cpus to run, -1 = use all available core
    :return: [(results from job1), (results from job2), (results from job3)]
    """
    return Parallel(n_jobs=n_jobs)(jobs)