import os, json
from time import sleep
import subprocess as sp
from jinja2 import Environment, FileSystemLoader, StrictUndefined

SLURM_STATE_MAP = {
    'PD': 'pending',   'PENDING': 'pending',
    'R': 'running',    'RUNNING': 'running',
    'CG': 'running',   'COMPLETING': 'running',
    'CD': 'done',      'COMPLETED': 'done',
    'F': 'failed',     'FAILED': 'failed',
    'CA': 'failed',    'CANCELLED': 'failed',
    'TO': 'failed',    'TIMEOUT': 'failed',
    'NF': 'failed',    'NODE_FAIL': 'failed',
    'OOM': 'failed',   'OUT_OF_MEMORY': 'failed',
}

class Scheduler:
    """
    Unified interface over a real queueing system (Slurm) or plain local
    execution (e.g. a login node with no queue, one job at a time).
    scheduler_type selects the backend.

    For 'local', submit_job runs synchronously and blocks until the job
    finishes (or raises on failure) -- there's no background process to
    poll, so get_job_status/block_until_complete/block_max_submit are all
    no-ops for local: by the time you'd call them, the job is already
    done.
    """

    def __init__(self, type='slurm', user=None, queue=None,
                 max_submit=None, check_freq=30):
        if type not in ('slurm', 'local'):
            raise ValueError(f"scheduler_type must be 'slurm' or 'local', got {type!r}")
        self.scheduler_type = type
        self.user = user or "$USER"
        self.queue = queue
        self.max_submit = max_submit
        self.check_freq = check_freq

    # -- shared helpers -------------------------------------------------

    @staticmethod
    def normalize_slurm_state(raw: str) -> str:
        raw = raw.strip()
        key = raw.split()[0].rstrip('+').upper() if raw else ''
        return SLURM_STATE_MAP.get(key, 'unknown')

    @staticmethod
    def get_job_id(str_output: str) -> int:
        split_output = str_output.split()
        for i in range(len(split_output) - 3):
            if (split_output[i] == 'Submitted'
                    and split_output[i + 1] == 'batch'
                    and split_output[i + 2] == 'job'
                    and split_output[i + 3].isdigit()):
                return int(split_output[i + 3])
            if (split_output[i] == 'srun:' and split_output[i + 1] == 'job'
                    and split_output[i + 2].isdigit()):
                return int(split_output[i + 2])
        if (len(split_output) >= 3 and split_output[-3] == 'srun:'
                and split_output[-2] == 'job' and split_output[-1].isdigit()):
            return int(split_output[-1])
        raise ValueError(f'Output string is unexpected format: {str_output.strip()}')

    # -- job status -------------------------------------------------------

    def get_job_status(self, job_ids):
        """
        Return a list of normalized states ('pending', 'running', 'done',
        'failed', 'unknown'), one per job id, in the same order as job_ids.
        No-op for local: submit_job already blocks until completion, so
        there's nothing left to poll by the time this could be called.
        """
        if self.scheduler_type == 'local':
            return ['done'] * len(job_ids if isinstance(job_ids, list) else [job_ids])

        if isinstance(job_ids, int):
            job_ids = [job_ids]
        if not isinstance(job_ids, list) or not all(isinstance(j, int) for j in job_ids):
            raise ValueError('Must be int or list of ints')
        if len(job_ids) == 0:
            return []

        job_str = ','.join(str(j) for j in job_ids)

        states = {}
        cmd = f'squeue -j {job_str} -o "%i %t %R"'
        output = sp.run(cmd, capture_output=True, shell=True, encoding='UTF-8')
        if output.returncode == 0:
            for line in output.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    states[int(parts[0])] = self.normalize_slurm_state(parts[1])

        missing = [j for j in job_ids if j not in states]
        if missing:
            missing_str = ','.join(str(j) for j in missing)
            cmd = f'sacct -j {missing_str} -n -P -X -o JobID,State'
            output = sp.run(cmd, capture_output=True, shell=True, encoding='UTF-8')
            if output.returncode == 0:
                for line in output.stdout.splitlines():
                    jid, _, state = line.partition('|')
                    if jid.isdigit():
                        states[int(jid)] = self.normalize_slurm_state(state)

        return [states.get(j, 'unknown') for j in job_ids]

    # -- submission -------------------------------------------------------

    def submit_job(self, submit_file, max_submit=None, check_freq=None):
        """
        Submit a job. For 'slurm', submit_file is passed to sbatch and
        this returns immediately with the job id. For 'local', submit_file
        is executed directly and this BLOCKS until it finishes, raising on
        nonzero exit; there is no job id to return.
        """
        submit_file = os.path.realpath(submit_file)
        if not os.path.isfile(submit_file):
            raise FileNotFoundError(f'Submit file not found: {submit_file}')

        if self.scheduler_type == 'local':
            return self._submit_job_local(submit_file)

        self.block_max_submit(max_submit=max_submit, check_freq=check_freq)
        return self._submit_job_slurm(submit_file)

    def _submit_job_slurm(self, submit_file):
        cmd = f'sbatch {os.path.basename(submit_file)}'
        output = sp.run(cmd, cwd=os.path.dirname(submit_file),
                        capture_output=True, shell=True, encoding='UTF-8')

        if output.returncode != 0:
            raise RuntimeError(f'sbatch failed (code {output.returncode}): '
                               f'{output.stderr.strip()}')

        return self.get_job_id(output.stdout)

    def _submit_job_local(self, submit_file):
        result = sp.run(['bash', os.path.basename(submit_file)],
                        cwd=os.path.dirname(submit_file))
        if result.returncode != 0:
            raise RuntimeError(f'Local job failed (code {result.returncode}): {submit_file}')
        print(f'Local job {submit_file} completed.')
        return None

    # -- waiting ------------------------------------------------------------

    def block_until_complete(self, calc_ids, check_freq=None):
        """No-op for local: submit_job already blocked until the job finished."""
        if self.scheduler_type == 'local':
            return

        check_freq = check_freq if check_freq is not None else self.check_freq

        if type(calc_ids) is list:
            if all(isinstance(cid, int) for cid in calc_ids):
                remaining_jobs = list(calc_ids)
            else:
                raise TypeError('calc_ids must be a list of int!')
        elif type(calc_ids) is int:
            remaining_jobs = [calc_ids]
        else:
            raise TypeError('Job IDs must be a single int, or a list of int!')

        wait_cycle_counter = 0
        failed_jobs = []
        while len(remaining_jobs) > 0:
            job_states = self.get_job_status(remaining_jobs)
            jobs_to_remove = []
            for job_id, job_state in zip(remaining_jobs, job_states):
                if job_state == 'done':
                    print(f'Job {job_id} completed, removing from waiting list')
                    jobs_to_remove.append(job_id)
                elif job_state == 'failed':
                    print(f'Job {job_id} failed, removing from waiting list')
                    jobs_to_remove.append(job_id)
                    failed_jobs.append(job_id)
            for job_id in jobs_to_remove:
                remaining_jobs.remove(job_id)
            if len(remaining_jobs) == 0:
                break
            if wait_cycle_counter % 5 == 0:
                print(f'Wait iteration {wait_cycle_counter}, '
                      f'[cycle time = {check_freq} s] with {len(remaining_jobs)} jobs left '
                      f'to complete')
            wait_cycle_counter += 1
            sleep(check_freq)

        if failed_jobs:
            raise RuntimeError(f'Job(s) {failed_jobs} failed while waiting for completion')

        print(f'Jobs {calc_ids} have completed, continuing...')

    # -- active job counting / rate limiting -------------------------------

    def get_active_job_count(self, queue=None, user=None) -> int:
        """Always 0 for local: submit_job is synchronous, so nothing is ever
        left running by the time this could be called."""
        if self.scheduler_type == 'local':
            return 0

        queue = queue if queue is not None else self.queue
        user = user if user is not None else self.user

        cmd = f"squeue -u {user} "
        if queue is not None:
            cmd += f"-p {queue} "
        cmd += "--noheader | wc -l"
        out = sp.run(cmd, capture_output=True, shell=True, encoding='UTF-8')
        return int(out.stdout)

    def block_max_submit(self, max_submit=None, queue=None, user=None, check_freq=None):
        """No-op for local: jobs already run one at a time, serially."""
        if self.scheduler_type == 'local':
            return

        max_submit = max_submit if max_submit is not None else self.max_submit
        check_freq = check_freq if check_freq is not None else self.check_freq

        if max_submit is None:
            return  # no cap configured -- nothing to block on

        wait_cycle_counter = 0
        job_count = self.get_active_job_count(queue=queue, user=user)
        while job_count >= max_submit:
            if wait_cycle_counter % 5 == 0:
                print(f'Wait iteration {wait_cycle_counter} '
                      f'[cycle time = {check_freq} s]: '
                      f'{job_count} active jobs >= max_submit '
                      f'({max_submit}), waiting...')
            wait_cycle_counter += 1
            sleep(check_freq)
            job_count = self.get_active_job_count(queue=queue, user=user)

        print(f'Active job count ({job_count}) now below max_submit '
              f'({max_submit}), continuing...')

def fill_template(template: str, template_fill: dict, dest_path: str):
    """
    Render a Jinja2 template file and write the result to dest_path.

    :param template: path to the template file
    :type template: str
    :param template_fill: variables to fill into the template
    :type template_fill: dict
    :param dest_path: where the filled file should be written
    :type dest_path: str
    :return: dest_path, for convenience
    """
    template = os.path.abspath(template)
    if not os.path.isfile(template):
        raise FileNotFoundError(f'Template not found: {template}')

    # Loading from the template's own directory lets templates use
    # {% include %} / {% extends %} with relative paths.
    env = Environment(
        loader=FileSystemLoader(os.path.dirname(template)),
#        undefined=StrictUndefined,   # raise on missing variables instead of silently rendering ''
        keep_trailing_newline=True,  # don't strip the final newline (matters for shell scripts)
    )
    rendered = env.get_template(os.path.basename(template)).render(**template_fill)

    dest_dir = os.path.dirname(os.path.abspath(dest_path))
    os.makedirs(dest_dir, exist_ok=True)
    with open(dest_path, 'w') as f:
        f.write(rendered)

    return dest_path

class Checkpointer:
    """
    Minimal class for reading, writing checkpoint & logging
    """

    def __init__(self, cp_file=None):
        self.cp_file = cp_file if cp_file is not None else "checkpoint.json"

    def write_checkpoint(self, checkpoint):
        with open(self.cp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def read_checkpoint(self):
        if not os.path.isfile(self.cp_file):
            return {}
        with open(self.cp_file) as f:
            return json.load(f)