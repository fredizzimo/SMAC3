import logging
import json
from subprocess import Popen, PIPE

from smac.tae.execute_ta_run import StatusType

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class ExecuteTARunAClib(object):

    """
        executes a target algorithm run with a given configuration
        on a given instance and some resource limitations
        Uses the AClib 2.0 style

        Attributes
        ----------
        ta : string
            the command line call to the target algorithm (wrapper)
        run_obj: str
            run objective (runtime or quality)
    """

    def __init__(self, ta, run_obj="runtime"):
        """
        Constructor

        Parameters
        ----------
            ta : list
                target algorithm command line as list of arguments
            run_obj: str
                run objective of SMAC
        """
        self.ta = ta
        self.logger = logging.getLogger("ExecuteTARun")
        self.run_obj = run_obj

    def run(self, config, instance=None,
            cutoff=99999999999999.,
            seed=12345):
        """
            runs target algorithm <self.ta> with configuration <config> on
            instance <instance> with instance specifics <specifics>
            for at most <cutoff> seconds and random seed <seed>

            Parameters
            ----------
                config : dictionary (or similar)
                    dictionary param -> value
                instance : string
                    problem instance
                cutoff : double
                    runtime cutoff
                seed : int
                    random seed
            Returns
            -------
                status: enum of StatusType (int)
                    {SUCCESS, TIMEOUT, CRASHED, ABORT}
                cost: float
                    cost/regret/quality/runtime (float) (None, if not returned by TA)
                runtime: float
                    runtime (None if not returned by TA)
                additional_info: dict
                    all further additional run information
        """

        if instance is None:
            instance = "0"

        # TOOD: maybe replace fixed instance specific and cutoff_length (0) to
        # other value
        cmd = []
        cmd.extend(self.ta)
        cmd.extend(["--instance", instance,
                    "--cutoff", str(cutoff),
                    "--seed", str(seed),
                    "--config"
                    ])

        for p in config:
            cmd.extend(["-" + p, config[p]])

        self.logger.debug("Calling: %s" % (" ".join(cmd)))
        p = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE)
        stdout_, stderr_ = p.communicate()

        self.logger.debug("Stdout: %s" % (stdout_))
        self.logger.debug("Stderr: %s" % (stderr_))

        for line in stdout_.split("\n"):
            if line.startswith("Result of this algorithm run:"):
                fields = ":".join(line.split(":")[1:])
                results = json.loads(fields)

        if results["status"] in ["SAT", "UNSAT", "SUCCESS"]:
            status = StatusType.SUCCESS
        elif results["status"] in ["TIMEOUT"]:
            status = StatusType.TIMEOUT
        elif results["status"] in ["CRASHED"]:
            status = StatusType.CRASHED
        elif results["status"] in ["ABORT"]:
            status = StatusType.ABORT
        elif results["status"] in ["MEMOUT"]:
            status = StatusType.MEMOUT

        if status in [StatusType.CRASHED, StatusType.ABORT]:
            self.logger.warn(
                "Target algorithm crashed. Last 5 lines of stdout and stderr")
            self.logger.warn(stdout_.split("\n")[-5:])
            self.logger.warn(stderr_.split("\n")[-5:])

        if results.get("runtime") is None:
            self.logger.warn(
                "The target algorithm has not returned a runtime -- imputed by 0.")
            # (TODO) Check 0
            results["runtime"] = 0

        runtime = float(results["runtime"])

        if self.run_obj == "quality" and results.get("cost") is None:
            self.logger.error(
                "The target algorithm has not returned a quality/cost value" +
                "although we optimize cost.")
            # (TODO) Do not return 0
            results["cost"] = 0

        if self.run_obj == "runtime":
            cost = float(results["runtime"])
        else:
            cost = float(results["cost"])

        del results["status"]
        try:
            del results["runtime"]
        except KeyError:
            pass
        try:
            del results["cost"]
        except KeyError:
            pass

        return status, cost, runtime, results
