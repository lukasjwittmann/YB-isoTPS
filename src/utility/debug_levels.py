from enum import Enum
from functools import total_ordering

@total_ordering
class DebugLevel(int, Enum):
    """
    Enum class for specifying debug levels.
    """
    NO_DEBUG = 0                                                            # No debug logging
    LOG_PER_SITE_ERROR_AND_WALLTIME = 1                                     # store per site YB and TEBD error and walltime
    LOG_DISENTANGLER_TRIPARTITE_DECOMPOSITION_ITERATION_INFO = 2            # Logs per-call information of disentanglers and tripartite decomposition subroutines in a consecutive list
    LOG_CONSECUTIVE_ERROR_AND_WALLTIME = 3                                  # Store error and walltime of every YB and TEBD move in consecutive lists
    LOG_COLUMN_ERRORS = 4                                                   # Store column errors of YB moves and TEBD in a consecutive list

    LOG_YB_TEBD_ENVIRONMENTS = 8                                            # store the environment before each YB move in a list
    LOG_PER_ITERATION_DEBUG_INFO_TEBD_UPDATE_STEP = 9                       # store intermediate results of TEBD step after each iteration
    LOG_PER_ITERATION_DEBUG_INFO_DISENTANGLER_TRIPARTITE_DECOMPOSITION = 10 # Store the intermediate results of disentangling and tripartite decomposition after each iteration

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        elif other.__class__ == int:
            return self.value < other
        else:
            raise NotImplementedError

def check_debug_level(debug_dict = None, debug_level = DebugLevel.NO_DEBUG):
    """
    Checks if the given debug dictionary has a debug level larger or equal than the given debug level.

    Parameters
    ----------
    debug_dict : dictionary or None, optional
        the debug dictionary. See src/isoTPS/isoTPS.py for more information. If this is None, False is returned.
        Default: None.
    debug_level : DebugLevel, optional
        the debug level to check against. Default: DebugLevel.NO_DEBUG.

    Returns
    -------
    result : boolean
        wether the debug level of the debug dictionary is >= the given debug level.
    """
    return (debug_dict is not None) and ("debug_level" in debug_dict) and (debug_dict["debug_level"] >= debug_level)
    