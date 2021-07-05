import os
import sys

import traceback

if __name__ == "__main__":
    return_code = 0
    try:
        AGENT_HOME = os.environ['AGENT_HOME']
    except Exception as e:
        traceback.print_exc()
        print('env_key is None. Finish process after running argument [{}].'.format(sys.argv[1:]))
        exit(1)
    sys.path.append(AGENT_HOME)
    import run_base
    return_code = run_base.main()

    exit(return_code)

