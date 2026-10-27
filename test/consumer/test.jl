if "-q" in ARGS
    redirect_stderr(devnull)
end

import JACC
JACC.@init_backend

a = JACC.ones(100)
if JACC.parallel_reduce(a) != 100
    exit(1)
end
