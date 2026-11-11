
using Replay

repl_script = """
import JACC
JACC.@init_backend
matrix = JACC.zeros((10,10));
JACC.@parallel_for range=(10,10) ((i,j,x)->x[i,j] = i+j-1)(matrix);
matrix
"""

replay(repl_script, stdout, julia_project=@__DIR__, use_ghostwriter=true)

