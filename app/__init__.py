"""Commercial Bank knowledge assistant."""
import os

# numpy (pulled in by the Pinecone/LangChain stacks) starts one OpenBLAS thread per core
# and can fail to allocate memory on Windows. The assistant does no heavy linear
# algebra, so a single BLAS thread is enough. Must run before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
