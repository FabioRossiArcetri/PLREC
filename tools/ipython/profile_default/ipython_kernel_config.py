# Kernels started with IPYTHONDIR=PLREC/tools/ipython (notebook execution in
# run_lantern.sh / leonardo/extras.slurm): no history database -- many kernels
# run concurrently, and nothing here needs it. The startup/ folder holds the
# figure export for the slides.
c = get_config()  # noqa: F821
c.HistoryManager.enabled = False
