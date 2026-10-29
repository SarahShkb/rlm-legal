# Invalid first attempt

All 3 runs here failed with HTTP 400 before any work: the Modal server started rlm30b with the
8,192-token sub-LLM window, and the 32K root asked for up to 30,081 answer tokens. Fixed in
ec2765e (32,768-token window for rlm30b). The valid pilot is in ../rlm30b_32k_pilot2/.
