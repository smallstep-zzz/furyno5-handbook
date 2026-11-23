# Activate the SDK-compatible Python environment before sourcing this file.
unset PYTHONPATH
export SDK_PY="$(command -v python)"
# Set this to the NPU assigned to you. Do not select a device used by another job.
export SAMPLE_LLM_DEVICE="${SAMPLE_LLM_DEVICE:-npu:0}"
