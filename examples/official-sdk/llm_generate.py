import json,time,os
from pathlib import Path
from furiosa_llm import LLM,SamplingParams
root=Path(__file__).resolve().parent
start=time.perf_counter()
llm=LLM('Qwen/Qwen3-8B-FP8',fxb=str(root/'qwen-small.fxb'),devices=os.environ.get('SAMPLE_LLM_DEVICE','npu:0'),max_model_len=128)
load=time.perf_counter()-start
out=llm.generate('The capital of France is',SamplingParams(temperature=0,max_tokens=8))
print(out)
(root/'llm-result.json').write_text(json.dumps({'load_seconds':load,'result':str(out)},indent=2))
