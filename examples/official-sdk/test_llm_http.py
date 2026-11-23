import subprocess,time,json,urllib.request,sys,os
from pathlib import Path
r=Path(__file__).resolve().parent
cmd=[str(Path(sys.executable).with_name('furiosa-llm')),'serve','Qwen/Qwen3-8B-FP8','--fxb',str(r/'qwen-small.fxb'),'--devices',os.environ.get('SAMPLE_LLM_DEVICE','npu:0'),'--max-model-len','128','--host','127.0.0.1','--port','18090','--served-model-name','qwen-test']
with (r/'llm-serve.log').open('w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=log,cwd=r)
 try:
  while True:
   if p.poll() is not None:raise RuntimeError('server exited '+str(p.returncode))
   try:
    urllib.request.urlopen('http://127.0.0.1:18090/health',timeout=2);break
   except OSError:time.sleep(2)
  body={'model':'qwen-test','messages':[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],'max_tokens':16,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
  request=urllib.request.Request('http://127.0.0.1:18090/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  result=json.load(urllib.request.urlopen(request,timeout=180));(r/'llm-http-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 finally:
  p.terminate()
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:p.kill();p.wait()
