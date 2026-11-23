import subprocess,time,json,urllib.request,sys
from pathlib import Path
r=Path(__file__).resolve().parent;py=sys.executable;results={}
for i,mode in enumerate(['torch','tcl','reload','backend']):
 port=18085+i
 with (r/('serve-'+mode+'.log')).open('w') as log:
  proc=subprocess.Popen([py,str(r/'serve.py'),mode,'--port',str(port)],stdout=log,stderr=log,cwd=r)
  try:
   while True:
    if proc.poll() is not None:raise RuntimeError('server exited '+str(proc.returncode))
    if 'READY' in (r/('serve-'+mode+'.log')).read_text():break
    time.sleep(1)
   req=urllib.request.Request('http://127.0.0.1:'+str(port)+'/predict',data=b'{"sample":true}',headers={'Content-Type':'application/json'})
   results[mode]=json.load(urllib.request.urlopen(req,timeout=120));print(mode,json.dumps(results[mode]),flush=True)
  finally:
   proc.terminate();proc.wait(timeout=20)
(r/'http-results.json').write_text(json.dumps(results,indent=2))
