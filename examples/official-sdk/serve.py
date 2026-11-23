from http.server import BaseHTTPRequestHandler,HTTPServer
import argparse,json,time
import sample
p=argparse.ArgumentParser();p.add_argument('mode',choices=['torch','tcl','reload','backend']);p.add_argument('--port',type=int,default=18085);args=p.parse_args()
m=sample.run(args.mode);a,b=sample.inputs();a=a.to('furiosa:0');b=b.to('furiosa:0')
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  if self.path!='/predict':self.send_error(404);return
  size=int(self.headers.get('Content-Length','0'))
  if size>1024:self.send_error(413);return
  try:
   body=json.loads(self.rfile.read(size));assert body=={} or body=={'sample':True}
   start=time.perf_counter()
   with sample.Profiler(str(sample.ROOT/('serve-'+args.mode+'-trace.json'))):
    y=m(a,b);y=y[0] if isinstance(y,(list,tuple)) else y;y=y.cpu()
   ref=sample.torch.bmm(*sample.inputs());sample.torch.testing.assert_close(y,ref,rtol=.05,atol=.05)
   result={'ok':True,'shape':list(y.shape),'max_abs_error':float((y-ref).abs().max()),'first_values':y.flatten()[:8].float().tolist(),'seconds':time.perf_counter()-start,'cpu_fallback':False}
   data=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
  except Exception as e:self.send_error(400,str(e))
print('READY 127.0.0.1:'+str(args.port),flush=True)
HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
