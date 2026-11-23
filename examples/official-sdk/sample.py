from pathlib import Path
import argparse,json,time
import torch
import furiosa.torch as ft
from furiosa.torch.profiler import Profiler
ROOT=Path(__file__).resolve().parent
ft.config.set_cpu_fallback(False)
class Model(torch.nn.Module):
    def forward(self,a,b):return torch.bmm(a,b)
def inputs():
    torch.manual_seed(23)
    return torch.randn(1,128,64,dtype=torch.bfloat16),torch.randn(1,64,128,dtype=torch.bfloat16)
def make():
    a,b=inputs();torch.export.save(torch.export.export(Model(),(a,b)),ROOT/'model.pt2')
def compile_model(mode):
    a,b=inputs()
    if mode=='torch':m=torch.export.load(ROOT/'model.pt2').module()
    elif mode=='tcl':
        # Explicitly replace this sample model's sole bmm with the equivalent TCL.
        original=torch.export.load(ROOT/'model.pt2').module()
        torch.testing.assert_close(original(a,b),torch.bmm(a,b),rtol=0,atol=0)
        m=ft.TclModule((ROOT/'matmul.tc').read_text())
    else:
        # Official wrapper performs its own toolchain compatibility check.
        return ft.compile_from_visa(ROOT/'visa','identity')
    out=ft.CompileModule.from_module(m,(a,b))
    (ROOT/(mode+'.runnable')).write_bytes(out.runnable._inner.serialize())
    return out

def run(mode):
    a,b=inputs();ref=torch.bmm(a,b);start=time.perf_counter()
    if mode=='edf':m=ft.Runnable(ft.ir.Runnable.from_serialized_edf((ROOT/'matmul.edf').read_bytes()))
    elif mode=='reload':m=ft.Runnable(ft.ir.Runnable.deserialize((ROOT/'torch.runnable').read_bytes()))
    elif mode=='backend':m=torch.compile(torch.export.load(ROOT/'model.pt2').module(),backend=ft.backend,fullgraph=True)
    else:m=compile_model(mode)
    dev=torch.device('furiosa:0');m=m.to(dev);aa=a.to(dev);bb=b.to(dev)
    with Profiler(str(ROOT/(mode+'-trace.json'))):
        y=m(aa,bb);y=y[0] if isinstance(y,(list,tuple)) else y;y=y.cpu()
    torch.testing.assert_close(y,ref,atol=.05,rtol=.05)
    result={'mode':mode,'ok':True,'max_abs_error':float((y-ref).abs().max()),'elapsed_seconds':time.perf_counter()-start,'cpu_fallback_enabled':ft.config.get_cpu_fallback(),'output_shape':list(y.shape)}
    (ROOT/(mode+'-result.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return m
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['make','torch','tcl','visa','reload','backend','edf']);a=p.parse_args()
    if a.action=='make':
        make()
        from furiosa.tcl import Kernel, specialize_kernel
        from furiosa.tcl.examples.contract import batched_matmul
        (ROOT/'matmul.tc').write_text(Kernel.from_edsl(batched_matmul).dsl)
        fixed=specialize_kernel(batched_matmul,{'B':1,'M':128,'K':64,'N':128})
        if not isinstance(fixed,Kernel):fixed=Kernel.from_edsl(fixed)
        (ROOT/'matmul-fixed.tc').write_text(fixed.dsl)
    else:run(a.action)
