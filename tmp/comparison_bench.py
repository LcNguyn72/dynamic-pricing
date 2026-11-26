import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from experiments.song_forecast import SongTransformer
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=True
print('bf16 support',torch.cuda.is_bf16_supported(),flush=True)
model=SongTransformer().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
for batch,mixed in [(512,False),(2048,True),(4096,True)]:
    h=torch.randn(batch,12,9,device='cuda');c=torch.randn(batch,8,device='cuda');y=torch.randn(batch,device='cuda')
    torch.cuda.synchronize();started=time.perf_counter()
    for i in range(8):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            loss=torch.nn.functional.mse_loss(model(h,c),y)
        loss.backward();optimizer.step()
    torch.cuda.synchronize()
    print({'batch':batch,'mixed':mixed,'seconds_per_step':(time.perf_counter()-started)/8,
           'peak_MiB':torch.cuda.max_memory_allocated()/1024**2},flush=True)
    torch.cuda.reset_peak_memory_stats()
