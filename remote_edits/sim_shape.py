import torch
import torch_npu
import os
torch.npu.set_device(int(os.environ.get('ASCEND_DEVICE_ID', '0')))
import custom_ops_lib
B = int(os.environ.get('SIM_B', '560'))
H = int(os.environ.get('SIM_H', '1024'))
g = torch.Generator(device='cpu')
g.manual_seed(7)
x = (torch.rand(B, H, generator=g) * 2 - 1).to(torch.float16)
r = (torch.rand(B, H, generator=g) * 2 - 1).to(torch.float16)
w = (torch.rand(H, generator=g) * 2.0).clamp(min=0.01).to(torch.float16)
y, res = custom_ops_lib.fused_add_rmsnorm(x.npu(), r.npu(), w.npu(), 1e-6)
torch.npu.synchronize()
print(f'[sim_shape] launched {B}x{H}')
