from lib import *


class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale# hệ số nhân vào weight bên trên
        self.reset_parameters()
        self.eps = 1e-10
    
    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)# nhân hệ số 20 vào tất cả các weight
    
    def forward(self, x):
        # x.size() = (batch_size, channel, height, width)
        # L2Norm
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps 
        #dim=1 :tính theo chiều dọc, tổng các phần tử channel, dim 0 là batchsize, dim 2 là height, dim 3 là width
        #keepdim =True là giữ lại các dim khác (batchsize,height,width) không tính tổng

        x = torch.div(x, norm) # chia x cho norm (vì vẫn là dạng tensor nên dùng torch)
        #weight.size() = (512) -> (1,512,1,1) # phải chuyển về cùng size vs x rồi mới nhân
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        return weights*x