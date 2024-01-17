import torch

class SingularValueGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hess_matrix): #U, S, V):
        #print('custom forward')
        #print(S, flush=True)
        with torch.enable_grad():
            hess_ = hess_matrix.clone().detach().requires_grad_()
            #U_ = U.clone().detach().requires_grad_()
            #S_ = S.clone().detach().requires_grad_()
            #V_ = V.clone().detach().requires_grad_()
            U_, S_, V_ = torch.linalg.svd(hess_)
            out = torch.sum(S_, dim=1)

        ctx.saved_input = [hess_, U_, S_, V_]
        ctx.save_for_backward(out)
        out_ = out.detach()
        return out_

    @staticmethod
    def backward(ctx, grad_output):
        #return 0
        #print("in back", flush=True)
        #print(grad_output.shape)
        #print(grad_output, flush=True)
        out_, = ctx.saved_tensors
        hess_, U_, S_, V_ = ctx.saved_input
        #print(U_[0]@V_[0].t())
        #with torch.enable_grad():
        #    S_.backward(grad_output)
        #return ((torch.matmul(U_, torch.transpose(V_,2,1)))*grad_output.view(-1,1,1))/(weight+1e-7)
        gradout = torch.matmul(U_, torch.transpose(V_,2,1))
        gradout = gradout*grad_output.view(-1,1,1)
        return gradout
        #return (U_@V_.t())
        #print("hi", flush=True)
        #print(grad_output.shape)
        #print(torch.matmul(U, torch.tranpose(V,2,1)).shape)
        #return grad_output*torch.matmul(U, torch.tranpose(V,2,1))
        #return U@V.t()
