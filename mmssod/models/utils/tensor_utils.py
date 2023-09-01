import torch
eps: float = 1e-14

def norm_tensor(data):
    # norm tensors to norm equal 1
    # input: N x dim
    # return: N x dim
    square_data = torch.square(data).sum(dim=1).sqrt().detach() + eps
    data_new = square_data.view(-1, 1).repeat(repeats=(1, data.shape[1]))
    # print(data_new)

    return torch.div(data, data_new)

def norm_tensor2(data):
    # norm tensors to norm equal 1
    # input: dim x N
    # return: dim x N
    square_data = torch.square(data).sum(dim=0).sqrt().detach() + eps
    data_new = square_data.view(1, -1).repeat(repeats=(data.shape[0], 1))
    return torch.div(data, data_new)


if __name__ == '__main__':
    from torch import nn
    aa = nn.functional.normalize(torch.rand((3,1024)), dim=0)
    print(norm_tensor(aa).transpose(0, 1))
    print(norm_tensor2(aa.transpose(0, 1)))