import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix


## Wasserstein of two gaussians
class WassersteinGaussian(nn.Module):
    def __init__(self):
        super(WassersteinGaussian, self).__init__()

    def forward(self, loc1, scale1, rot_matrix1, loc2, scale2, rot_matrix2):
        """
        compute the Wasserstein distance between two Gaussians
        loc1, loc2: Bx3
        scale1, scale2: Bx3
        rot_matrix1, rot_matrix2: Bx3x3
        """
        
        loc_diff2 = torch.sum((loc1 - loc2)**2, dim=-1)

        ## Wasserstein distance Tr(C1 + C2 - 2(C1^0.5 * C2 * C1^0.5)^0.5)

        cov1_sqrt_diag = torch.sqrt(scale1).diag_embed() # Bx3x3

        cov2 = torch.bmm(rot_matrix2, torch.bmm(torch.diag_embed(scale2), rot_matrix2.transpose(1, 2))) # covariance matrix Bx3x3
        cov2_R1 = torch.bmm(rot_matrix1.transpose(1, 2), cov2).matmul(rot_matrix1) # Bx3x3
        # E = cv1^0.5*cv2*cv1^0.5

        E = torch.bmm(torch.bmm(cov1_sqrt_diag, cov2_R1), cov1_sqrt_diag) # Bx3x3

        E = (E + E.transpose(1, 2))/2
        E_eign = torch.linalg.eigvalsh(E)


        E_sqrt_trace = (E_eign.pow(2).pow(1/4)).sum(dim=-1)

        CovWasserstein = scale1.sum(dim=-1) + scale2.sum(dim=-1) - 2*E_sqrt_trace
        
        CovWasserstein = torch.clamp(CovWasserstein, min=0) # numerical stability for small negative values

        return torch.sqrt(loc_diff2 + CovWasserstein)
    

# example
if __name__ == "__main__":

    B = 6 # batch size
    loc = torch.randn(B, 3) # location Bx3
    rot = torch.randn(B, 4) # quaternion Bx4
    rot = F.normalize(rot, p=2, dim=1) # normalize quaternion
    scale = torch.randn(B, 3) # scale Bx3
    scale = torch.exp(scale) # make sure scale is positive

    # convert quaternion to rotation matrix
    rot_matrix = quaternion_to_matrix(rot) # rotation matrix Bx3x3
    cov = torch.bmm(rot_matrix, torch.bmm(torch.diag_embed(scale), rot_matrix.transpose(1, 2))) # covariance matrix Bx3x3

    wasserstein = WassersteinGaussian()
    wasserstein(loc[:3], scale[:3], rot_matrix[:3], loc[3:], scale[3:], rot_matrix[3:])