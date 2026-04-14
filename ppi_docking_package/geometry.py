import torch

def rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """
    Compute a single or batched rotation matrix from Euler angles.
    angles: [3] for single, or [B,3] for batch
    returns: [3,3] or [B,3,3]
    """
    if angles.ndim == 1:
        rx, ry, rz = angles.unbind()
        cx, sx = rx.cos(), rx.sin()
        cy, sy = ry.cos(), ry.sin()
        cz, sz = rz.cos(), rz.sin()
        Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=angles.device, dtype=angles.dtype)
        Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=angles.device, dtype=angles.dtype)
        Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=angles.device, dtype=angles.dtype)
        return Rz @ Ry @ Rx
    else:
        B = angles.size(0)
        rx, ry, rz = angles[:,0], angles[:,1], angles[:,2]
        cx, sx = rx.cos(), rx.sin()
        cy, sy = ry.cos(), ry.sin()
        cz, sz = rz.cos(), rz.sin()
        Rx = torch.zeros(B,3,3,device=angles.device)
        Rx[:,0,0] = 1
        Rx[:,1,1] = cx; Rx[:,1,2] = -sx
        Rx[:,2,1] = sx; Rx[:,2,2] = cx
        Ry = torch.zeros(B,3,3,device=angles.device)
        Ry[:,0,0] = cy; Ry[:,0,2] = sy
        Ry[:,1,1] = 1
        Ry[:,2,0] = -sy; Ry[:,2,2] = cy
        Rz = torch.zeros(B,3,3,device=angles.device)
        Rz[:,0,0] = cz; Rz[:,0,1] = -sz
        Rz[:,1,0] = sz; Rz[:,1,1] = cz
        Rz[:,2,2] = 1
        return torch.bmm(Rz, torch.bmm(Ry, Rx))

def apply_transformation(coords: torch.Tensor, translation: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation + translation to coords.
    translation: [3] for single, or [B,3] for batch
    angles: [3] single, or [B,3] batch
    """
    if angles.ndim == 1:
        R = rotation_matrix(angles)
        return coords @ R.T + translation
    else:
        B = angles.size(0)
        N = coords.size(0)
        coords_exp = coords.unsqueeze(0).expand(B,N,3)
        R = rotation_matrix(angles)
        return torch.bmm(coords_exp, R.transpose(1,2)) + translation.unsqueeze(1)

def get_com(coords: torch.Tensor) -> torch.Tensor:
    return coords.mean(dim=0)