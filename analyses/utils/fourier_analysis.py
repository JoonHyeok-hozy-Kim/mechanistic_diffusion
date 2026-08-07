import einops
import torch
import numpy as np
import pandas as pd
import random
import os


def get_fourier_basis_and_names(p, latex=False):    
    fourier_basis = []
    fourier_basis.append(torch.ones(p)/np.sqrt(p))    # Constant term for 2-degree Fourier Basis
    fourier_basis_names = ['C']
    
    for freq in range(1, p//2 +1):  # k : 1 ~ P
        fourier_basis.append(torch.cos(2 * freq * torch.pi / p * torch.arange(p)))    # cos(2k(pi)/P * a)
        fourier_basis.append(torch.sin(2 * freq * torch.pi / p * torch.arange(p)))    # sin(2k(pi)/P * a)
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        if latex:
            fourier_basis_names.append(f'\\cos {freq}')
            fourier_basis_names.append(f'\\sin {freq}')
        else:
            fourier_basis_names.append(f'cos_{freq}')
            fourier_basis_names.append(f'sin_{freq}')
    fourier_basis = torch.stack(fourier_basis, dim=0).to('cuda')    # (1+2*num_freqs, P)
    
    return fourier_basis, fourier_basis_names


def get_d2_indices_by_frequency(p, k, latex=False):
    assert k < p, f"Frequency {k} greater than or equal to the basis P {p}!"
    
    cos_k = 2*k - 1     # 1, 3, 5, ..., 2*num_freqs-1 (k starts from 1)
    sin_k = 2*k         # 2, 4, 6, ..., 2*num_freqs   (k starts from 1)
    
    if latex:
        cos_k_str = f"\\cos(w_{{{k}}}"
        sin_k_str = f"\\sin(w_{{{k}}}"
    else:
        cos_k_str = f"cos(w_{k}"
        sin_k_str = f"sin(w_{k}"
    indices_and_desc = [
        # Degree One Bases
        (cos_k * p + 0  , f"{cos_k_str} a)"),
        (sin_k * p + 0  , f"{sin_k_str} a)"),
        (cos_k          , f"{cos_k_str} b)"),
        (sin_k          , f"{sin_k_str} b)"),
        # Degree Two Bases
        ((cos_k * p) + cos_k, f"{cos_k_str} a) {cos_k_str} b)"),
        ((cos_k * p) + sin_k, f"{cos_k_str} a) {sin_k_str} b)"),
        ((sin_k * p) + cos_k, f"{sin_k_str} a) {cos_k_str} b)"),
        ((sin_k * p) + sin_k, f"{sin_k_str} a) {sin_k_str} b)"),
    ]
    return indices_and_desc


def explain_trigonometric_basis(fourier_basis, fourier_basis_names, p):
    
    idx = 0
    k = 0
    B = []
    for i, basis in enumerate(fourier_basis):
        for j, basis in enumerate(fourier_basis):
            # print(f"idx={idx} : ({i}, {j}) - {fourier_basis_names[i]} * {fourier_basis_names[j]}")
            # B.append(f"idx={idx} : ({i}, {j}) - ({fourier_basis_names[i]} a) * ({fourier_basis_names[j]} b)")
            B.append(f"({fourier_basis_names[i]} a) * ({fourier_basis_names[j]} b)")
            idx += 1
            
    for freq in range(1, p//2+1):
        print(f"\n======< k={freq} Expected vs Real >======")
        for idx, desc in get_d2_indices_by_frequency(freq, p):
            print(f"{desc}  vs {B[idx]}")


def get_indices_all_frequencies(p, latex=False):
    indices, descs = [], []
    for k in range(1, p//2+1):
        k_indices, k_descs = [], []
        for idx, desc in get_d2_indices_by_frequency(p, k, latex=latex):
            k_indices.append(idx)
            k_descs.append(desc.replace("k", str(k)))
        indices.append(k_indices)
        descs.append(k_descs)
    indices = np.array(indices)
    descs = np.array(descs)
    return indices, descs


def two_degree_fourier_projection(activation, p, fourier_basis_d1):
    original_shape = activation.shape
    activation = einops.rearrange(activation, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_coeffs = torch.einsum('xyz,fx,Fy->fFz', activation, fourier_basis_d1, fourier_basis_d1)
    return fourier_coeffs.reshape(original_shape)


def get_FFN_hidden_dim(model):
    ffn = model.layers[0].feed_forward
    hidden_dim = ffn.w1.out_features
    return hidden_dim


def unflatten_first(tensor, p):
    if tensor.shape[0]==p*p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else: 
        return tensor
   

def cos(x, y):
    return (x.dot(y))/x.norm()/y.norm()


def mod_div(a, b, p):
    return (a*pow(b, p-2, p))%p


def normalize(tensor, axis=0):
    return tensor/(tensor).pow(2).sum(keepdim=True, axis=axis).sqrt()


def extract_freq_2d(tensor, freq, p):
    # Takes in a pxpx... or batch x ... tensor, returns a 3x3x... tensor of the 
    # Linear and quadratic terms of frequency freq
    tensor = unflatten_first(tensor, p)
    # Extracts the linear and quadratic terms corresponding to frequency freq
    index_1d = [0, 2*freq-1, 2*freq]
    # Some dumb manipulation to use fancy array indexing rules
    # Gets the rows and columns in index_1d
    return tensor[[[i]*3 for i in index_1d], [index_1d]*3]


def get_cov(tensor, norm=True):
    # Calculate covariance matrix
    if norm:
        tensor = normalize(tensor, axis=1)
    return tensor @ tensor.T


def is_close(a, b):
    return ((a-b).pow(2).sum()/(a.pow(2).sum().sqrt())/(b.pow(2).sum().sqrt())).item()


def fft1d(tensor, fourier_basis):
    # Converts a tensor with dimension p into the Fourier basis
    return tensor @ fourier_basis.T

def fourier_2d_basis_term(x_index, y_index, fourier_basis):
    # Returns the 2D Fourier basis term corresponding to the outer product of 
    # the x_index th component in the x direction and y_index th component in the 
    # y direction
    # Returns a 1D vector of length p^2
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()

def fft2d(mat, p, fourier_basis):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
    return fourier_mat.reshape(shape)

def analyse_fourier_2d(tensor, p, fourier_basis_names, top_k=10):
    # Processes a (p,p) or (p*p) tensor in the 2D Fourier Basis, showing the 
    # top_k terms and how large a fraction of the variance they explain
    values, indices = tensor.flatten().pow(2).sort(descending=True)
    rows = []
    total = values.sum().item()
    for i in range(top_k):
        rows.append([tensor.flatten()[indices[i]].item(),
                     values[i].item()/total, 
                     values[:i+1].sum().item()/total, 
                     fourier_basis_names[indices[i].item()//p], 
                     fourier_basis_names[indices[i]%p]])
    
    # display(pd.DataFrame(rows, columns=['Coefficient', 'Frac explained', 'Cumulative frac explained', 'x', 'y']))
    return pd.DataFrame(rows, columns=['Coefficient', 'Frac explained', 'Cumulative frac explained', 'x', 'y'])


def get_2d_fourier_component(tensor, x, y):
    # Takes in a batch x ... tensor and projects it onto the 2D Fourier Component 
    # (x, y)
    vec = fourier_2d_basis_term(x, y).flatten()
    return vec[:, None] @ (vec[None, :] @ tensor)


def get_component_cos_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to cos(freq*(x+y)) in the 2D Fourier basis
    # This is equivalent to the matrix cos((x+y)*freq*2pi/p)
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq).flatten()
    # Divide by sqrt(2) to ensure it remains normalised
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction)/np.sqrt(2)
    # Collapse_dim says whether to project back into R^(p*p) space or not
    if collapse_dim:
        return (cos_xpy_direction @ tensor)
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)


def get_component_sin_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to sin((x+y)*freq*2pi/p) in the 2D Fourier basis
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (sin_xpy_direction @ tensor)
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)
   

def to_numpy(tensor, flat=False):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        # if isinstance(tensor[0])
        return np.array(tensor)
    elif isinstance(tensor, torch.Tensor):
        if flat:
            return tensor.flatten().detach().cpu().numpy()
        else:
            return tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")