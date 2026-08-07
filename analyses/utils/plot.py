import os

import torch
import torchvision.utils as tvu

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from PIL import Image

PALLETE = ['#51F0C4', '#F0C651', '#52C6F0', '#688D9B', '#F0A351', '#3E7566'] 


def draw_fve_bar_graph_one_degree_neuron_level(fourier_coeffs_d1, num_freqs, out_dir, file_name=None, title=None):
    colors = []
    fves_to_plot = []
    powers = fourier_coeffs_d1.pow(2).sum(dim=1)[1:]  # Exclude the constant term
    total_powers = powers.sum(dim=0, keepdim=True) + 1e-10
    fves_by_freqs = (powers / total_powers).detach().cpu().numpy() 
    
    freq_names = []
    for k in range(1, num_freqs + 1):
        freq_names.append(fr'$\cos(w_{{{k}}}x)$')
        freq_names.append(fr'$\sin(w_{{{k}}}x)$')
    
    for k in range(1, num_freqs+1):    # k : 1 ~ P
        fves_to_plot.append(fves_by_freqs[2*k - 2])    # cos
        fves_to_plot.append(fves_by_freqs[2*k - 1])    # sin            
        
        color_idx = (k) % len(PALLETE)
        target_color = PALLETE[color_idx]
        colors.extend([target_color, target_color]) # Same color to the pair (cos, sin)

    plt.figure(figsize=(15, 6))
    x_pos = np.arange(len(fves_to_plot))
    
    plt.bar(x_pos, fves_to_plot, color=colors, edgecolor='white', linewidth=0.5)

    plt.xlabel(fr"Frequency $w_{{k}}$ (Each pair has the same color)")
    plt.ylabel("Norm of Fourier Component")
    if title is None:
        title = fr"Fourier Spectrum Grouped by Frequency $w_{{k}}$"
    plt.title(title)

    tick_pos = [] 
    tick_labels = []
    
    for k in range(1, num_freqs+1): # k : 1 ~ P
        tick_pos.append(2*k - 1.5) 
        tick_labels.append(fr"$w_{{{k}}}$")
        
    plt.xticks(tick_pos, tick_labels)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    if file_name is not None:
        file_name = f"{file_name}.png"
    else:
        file_name = f"bar_graph_d1_neuron_level.png"
    file_path = os.path.join(out_dir, file_name)
    plt.savefig(file_path)
    plt.close()


def draw_fve_bar_graph_one_degree_head_level(fourier_coeffs_d1, num_freqs, num_heads, out_dir, file_name=None, title=None):
    powers = fourier_coeffs_d1.pow(2).sum(dim=2)[1:]        # (2*num_freqs, num_heads) : Sum neurons and group by heads. Exclude the constant basis.
    total_powers = powers.sum(dim=0, keepdim=True) + 1e-20  # (1, num_heads)
    fves_by_freqs_heads = (powers / total_powers).detach().cpu().numpy() 
    max_fve = max(0.5, fves_by_freqs_heads.max())
    # max_fve = fves_by_freqs_heads.max()
    
    n_plots = num_heads 
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
    axes = axes.flatten()  

    for head in range(n_rows * n_cols):
        ax = axes[head]
        
        if head < n_plots:
            fves_to_plot = [] 
            current_colors = [] 
            
            for k in range(1, num_freqs + 1):
                fves_to_plot.append(fves_by_freqs_heads[2*k-2, head].item()) # cos
                fves_to_plot.append(fves_by_freqs_heads[2*k-1, head].item()) # sin
                
                target_color = PALLETE[(k-1) % len(PALLETE)]
                current_colors.extend([target_color, target_color])
            
            x_pos = np.arange(len(fves_to_plot))
            
            ax.bar(x_pos, fves_to_plot, color=current_colors, edgecolor='white', linewidth=0.5)
            ax.set_ylim([0, max_fve * 1.1])

            tick_pos = np.arange(num_freqs) * 2 + 0.5 
            tick_labels = [fr"$w_{{{k}}}$" for k in range(1, num_freqs + 1)]
            
            ax.set_xticks(tick_pos) # ax.xticks -> ax.set_xticks
            ax.set_xticklabels(tick_labels)
            ax.set_title(f"Head {head}") 
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax.axis('off')

    plt.tight_layout()
    if file_name is None:
        file_path = os.path.join(out_dir, f"fve_bar_graph_by_freq_head_level_one_degree.png")
    else:
        file_path = os.path.join(out_dir, f"{file_name}.png")
    plt.savefig(file_path)
    plt.close()
    



def draw_fve_bar_graph_by_freq_head_level_one_degree(fourier_norms, num_freqs, num_heads, save_dir, file_name=None): 
    max_norm = max(0.5, fourier_norms.max())
    
    n_plots = num_heads 
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
    axes = axes.flatten()  

    for head in range(n_rows * n_cols):
        ax = axes[head]
        
        if head < n_plots:
            norms_to_plot = [] 
            current_colors = [] 
            
            for k in range(1, num_freqs + 1):
                norms_to_plot.append(fourier_norms[2*k - 1, head].item()) # cos
                norms_to_plot.append(fourier_norms[2*k, head].item())     # sin
                
                target_color = PALLETE[(k) % len(PALLETE)]
                current_colors.extend([target_color, target_color])
            
            x_pos = np.arange(len(norms_to_plot))
            
            ax.bar(x_pos, norms_to_plot, color=current_colors, edgecolor='white', linewidth=0.5)
            ax.set_ylim([0, max_norm * 1.1])

            tick_pos = np.arange(num_freqs) * 2 + 0.5 
            tick_labels = [fr"$w_{{{k}}}$" for k in range(1, num_freqs + 1)]
            
            ax.set_xticks(tick_pos) # ax.xticks -> ax.set_xticks
            ax.set_xticklabels(tick_labels)
            ax.set_title(f"Head {head}") 
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax.axis('off')

    plt.tight_layout()
    if file_name is None:
        file_path = os.path.join(save_dir, f"fve_bar_graph_by_freq_head_level_one_degree.png")
    else:
        file_path = os.path.join(save_dir, f"{file_name}.png")
    plt.savefig(file_path)
    plt.close()
        

def draw_fve_bar_graph_by_freq_head_level_two_degree(fve_by_heads, num_freqs, num_heads, save_dir, file_name=None):                    
    # FVE Bar Plots of each head
    max_norm = max(0.5, fve_by_heads.max().item())                
    n_plots = num_heads 
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), squeeze=False)
    axes = axes.flatten()  

    for head in range(n_rows * n_cols):
        ax = axes[head]
        
        if head < n_plots:
            fves_to_plot = [] 
            current_colors = [] 
            
            for freq in range(num_freqs):
                fves_to_plot.append(fve_by_heads[freq, head].item())
                
                target_color = PALLETE[(freq+1) % len(PALLETE)]
                current_colors.append(target_color)
            
            x_pos = np.arange(len(fves_to_plot))
            
            ax.bar(x_pos, fves_to_plot, color=current_colors, edgecolor='white', linewidth=0.5)
            ax.set_ylim([0, max_norm * 1.1])

            tick_pos = x_pos
            tick_labels = [fr"$w_{{{k}}}$" for k in range(1, num_freqs + 1)]
            
            ax.set_xticks(tick_pos) # ax.xticks -> ax.set_xticks
            ax.set_xticklabels(tick_labels)
            ax.set_title(f"Head {head}") 
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax.axis('off')

    plt.tight_layout()
    if file_name is None:
        file_path = os.path.join(save_dir, f"fve_bar_graph_by_freq_head_level_two_degree.png")
    else:
        file_path = os.path.join(save_dir, f"{file_name}.png")
    plt.savefig(file_path)
    plt.close()


def plot_sinlge_promient_neuron_per_freq(
    p, operand, fourier_coeffs, fourier_basis_names, curr_activation, out_dir, file_name=None, title=None
):
    powers = fourier_coeffs.pow(2)
    total_powers = powers.sum(dim=0, keepdim=True) + 1e-10
    normalized = powers / total_powers    
    representative_neurons_idx = normalized.argmax(dim=1)
    representative_patterns = curr_activation[:, representative_neurons_idx]
    
    n_plots = p
    n_rows = 4
    n_cols = n_plots // n_rows
    if n_plots % n_rows != 0:
        n_cols += 1
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    axes = axes.flatten()  

    for k in range(n_rows * n_cols):
        ax = axes[k]
        
        if k < n_plots:
            neuron_idx = representative_neurons_idx[k].item()
            purity = normalized[k, neuron_idx].item() * 100
            pattern = representative_patterns[:, k].cpu().numpy()
            
            ax.plot(range(p), pattern, marker='o', color='red', markersize=4, linestyle='None')
            if title is not None:
                ax.set_title(f"{title}", fontsize=10)   
            else:
                ax.set_title(f"{fourier_basis_names[k]}\nN{neuron_idx} ({purity:.1f}%)", fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        if k >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Input a")

    plt.tight_layout()
    if file_name is None:
        file_path = os.path.join(out_dir, f"plot_sinlge_promient_neuron_per_freq(operand {operand}).png")
    else:
        file_path = os.path.join(out_dir, f"{file_name}.png")
    plt.savefig(file_path)
    plt.close()


def draw_one_degree_heatmap(fourier_coeffs, num_freqs, out_dir, file_name=None, title=None):
    powers = fourier_coeffs.pow(2)[1:]  # Exclude the constant term
    total_powers = powers.sum(dim=1, keepdim=True) + 1e-10
    normalized = (powers / total_powers).detach().cpu().numpy() 
    total_neurons = normalized.shape[1]
    
    freq_names = []
    for k in range(1, num_freqs + 1):
        freq_names.append(fr'$\cos(w_{{{k}}}x)$')
        freq_names.append(fr'$\sin(w_{{{k}}}x)$')
        
    fig, (ax_main, ax_text) = plt.subplots(1, 2, figsize=(28, 10), gridspec_kw={'width_ratios': [4, 1]})
    
    x_ticks = np.linspace(0, total_neurons - 1, 9, dtype=int)
    ax_main.set_xticks(x_ticks + 0.5) 
    ax_main.set_xticklabels(x_ticks, rotation=0, ha='center')
    
    sns.heatmap(normalized, 
                cmap='viridis', 
                xticklabels=64, 
                yticklabels=freq_names,
                ax=ax_main,
                cbar_kws={'label': 'Normalized Power', 'location': 'left', 'pad': 0.1})

    for i in range(2, len(freq_names), 2):
        ax_main.axhline(i, color='white', linestyle='--', linewidth=0.8, alpha=0.8)

    ax_text.axis('off')
    
    if title is not None:
        ax_main.set_title(title, fontsize=18)
    else:
        ax_main.set_title(f"Neuron Specialization", fontsize=18)
        
    ax_main.set_xlabel("Neuron Index")
    ax_main.set_ylabel("Fourier Components")
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.tight_layout()
    if file_name is not None:
        file_path = os.path.join(out_dir, f"{file_name}.png")
    else:
        file_path = os.path.join(out_dir, f"heatmap_by_neuron.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


def draw_one_degree_heatmap_head_level(heatmap_data, num_freqs, num_heads, head_dim, save_dir):
    plt.figure(figsize=(25, 10))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='upper')
    plt.colorbar(label='Fourier Coefficient Norm')

    y_ticks = np.arange(num_freqs)
    y_labels = []
    num_ks = num_freqs // 2

    for k in range(1, num_ks + 1):
        y_labels.append(fr"$\cos(w_{{{k}}}x)$")
        y_labels.append(fr"$\sin(w_{{{k}}}x)$")


    plt.yticks(y_ticks, y_labels)
    plt.ylabel('Frequency Component')

    total_neurons = num_heads * head_dim
    x_ticks = []
    x_labels = []
    head_boundaries = [] 

    for h_idx in range(num_heads):
        start_idx = h_idx * head_dim
        end_idx = (h_idx + 1) * head_dim - 1
        
        x_ticks.append(start_idx + head_dim // 2) 
        x_labels.append(f"Head {h_idx}")
        
        if h_idx > 0:
            head_boundaries.append(start_idx - 0.5) 

    plt.xticks(x_ticks, x_labels, ha='center')
    plt.xlabel('Head and Neuron Index')

    for boundary in head_boundaries:
        plt.axvline(x=boundary, color='white', linestyle='--', linewidth=0.5)

    plt.title('Fourier Coefficient Norms per Frequency, Head, and Neuron')
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"graph_{2}_fourier_norm_heatmap.png")
    plt.savefig(file_path)
    plt.close() 


def draw_neuron_activation_approx_sin_trajectory(
    curr_activation, high_neurons, k, freq_coeffs_pair, freq_names_pair, out_dir
):
    p = curr_activation.shape[0]
    n_plots = len(high_neurons)
    n_cols = 1
    n_rows = n_plots // n_cols
    if n_plots % n_cols != 0:
        n_rows += 1
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows), squeeze=False)
    axes = axes.flatten()  

    for i in range(n_rows * n_cols):
        ax = axes[i]
        
        if i < n_plots:
            # Plot neuron activation
            neuron_idx = high_neurons[i]
            neuron_activation = curr_activation[:, neuron_idx].cpu().numpy()
            neuron_label = "Neuron Activation"
            ax.plot(range(p), neuron_activation, label=neuron_label, marker='o', color='red', markersize=4, linestyle='None')
            
            # Cos/Sin function trajectory
            cos_coeff = freq_coeffs_pair[0][neuron_idx].cpu().numpy()
            sin_coeff = freq_coeffs_pair[1][neuron_idx].cpu().numpy()
            x = np.linspace(0, p-1, 1000)
            
            # Cos funtion
            cos = cos_coeff * np.cos(k * 2 * np.pi * x / p)
            cos_label = fr"${cos_coeff:.2f}\cos(w_{{{k}}}x)$"
            ax.plot(x, cos, label=cos_label, linestyle='--', color='green', alpha=0.6, linewidth=1)  
            
            # Sin function
            sin = sin_coeff * np.sin(k * 2 * np.pi * x / p)
            sin_label = fr"${sin_coeff:.2f}\sin(w_{{{k}}}x)$"
            ax.plot(x, sin, label=sin_label, linestyle='--', color='blue', alpha=0.6, linewidth=1)  
            
            # Approximated Sin function
            approx_sin = cos + sin            
            approx_width = np.sqrt(cos_coeff**2 + sin_coeff**2)
            approx_sin_label = fr"${approx_width:.2f}\sin(w_{{{k}}}x + \phi)$"
            ax.plot(x, approx_sin, label=approx_sin_label, color='blue', linewidth=2)         
            ax.set_title(
                cos_label + "+" + sin_label + "=" + approx_sin_label + f"\nNeuron : {neuron_idx}",
                fontsize=10
            )
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

    plt.tight_layout()
    file_path = os.path.join(out_dir, f"neuron_activation_approx_sin_trajectory_k_{k}.png")
    plt.savefig(file_path)
    plt.close()


def draw_two_degree_heatmap(
    fourier_coeffs_2d, num_freqs, num_2d_bases, two_degree_idxes, two_degree_idxes_descs, save_dir, 
    num_heads=None, head_dim=None, file_name=None, title=None
):
    if num_heads is not None:
        assert head_dim is not None, f"head_dim not provided."        
        fourier_coeffs_2d = fourier_coeffs_2d.reshape(fourier_coeffs_2d.shape[0], num_heads*head_dim)
    
    curr_dim = fourier_coeffs_2d.shape[1]
    
    # Filter 2-Degree bases sharing same frequencies
    target_fourier_coeffs_2d = fourier_coeffs_2d[two_degree_idxes]  # (num_freqs, num_2d_bases, curr_dim)
    
    # FVE investigation\
    total_powers_by_neurons = fourier_coeffs_2d[1:].pow(2).sum(0) + 1e-9          # (curr_dim) Sum excluding the const basis
    # Heat map                
    target_coeffs_by_neurons = target_fourier_coeffs_2d.reshape(    # (num_freqs*num_2d_bases, curr_dim)
            num_freqs * num_2d_bases, curr_dim
        ).pow(2)
    FVEs_by_neurons_all_bases = target_coeffs_by_neurons / total_powers_by_neurons.unsqueeze(0)
    FVEs_by_neurons_all_bases = FVEs_by_neurons_all_bases.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(25, 12)) 
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    
    im = ax.imshow(FVEs_by_neurons_all_bases, cmap='viridis', aspect='auto', origin='upper')        

    # --- Y-axis Left ---
    left_ticks = np.arange(num_freqs) * num_2d_bases + (num_2d_bases // 2)
    left_labels = [fr"$w_{{{k+1}}}$" for k in range(num_freqs)]

    ax.set_yticks(left_ticks)
    ax.set_yticklabels(left_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequencies', fontsize=14)
    
    # --- Y-axis Right ---
    ax2 = ax.twinx() 
    ax2.set_ylim(ax.get_ylim()) 

    right_ticks = np.arange(num_freqs * num_2d_bases)
    right_labels = []
    for freq_bases in two_degree_idxes_descs:
        right_labels.extend(freq_bases)
        
    ax2.set_yticks(right_ticks)
    ax2.set_yticklabels(right_labels, fontsize=7)
    ax2.set_ylabel('Individual Basis Components', fontsize=12)
    
    for k in range(num_freqs):
        if k > 0:
            ax.axhline(y=k * num_2d_bases - 0.5, color='white', linestyle='-', linewidth=1.2, alpha=0.8)
        
        mid_boundary = k * num_2d_bases + 3.5
        ax.axhline(y=mid_boundary, color='white', linestyle=':', linewidth=0.8, alpha=0.4)

    # --- X-axis ---
    if num_heads is None:
        x_ticks = np.linspace(0, curr_dim - 1, 8, dtype=int)
        x_labels = [str(idx) for idx in x_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlabel('Neuron Index', fontsize=12)
        ax.set_xlim(-0.5, curr_dim - 0.5)
    else:
        x_ticks = []
        x_labels = []
        head_boundaries = [] # head별 유격을 위한 구분선

        for h_idx in range(num_heads):
            start_idx = h_idx * head_dim
            end_idx = (h_idx + 1) * head_dim - 1
            
            x_ticks.append(start_idx + head_dim // 2) 
            x_labels.append(f"Head {h_idx}")
            
            if h_idx > 0:
                head_boundaries.append(start_idx - 0.5) 

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, ha='center')
        ax.set_xlabel('Head and Neuron Index')


        for boundary in head_boundaries:
            ax.axvline(x=boundary, color='white', linestyle='--', linewidth=0.5)

    color_bar = fig.colorbar(im, ax=ax, location='left', pad=0.04, fraction=0.03)
    color_bar.set_label('Fourier Coefficient Norm', fontsize=12)
    if title is None:
        title = 'Fourier Coefficient Norms per Frequency, Head, and Neuron'
    plt.title(title)
    
    if file_name is not None:
        file_path = os.path.join(save_dir, f"{file_name}.png")  
    else:
        file_path = os.path.join(save_dir, f"graph_{2}_fourier_norm_heatmap.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close() 


def draw_ab_coeff_heatmap(p, k, heatmap_data, out_dir, cos=True):
    assert tuple(heatmap_data.shape) == (p, p), f"Invalid heat_map_data shape : {heatmap_data.shape}"
    
    plt.imshow(heatmap_data, cmap='RdBu', origin='lower') 
    plt.xlabel('Operand a')
    plt.ylabel('Operand b')
    if cos:
        title = fr'Projection onto $\cos u_{k}$'
        file_name = f"heatmap-k_{k}-cos-u.png"
    else:
        title = fr'Projection onto $\sin v_{k}$'
        file_name = f"heatmap-k_{k}-sin-v.png"
    plt.title(title)
    file_path = os.path.join(out_dir, file_name)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close() 


# Separate rendering the patch-level graphs and saving graphs on the grid    
def render_bar_graph_neuron_level_on_ax(ax, fourier_coeffs_d1, num_freqs, title=None):
    powers = fourier_coeffs_d1.pow(2).sum(dim=1)[1:]
    total_powers = powers.sum(dim=0, keepdim=True) + 1e-10
    fves_by_freqs = (powers / total_powers).detach().cpu().numpy() 
    
    fves_to_plot, colors = [], []
    for k in range(1, num_freqs + 1):
        fves_to_plot.extend([fves_by_freqs[2*k-2], fves_by_freqs[2*k-1]])
        target_color = PALLETE[(k-1) % len(PALLETE)]
        colors.extend([target_color, target_color])

    x_pos = np.arange(len(fves_to_plot))
    ax.bar(x_pos, fves_to_plot, color=colors, edgecolor='white', linewidth=0.3)
    if title: ax.set_title(title, fontsize=10)
    
    ax.set_xticks([2*k - 1.5 for k in range(1, num_freqs+1)])
    ax.set_xticklabels([f"$w_{{{k}}}$" for k in range(1, num_freqs+1)], fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
def render_bar_graph_head_level_on_ax(ax, fourier_coeffs_d1, num_freqs, title=None):
    powers = fourier_coeffs_d1.pow(2).sum(dim=1)[1:]
    total_powers = powers.sum(dim=0, keepdim=True) + 1e-10
    fves_by_freqs = (powers / total_powers).detach().cpu().numpy() 
    
    fves_to_plot, colors = [], []
    for k in range(1, num_freqs + 1):
        fves_to_plot.extend([fves_by_freqs[2*k-2], fves_by_freqs[2*k-1]])
        target_color = PALLETE[(k-1) % len(PALLETE)]
        colors.extend([target_color, target_color])

    x_pos = np.arange(len(fves_to_plot))
    ax.bar(x_pos, fves_to_plot, color=colors, edgecolor='white', linewidth=0.3)
    if title: ax.set_title(title, fontsize=10)
    
    ax.set_xticks([2*k - 1.5 for k in range(1, num_freqs+1)])
    ax.set_xticklabels([f"$w_{{{k}}}$" for k in range(1, num_freqs+1)], fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def render_bar_graph_two_degree_neuron_level_on_ax(ax, fourier_coeffs_2d, num_freqs, num_2d_bases, two_degree_idxes, two_degree_idxes_descs, title=None, max_fve_val=None):
    curr_dim = fourier_coeffs_2d.shape[1]
    target_coeffs = fourier_coeffs_2d[two_degree_idxes].reshape(-1, curr_dim).pow(2)
    total_powers = fourier_coeffs_2d[1:].pow(2).sum(0) + 1e-9
    fves = target_coeffs / total_powers.unsqueeze(0)
    fves_sum_by_dim = torch.mean(fves, dim=1).reshape(num_freqs, num_2d_bases)
    fves_1d = fves_sum_by_dim[:, :num_2d_bases//2].sum(1).cpu().numpy()
    fves_2d = fves_sum_by_dim[:, num_2d_bases//2:].sum(1).cpu().numpy()
    
    fves_to_plot, colors, x_labels = [], [], []
    for k in range(1, num_freqs+1):
        fves_to_plot.extend([fves_1d[k-1], fves_2d[k-1]])
        target_color = PALLETE[k % len(PALLETE)]
        colors.extend([target_color, target_color])
        x_labels.extend(["D1", "D2"])

    x_pos = np.arange(len(fves_to_plot))
    ax.bar(x_pos, fves_to_plot, color=colors, edgecolor='white', linewidth=0.3)
    
    if title: 
        ax.set_title(title, fontsize=12)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=5)
    
    for k in range(num_freqs):
        center_pos = 2 * k + 0.5
        ax.text(center_pos, -0.07, f"$w_{{{k+1}}}$", 
                transform=ax.get_xaxis_transform(), 
                ha='center', va='top', fontsize=9, fontweight='bold')

    if max_fve_val is None:
        max_fve_val = max(fves_to_plot) * 1.2 if len(fves_to_plot) > 0 else 1.0
    ax.set_ylim(0, max_fve_val)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def render_bar_graph_two_degree_head_level_on_ax(fig, ax, fourier_coeffs_2d, num_freqs, num_2d_bases, two_degree_idxes, two_degree_idxes_descs, title=None, max_fve_val=None):
    num_heads = fourier_coeffs_2d.shape[1]
    head_dim = fourier_coeffs_2d.shape[2]
    target_coeffs = fourier_coeffs_2d[two_degree_idxes].reshape(-1, num_heads, head_dim).pow(2)
    total_powers = fourier_coeffs_2d[1:].pow(2).sum(0) + 1e-9
    fves = target_coeffs / total_powers.unsqueeze(0)
    fves_sum_by_dim = torch.mean(fves, dim=2).reshape(num_freqs, num_2d_bases, num_heads)
    fves_1d = fves_sum_by_dim[:, :num_2d_bases//2].sum(1).cpu().numpy()
    fves_2d = fves_sum_by_dim[:, num_2d_bases//2:].sum(1).cpu().numpy()
    
    n_cols = 4
    n_rows = (num_heads + n_cols - 1) // n_cols
    
    ax.axis('off')
    if title: 
        ax.set_title(title, fontsize=12)
    
    sub_grid = ax.get_subplotspec().subgridspec(n_rows, n_cols)
    head_cnt = 0
    for rdx in range(n_rows):
        if head_cnt == num_heads:
            break
        for cdx in range(n_cols):
            if head_cnt == num_heads:
                break
            curr_fves_1d = fves_1d[:, head_cnt]
            curr_fves_2d = fves_2d[:, head_cnt]            
            curr_ax = fig.add_subplot(sub_grid[rdx, cdx])
            fves_to_plot, colors, x_labels = [], [], []
            for k in range(1, num_freqs+1):
                fves_to_plot.extend([curr_fves_1d[k-1], curr_fves_2d[k-1]])
                target_color = PALLETE[k % len(PALLETE)]
                colors.extend([target_color, target_color])
                x_labels.extend(["D1", "D2"])
    
            x_pos = np.arange(len(fves_to_plot))
            curr_ax.bar(x_pos, fves_to_plot, color=colors, edgecolor='white', linewidth=0.3)
            curr_ax.set_xticks(x_pos)
            curr_ax.set_xticklabels(x_labels, fontsize=5)
            
            for k in range(num_freqs):
                center_pos = 2 * k + 0.5
                curr_ax.text(center_pos, -0.07, f"$w_{{{k+1}}}$", 
                            transform=curr_ax.get_xaxis_transform(), 
                            ha='center', va='top', fontsize=9, fontweight='bold')
                        
            if max_fve_val is None:
                max_fve_val = max(fves_to_plot) * 1.2 if len(fves_to_plot) > 0 else 1.0
                        
            curr_ax.set_ylim(0, max_fve_val)
            curr_ax.grid(axis='y', alpha=0.3, linestyle='--')
            curr_ax.set_title(f"Head {head_cnt}", fontsize=9)
            
            head_cnt += 1    


def render_heatmap_neuron_level_on_ax(ax, fourier_coeffs_2d, num_freqs, num_2d_bases, two_degree_idxes, two_degree_idxes_descs, title=None):
    curr_dim = fourier_coeffs_2d.shape[1]
    target_coeffs = fourier_coeffs_2d[two_degree_idxes].reshape(-1, curr_dim).pow(2)
    total_powers = fourier_coeffs_2d[1:].pow(2).sum(0) + 1e-9
    fves = (target_coeffs / total_powers.unsqueeze(0)).cpu().numpy()
    
    im = ax.imshow(fves, cmap='viridis', aspect='auto', origin='upper')
    if title: ax.set_title(title, fontsize=12)

    # Y-axis Left: Frqeuency k (w_k)
    left_ticks = np.arange(num_freqs) * num_2d_bases + (num_2d_bases // 2) - 0.5
    ax.set_yticks(left_ticks)
    ax.set_yticklabels([fr"$w_{{{k+1}}}$" for k in range(num_freqs)], fontsize=10)

    # Y-axis Right: Two degree bases (cos*cos, ...)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    right_labels = []
    for group in two_degree_idxes_descs:
        right_labels.extend(group)
    
    ax2.set_yticks(np.arange(len(right_labels)))
    ax2.set_yticklabels(right_labels, fontsize=6)

    # for k in range(1, num_freqs):
    #     ax.axhline(y=k * num_2d_bases - 0.5, color='white', linestyle='-', linewidth=0.8, alpha=0.6)
        
    for k in range(num_freqs):
        for b in range(num_2d_bases):
            y_pos = k * num_2d_bases + b - 0.5
            
            if b == 0 and k > 0:
                ax.axhline(y=y_pos, color='white', linestyle='-', linewidth=0.3, alpha=0.6)
            if b == 4:
                ax.axhline(y=y_pos, color='white', linestyle='--', linewidth=0.2, alpha=0.4)
            elif b > 0:
                ax.axhline(y=y_pos, color='white', linestyle=':', linewidth=0.1, alpha=0.2)
            
    return im


def render_heatmap_head_level_on_ax(ax, fourier_coeffs_2d, num_freqs, num_2d_bases, two_degree_idxes, two_degree_idxes_descs, title=None):
    num_heads = fourier_coeffs_2d.shape[-2]
    head_dim = fourier_coeffs_2d.shape[-1]
    fourier_coeffs_2d = fourier_coeffs_2d.reshape(-1, num_heads*head_dim)
    target_coeffs = fourier_coeffs_2d[two_degree_idxes].reshape(-1, num_heads*head_dim).pow(2)
    total_powers = fourier_coeffs_2d[1:].pow(2).sum(0) + 1e-9
    fves = (target_coeffs / total_powers.unsqueeze(0)).cpu().numpy()
    total_dim = num_heads * head_dim
    
    im = ax.imshow(fves, cmap='viridis', aspect='auto', origin='upper')
    if title: ax.set_title(title, fontsize=12)

    # X-axis: Head labels and separators
    # ax.set_xticks(np.arange(num_heads) * head_dim + (head_dim / 2) - 0.5)
    # ax.set_xticklabels([f"head {i}" for i in range(num_heads)], fontsize=8, rotation=45)
    x_ticks = np.linspace(0, total_dim - 1, 10, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=8)
    
    for i in range(1, num_heads):
        ax.axvline(x=i * head_dim - 0.5, color='white', linestyle='-', linewidth=0.1)

    # Y-axis Left: Frqeuency k (w_k)
    left_ticks = np.arange(num_freqs) * num_2d_bases + (num_2d_bases // 2) - 0.5
    ax.set_yticks(left_ticks)
    ax.set_yticklabels([fr"$w_{{{k+1}}}$" for k in range(num_freqs)], fontsize=10)

    # Y-axis Right: Two degree bases (cos*cos, ...)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    right_labels = []
    for group in two_degree_idxes_descs:
        right_labels.extend(group)
    
    ax2.set_yticks(np.arange(len(right_labels)))
    ax2.set_yticklabels(right_labels, fontsize=6)

    # for k in range(1, num_freqs):
    #     ax.axhline(y=k * num_2d_bases - 0.5, color='white', linestyle='-', linewidth=0.8, alpha=0.6)
        
    for k in range(num_freqs):
        for b in range(num_2d_bases):
            y_pos = k * num_2d_bases + b - 0.5
            
            if b == 0 and k > 0:
                ax.axhline(y=y_pos, color='white', linestyle='-', linewidth=0.3, alpha=0.3)
            if b == 4:
                ax.axhline(y=y_pos, color='white', linestyle='--', linewidth=0.2, alpha=0.2)
            elif b > 0:
                ax.axhline(y=y_pos, color='white', linestyle=':', linewidth=0.1, alpha=0.1)
            
    return im

def save_patch_grid_plot_single_variable(grid_size, variable_data, num_freqs, out_dir, var_name, plot_type='bar', **kwargs):
    """
    variable_data: list of grid_size**2 patch tensors 
    """
    if plot_type.split('_')[0] == 'bar':
        if 'head' in plot_type.split('_'):
            fig_size = (16*grid_size, 16*grid_size)
        else:            
            fig_size = (8*grid_size, 6*grid_size)
    elif plot_type.split('_')[0] == 'heatmap':
        fig_size = (16*grid_size, 12*grid_size)
    else:
        raise NotImplemented(f"Unavailable plot_type provided : {plot_type}")
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=fig_size, constrained_layout=True, squeeze=False)
    
    for pdx in range(grid_size ** 2):
        ax = axes[pdx // grid_size, pdx % grid_size]
        curr_data = variable_data[pdx]
        
        if plot_type == 'bar_neuron_level':
            render_bar_graph_neuron_level_on_ax(ax, curr_data, num_freqs, title=f"Patch {pdx}")
        elif plot_type == 'bar_head_level':
            raise NotImplementedError("bar_head_level not implemented. Too many heads to visualize on a small patch grid?")
        elif plot_type == 'bar_two_degree_neuron_level':
            render_bar_graph_two_degree_neuron_level_on_ax(
                ax, curr_data, num_freqs, 
                kwargs['num_2d_bases'], kwargs['two_degree_idxes'], kwargs['two_degree_idxes_descs'], 
                title=f"Patch {pdx}"
            )
        elif plot_type == 'bar_two_degree_head_level':
            render_bar_graph_two_degree_head_level_on_ax(
                fig, ax, curr_data, num_freqs, 
                kwargs['num_2d_bases'], kwargs['two_degree_idxes'], kwargs['two_degree_idxes_descs'], 
                title=f"Patch {pdx}"
            )
        elif plot_type == 'heatmap_neuron_level':
            render_heatmap_neuron_level_on_ax(
                ax, curr_data, num_freqs, 
                kwargs['num_2d_bases'], kwargs['two_degree_idxes'], kwargs['two_degree_idxes_descs'], 
                title=f"Patch {pdx}"
            )
        elif plot_type == 'heatmap_head_level':
            render_heatmap_head_level_on_ax(
                ax, curr_data, num_freqs, 
                kwargs['num_2d_bases'], kwargs['two_degree_idxes'], kwargs['two_degree_idxes_descs'], 
                title=f"Patch {pdx}"
            )
        else:
            raise NotImplemented(f"Unavailable plot_type provided : {plot_type}")

    k_pdx = kwargs.get('k_pdx')
    if k_pdx is not None:
        title = f"Variable {var_name} Key Patch {k_pdx} - {grid_size}x{grid_size} Query Patch Grid Analysis"
        file_name = f"Grid_{plot_type}-var_{var_name}-key_patch_{k_pdx}"
    else:
        title = kwargs.get('title')
        if title is None:
            title = f"Variable {var_name} - {grid_size}x{grid_size} Patch Grid Analysis"
            
        file_name = kwargs.get('file_name')
        if file_name is None:
            file_name = f"Grid_{plot_type}_var_{var_name}"
        
    fig.suptitle(title, fontsize=16)
    file_path = os.path.join(out_dir, f"{file_name}.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    

def plot_pca_all_layers(layer_activations, labels, save_dir, time_step, var_name, title=None, plot_centroid=False):
    """
    layer_activations: {layer_name: activation_tensor (P^2, D)}
    """
    n_layers = len(layer_activations)
    cols = 7
    rows = (n_layers + cols - 1) // cols  # FFN-Sandwich : 32 layers -> 7 x 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
    axes = axes.flatten()
    
    labels_np = np.array(labels)
    unique_labels = np.unique([l for l in labels_np if l is not None])
    
    for i, (layer_name, act) in enumerate(layer_activations.items()):
        ax = axes[i]
        # PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(act.cpu().numpy())
        
        # Scatter plot batch data
        scatter_s = 5
        ax.scatter(reduced[:, 0], reduced[:, 1], c=labels_np, cmap='hsv', s=scatter_s, alpha=0.6)
        
        if plot_centroid:
            # Spot Centroids of each label
            norm = plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max())
            cmap = plt.get_cmap('hsv')
            centroid_scale = 4
            
            for u_label in unique_labels:
                mask = (labels_np == u_label)
                if not np.any(mask): continue
                
                centroid = reduced[mask].mean(axis=0)   # Mean of batches (multiple images)
                ax.scatter(centroid[0], centroid[1],
                        color=cmap(norm(u_label)),
                        s=scatter_s*pow(centroid_scale,2), marker='o',
                        edgecolors='black', linewidths=1.0,
                        alpha=1.0, zorder=5)
        
        ax.set_title(f"L{i}: {layer_name}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    if title is None:
        fig.suptitle(f"Variable: {var_name} - All Layers Evolution (Timestep: {time_step:02d})", fontsize=20)
    else:
        fig.suptitle(title)
    
    plt.savefig(save_dir, dpi=80)
    plt.close(fig)


def plot_fve_barchart_and_heatmap(
    p, x_src, x_tgt, correct_mask, low_confidence_mask, 
    act1_fourier_d2, act2_fourier_d2, 
    num_freqs, num_2d_bases, two_degree_idxes, two_degree_idxes_descs,
    save_dir, title=None, sub_bar1_title=None, sub_bar2_title=None,
):
    fig = plt.figure(figsize=(24, 12))
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.02)
            
    # Left Sample Image Grid
    ax_img = fig.add_subplot(outer_gs[0])
    n_cols = 8
    n_rows = p + 10
    num_vis = n_rows * n_cols 
    
    vis_image = torch.cat([x_src[:num_vis], x_tgt[:num_vis]], dim=3)
    vis_correct_mask = correct_mask[:num_vis]
    vis_low_confidence_mask = low_confidence_mask[:num_vis]
    rgb_image = vis_image.repeat(1, 3, 1, 1) # (B, 3, 32, 96)
    # rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    rgb_image = torch.clamp(rgb_image, 0.0, 1.0)
    img_device = rgb_image.device
    for i in range(len(vis_correct_mask)):
        if vis_low_confidence_mask[i]:
            color = torch.tensor([1.0, 1.0, 0.0], device=img_device) # Yellow
        elif vis_correct_mask[i]:
            color = torch.tensor([0.0, 1.0, 0.0], device=img_device) # Green
        else:
            color = torch.tensor([1.0, 0.0, 0.0], device=img_device) # Red
        
        c_view = color.view(3, 1, 1)
        rgb_image[i, :, :2, 64:] = c_view   # Top
        rgb_image[i, :, -2:, 64:] = c_view  # Bottom
        rgb_image[i, :, :, -2:] = c_view    # Right
        rgb_image[i, :, :, 64:66] = c_view  # Left
    
    combined_grid = tvu.make_grid(rgb_image, nrow=n_cols, padding=0, normalize=False)
    combined_grid_np = combined_grid.permute(1, 2, 0).cpu().numpy()
    combined_grid_np = (combined_grid_np + 1.0) / 2.0  # -1->0, 1->1 로 매핑
    combined_grid_np = np.clip(combined_grid_np, 0, 1) # 안전하게 다시 한번 클리핑
    
    ax_img.imshow(combined_grid.permute(1, 2, 0).cpu().numpy())
    # ax_img.set_title(f"Generated Samples (T={time_step:02d})") 
    ax_img.axis('off')


    # Right Graphs Grid
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1], wspace=0.1, hspace=0.15)
    # Top Right
    ax_v_bar = fig.add_subplot(inner_gs[0, 0])
    ax_v_hm = fig.add_subplot(inner_gs[0, 1])

    render_bar_graph_two_degree_neuron_level_on_ax(
        ax_v_bar, act1_fourier_d2, num_freqs, 
        num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs,
        title=sub_bar1_title,
    )
    
    render_heatmap_neuron_level_on_ax(
        ax_v_hm, act1_fourier_d2, num_freqs, 
        num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs,
        title=f"2-Degree Heatmap"
    )    
    
    # Bottom Right
    ax_ff_bar = fig.add_subplot(inner_gs[1, 0])
    ax_ff_hm = fig.add_subplot(inner_gs[1, 1])

    render_bar_graph_two_degree_neuron_level_on_ax(
        ax_ff_bar, act2_fourier_d2, num_freqs, 
        num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs,
        title=sub_bar2_title,
    )
    
    render_heatmap_neuron_level_on_ax(
        ax_ff_hm, act2_fourier_d2, num_freqs, 
        num_2d_bases=num_2d_bases, two_degree_idxes=two_degree_idxes, two_degree_idxes_descs=two_degree_idxes_descs,
        title=f"2-Degree Heatmap"
    )
    
    if title is None:
        fig.suptitle(f"Inference and FVEs", fontsize=20)
    else:
        fig.suptitle(title)
    
    plt.savefig(save_dir, dpi=80)
    plt.close(fig)



def generate_gif_animation(root_dir, file_name):
    png_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')])
    if len(png_files) > 1:
        frames = []
        for filename in png_files:
            img_path = os.path.join(root_dir, filename)
            img = Image.open(img_path)
            frames.append(img)
            
        gif_path = os.path.join(root_dir, file_name)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # 200ms per frame (5 FPS)
            loop=0         # repeat
        )
        print(f"✅ GIF Created: {gif_path}")



import argparse
import os
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import io
from PIL import Image
import time
from copy import deepcopy
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

from utils.manage_project import make_weight_and_result_dir


# def draw_pca(patch_activation, ax, labels, plot_centroid=True):
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(patch_activation.cpu().numpy())
    
#     unique_labels = np.unique([l for l in labels if l is not None])
    
#     # Scatter plot batch data
#     scatter_s = 5
#     ax.scatter(reduced[:, 0], reduced[:, 1], c=labels_np, cmap='hsv', s=scatter_s, alpha=0.6)
    
#     if plot_centroid:
#         # Spot Centroids of each label
#         norm = plt.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max())
#         cmap = plt.get_cmap('hsv')
#         centroid_scale = 4
        
#         for u_label in unique_labels:
#             mask = (labels_np == u_label)
#             if not np.any(mask): continue
            
#             centroid = reduced[mask].mean(axis=0)   # Mean of batches (multiple images)
#             ax.scatter(centroid[0], centroid[1],
#                     color=cmap(norm(u_label)),
#                     s=scatter_s*pow(centroid_scale,2), marker='o',
#                     edgecolors='black', linewidths=1.0,
#                     alpha=1.0, zorder=5)
    
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.set_title(f"PCA on Activations", fontsize=12)


def draw_entropy_accuracy_graph(
    timesteps, 
    opt_entropies_list, entropy_type,
    accuracies_list, ode_fail_tdx, ode_fail_val, save_path
):
    print(f"Draw entropy_accuracy_graph. Entropy Type : {entropy_type}")
    # Color Gradation by ODE Final Accuracy
    cmap_name = "traffic_sign_grad"
    high_accuracy = "#00FF00"
    low_accuracy = "#FF0000"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, [low_accuracy, high_accuracy], N=100)
    color_gradation = custom_cmap(accuracies_list)
        
    entropy_ylabel = ""
    if entropy_type in ("FVE Entropy", "Neumann Entropy"):
        entropy_ylabel = 'Entropy (nats)'
    elif entropy_type == "Stable Rank":
        entropy_ylabel = 'Rank'
                
    # (Needs Improvement) Average out patches for now
    opt_entropies_list = np.average(opt_entropies_list, axis=1)
    
    entropy_data_list = []
    entropy_data_list.append(
        # [marker, data_list, pred_tdx, data_label, y_axis_label, y_min, y_max]
        ["o", opt_entropies_list, -1, f"{entropy_type} Raw", entropy_ylabel, min(opt_entropies_list), max(opt_entropies_list)]
    )
        
    # 1st-Derivative Data
    d_opt_entropies = np.diff(opt_entropies_list)
    d_opt_entropies_list = d_opt_entropies.tolist()
    # (Needs Improvement) Currently argmax -> Later modify to the first peak!
    d_pred_tdx = int(np.nanargmax(d_opt_entropies_list))    
    d_min, d_max = min(d_opt_entropies_list), max(d_opt_entropies_list)
    # Pad Nones for empty timesteps (1st derivative!)
    d_opt_entropies_list.insert(0, None)
    d_pred_tdx += 1
    # entropy_data_list.append(["P", d_opt_entropies_list, d_pred_tdx, f"{entropy_type} 1st Derivative", "", d_min, d_max])
    
    # 2nd-Derivative Data
    dd_opt_entropies = np.diff(d_opt_entropies)
    dd_opt_entropies_list = dd_opt_entropies.tolist()
    # (Needs Improvement) Currently argmax -> Later modify to the first peak!
    dd_pred_tdx = int(np.nanargmax(dd_opt_entropies_list))    
    dd_min, dd_max = min(dd_opt_entropies_list), max(dd_opt_entropies_list)
    # Pad Nones for empty timesteps (2nd derivative!)
    dd_opt_entropies_list.insert(0, None)
    dd_pred_tdx += 1
    dd_opt_entropies_list.append(None)
    print(f"2nd_deriv_tdx : {dd_pred_tdx}", end=" ")
    entropy_data_list.append(["*", dd_opt_entropies_list, dd_pred_tdx, f"{entropy_type} 2nd Derivative", "", dd_min, dd_max])
    
    fig, ax = plt.subplots(figsize=(12, 7))      
    ax.set_zorder(len(entropy_data_list)+1)
    ax.patch.set_visible(False)  
    # print(f"ODE_fail_tdx : {ode_fail_tdx}")
    _draw_accuracy_graph(ax, timesteps, color_gradation, accuracies_list, ode_fail_tdx, ode_fail_val)
    
    axes_list = []
    for i, (marker, data_list, pred_tdx, data_label, y_axis_label, y_min, y_max) in enumerate(reversed(entropy_data_list)):
        curr_axis = ax.twinx()
        curr_axis.set_zorder(i)
        axes_list.append(curr_axis)
        _draw_entropy_graph(ax, curr_axis, timesteps, color_gradation, marker, data_list, pred_tdx, data_label, y_axis_label, y_min, y_max)
        # curr_axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
        # curr_axis.spines['right'].set_position(('axes', 1.0 + 0.08*i))
    
    ax.set_xlim(timesteps[0]-0.1, timesteps[-1]+0.1)
    ax.set_ylim(-0.05, 1.05)
    axes_list.append(ax)

    # 4. 스타일링 및 축 설정
    num_xticks = 6
    # 리스트로 변환하여 조작 가능하게 만듦
    display_timesteps = np.linspace(0, len(timesteps) - 1, num_xticks, dtype=int).tolist()
    ax.set_xticks(display_timesteps)
    
    # 3. 동적 라벨 생성 (insert 대신 조건문 사용)
    xtick_labels = [] 
    for tick in display_timesteps:
        noise_pct = int((fixed_total_sample_step - tick) / fixed_total_sample_step * 100)
        xtick_labels.append(f"{tick+1}/50\n({noise_pct}%)")
    
    # 양 끝단 부가 설명 추가
    xtick_labels[0] += "\nPure Noise"
    xtick_labels[-1] += "\nIncorrect Image"
    
    # 4. 축 적용
    ax.set_xticklabels(xtick_labels, fontsize=13)
    
    ax.set_ylabel('Correction Accuracy', fontsize=16, color="black")
    ax.set_ylim(-0.05, 1.05)
    # ax2.set_ylim(1.5, 2.5)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    lines_list, labels_list = [], []
    for curr_axis in reversed(axes_list):
        curr_lines, curr_labels = curr_axis.get_legend_handles_labels()
        lines_list.extend(curr_lines)
        labels_list.extend(curr_labels)
    ax.legend(lines_list, labels_list, 
            bbox_to_anchor=(1.0, 0.9),
            loc='upper right', fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=80)
    plt.close(fig)   


def _draw_accuracy_graph(
    ax, timesteps, color,
    accuracies_list, ode_fail_tdx, ode_fail_val, 
):    
    # ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    
    # ODE Final Accuracy
    ax.plot(timesteps, accuracies_list, color="black", linestyle='-', alpha=0.3, 
            linewidth=1.0, zorder=1)
    ax.scatter(timesteps, accuracies_list, color=color, s=60, 
                alpha=1.0, zorder=5, edgecolors="black", marker='^', label="ODE Final Accuracy")
        
    # ODE Failure timestep
    if ode_fail_tdx != -1:
        ax.axvline(x=ode_fail_tdx, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
        annotate_label = f"ODE Final Accuracy\n{ode_fail_val*100:.1f}% " 
        ax.annotate(
            annotate_label,
            xy=(ode_fail_tdx, ax.get_ylim()[1]),
            xytext=(5, 0),
            textcoords="offset points",
            ha='left', va='top',                    
            color='red', fontsize=10, 
        )

    # Find complete failure tdx (i.e. accuracy = 0.0)
    complete_fail_tdx = 0
    for accy in accuracies_list:
        if accy == 0.0:
            break
        complete_fail_tdx += 1
    
    print(f"complete_fail_tdx : {complete_fail_tdx}")
    ax.axvline(x=complete_fail_tdx, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
    annotate_label_complete = f"\n0.0%" 
    ax.annotate(
        annotate_label_complete,
        xy=(complete_fail_tdx, ax.get_ylim()[1]),
        xytext=(-5, 0),
        textcoords="offset points",
        ha='right', va='top',                    
        color='red', fontsize=10, 
    )
    
    if ode_fail_tdx != -1:
        ax.axvspan(complete_fail_tdx, ode_fail_tdx, color='red', alpha=0.15, zorder=0)


def _draw_entropy_graph(
    ax1, ax2, timesteps, color, marker,
    input_entropies_list, dd_pred_tdx, entropy_label, y_axis_label, y_min, y_max
):
    # Entropy 
    ax2.plot(timesteps, input_entropies_list, color="black", 
            linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)
    ax2.scatter(timesteps, input_entropies_list, color=color, s=40, alpha=0.8, marker=marker,
                zorder=3, edgecolors="black", label=entropy_label)
    
    # Prediction timestep
    ax1.axvline(x=dd_pred_tdx, color='blue', linestyle=':', linewidth=2.0, alpha=1.0, zorder=10)
    ax1.annotate(
        # f"Prediction {entropy_label}",
        f"Pred",
        xy=(dd_pred_tdx, ax1.get_ylim()[1]),
        xytext=(5, 1),  
        textcoords="offset points",
        ha='left', va='bottom',                    
        color='blue', fontsize=10
    )