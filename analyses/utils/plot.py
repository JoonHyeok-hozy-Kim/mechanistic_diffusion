import os

import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA

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
    

def save_full_layers_grid(layer_activations, labels, save_dir, time_step, var_name, title=None, plot_centroid=False):
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
    