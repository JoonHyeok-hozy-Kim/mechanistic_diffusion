import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
import os
from datetime import datetime
from utils.seed import seed_everything

RADIUS = 15.0
NOISE_STD = 0.5

def generate_2d_modular_data(p, samples_per_class, radius, noise_std):
    classes = np.repeat(np.arange(p), samples_per_class)
    angles = (2 * np.pi / p) * classes
    
    x_ideal = radius * np.cos(angles)
    y_ideal = radius * np.sin(angles)
    ideal_points = np.stack([x_ideal, y_ideal], axis=1)
    
    noise = np.random.normal(scale=noise_std, size=ideal_points.shape)
    noisy_points = ideal_points + noise
    
    x_tensor = torch.tensor(noisy_points, dtype=torch.float32).unsqueeze(-1)    # (N, 2, 1)
    y_tensor = torch.tensor(classes, dtype=torch.long)
    
    return x_tensor, y_tensor


def plot_2d_modular_data(x_tensor, y_tensor, p, radius=1.0, save_dir=None, file_name=None):
    x_np = x_tensor.numpy()
    y_np = y_tensor.numpy()
    
    plt.figure(figsize=(8, 8))
    
    circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    scatter = plt.scatter(x_np[:, 0], x_np[:, 1], c=y_np, cmap='tab20', 
                          alpha=0.6, edgecolors='none', s=15, label='Noisy Samples')
    
    ideal_angles = (2 * np.pi / p) * np.arange(p)
    ideal_x = radius * np.cos(ideal_angles)
    ideal_y = radius * np.sin(ideal_angles)
    plt.scatter(ideal_x, ideal_y, c='black', marker='o', s=50, label='Ideal Centers')
    
    cbar = plt.colorbar(scatter, ticks=range(p))
    cbar.set_label('Modular Class (0 to P-1)')
    
    plt.title(f'2D Modular Sandbox Dataset ($P={p}$)', fontsize=14, pad=15)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True, alpha=0.3)
    
    plt.axis('equal') 
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_dir is None:
        img_dir = os.path.join("results", "data")
        img_dir = os.path.join(img_dir, "get_2d_dataset")
        os.makedirs(img_dir, exist_ok=True)
    else:
        assert os.path.exists(save_dir), f"save_dir(={save_dir}) does not exists."
        img_dir = save_dir
    
    if file_name is None:
        now_str = datetime.now().strftime("%y%m%d_%H%M")
        file_path = os.path.join(img_dir, f"two_dimensional-{now_str}.png")
    else:
        file_path = os.path.join(img_dir, f"{file_name}.png")
        
    plt.savefig(file_path)
    
    plt.show()


def get_two_dimensional_image_data_dict(num_classes, num_images_per_class, data_radius, noise_std):
    x, _ = generate_2d_modular_data(
        p=num_classes, samples_per_class=num_images_per_class, radius=data_radius, noise_std=noise_std
    )  # (N, 2, 1), (N, )
    
    # # (N, C, H, W) = (N, 1, 2, 1)
    # N, D, C = x.shape
    # x = x.reshape(N, C, D)
    # x = x.unsqueeze(-1)         # (N, C, H, W) = (N, 1, 2, 1)
    
    # (N, C, H, W) = (N, 2, 1, 1)
    x = x.unsqueeze(-1)         # (N, C, H, W) = (N, 2, 1, 1)
    
    res_dict = {}
    for label in range(num_classes):
        sdx = num_images_per_class * label
        edx = num_images_per_class * (label+1)
        curr_label_tensor = x[sdx:edx]
        res_dict[label] = list(torch.unbind(curr_label_tensor, dim=0))
    return res_dict


def squeeze_input_data(input_datapoints):
    if len(input_datapoints.shape) == 4:
        if input_datapoints.shape[1] == 2 and \
            input_datapoints.shape[2] == input_datapoints.shape[3] == 1:
                res = input_datapoints.squeeze()
        else:
            raise NotImplementedError(f"Unexpected shape of : {input_datapoints.shape}")
    else:
        raise NotImplementedError(f"Unexpected shape of : {input_datapoints.shape}") 

    return res


def determine_classes(squeezed_datapoints, p):
    assert len(squeezed_datapoints.shape) == 2 and squeezed_datapoints.shape[1] == 2, \
        f"Assumed shape is (N, C), but provided : {squeezed_datapoints.shape}"
    
    x = squeezed_datapoints[:, 0]
    y = squeezed_datapoints[:, 1]
    
    angles = torch.atan2(y, x)
    angles = angles % (2 * np.pi)
    shift_angle = np.pi / p             # Shift by half (2*pi / p)
    shifted_angles = (angles + shift_angle) % (2 * np.pi)
    
    predicted_classes = torch.floor(shifted_angles / (2 * np.pi / p)).long()
    
    return predicted_classes


def get_radii(squeezed_datapoints):
    x = squeezed_datapoints[:, 0]
    y = squeezed_datapoints[:, 1]
    
    return torch.sqrt(x**2 + y**2)
    

def evaluate_2d_accuracy(generated_points, target_classes, p, radius_threshold=1.0):
    # Input Format : (N, C, H, W)
    # N, C, H, W = generated_points.shape
    # generated_points = generated_points.squeeze(-1)         # Cancel out width-dim
    # # generated_points = generated_points.reshape(N, H, C)    # For the prev shape of (N, 1, 2, 1) 
    # generated_points = generated_points.squeeze(-1)
    
    sqzd_datapoints = squeeze_input_data(generated_points)
    predicted_classes = determine_classes(sqzd_datapoints, p)
    target_classes = target_classes.squeeze(-1)
        
    radii = get_radii(sqzd_datapoints)
    valid_radius_mask = radii >= radius_threshold
    
    is_correct = (predicted_classes == target_classes) & valid_radius_mask    
    accuracy = is_correct.float().mean().item()
    
    return accuracy, predicted_classes


def plot_2d_answer_and_pred(pred_points, answer_labels, label_a=None, label_b=None, p=23, radius=1.0, title=None, save_dir=None, file_name=None):
    accuracy, predicted_labels = evaluate_2d_accuracy(pred_points, answer_labels, p)
    
    N = pred_points.shape[0]
    pred_points_2d = pred_points.view(N, 2)
    pred_points_np = pred_points_2d.cpu().numpy()
    answer_labels_np = answer_labels.cpu().numpy()
    predicted_labels_np = predicted_labels.squeeze(-1).cpu().numpy()
    
    ideal_angles = (2 * np.pi / p) * np.arange(p)
    ideal_x = radius * np.cos(ideal_angles)
    ideal_y = radius * np.sin(ideal_angles)
    
    plt.figure(figsize=(8, 8))
    circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    # Boundary Dashlines
    boundary_angles = ideal_angles + (np.pi / p)
    for b_angle in boundary_angles:
        bx = radius * np.cos(b_angle) * 1.2 
        by = radius * np.sin(b_angle) * 1.2 
        plt.plot(
            [0, bx], [0, by],
            linestyle='--', color='gray', alpha=0.4, linewidth=1.0
        )
        
    # Origin
    plt.scatter(
        0, 0, 
        color="black",             
        marker="o",       
        s=80,             
        alpha=1.0,
    )
    
    correct_mask = (predicted_labels_np == answer_labels_np)
    cmap = plt.get_cmap('hsv', p)
    
    for pp in range(p):
        plt.scatter(
            ideal_x[pp], ideal_y[pp], 
            color="black",             
            marker="o",       
            s=80,             
            alpha=1.0,
        )
        
        label_radius = radius * 1.18 
        lx = label_radius * np.cos(ideal_angles[pp])
        ly = label_radius * np.sin(ideal_angles[pp])
        
        plt.text(
            lx, ly, str(pp), 
            color='black', 
            fontsize=11, 
            fontweight='bold',
            ha='center', 
            va='center'
        )
    
    
    for i in range(N):
        marker_style = 'o' if correct_mask[i] else 'x'
        is_correct_str = "Correct" if correct_mask[i] else "Wrong"
        color = 'green' if correct_mask[i] else 'red'
        
        p_x = pred_points_np[i, 0]
        p_y = pred_points_np[i, 1]        
        pp = answer_labels_np[i]
        
        # 1. 예측된 위치 산점도
        plt.scatter(
            p_x, p_y, 
            color=color,             
            marker=marker_style,       
            s=20,                     
            alpha=0.5,
            linewidths=1,              
            # label=f'Case {i} ({is_correct_str})'
        )
        
        if not correct_mask[i] and label_a is not None and label_b is not None:
            plt.text(
                p_x, p_y+0.03, f"{label_a[i]}+{label_b[i]}", 
                color='black', 
                fontsize=4, 
                ha='center', 
                va='center'
            )
        
        # plt.plot(
        #     [p_x, ideal_x[pp]], 
        #     [p_y, ideal_y[pp]], 
        #     marker='>', markersize=10, linestyle=':', color=color, alpha=0.4, linewidth=0.8
        # )
        
        plt.annotate('', 
             xy=(ideal_x[pp], ideal_y[pp]),      
             xytext=(p_x, p_y),                  
             arrowprops=dict(arrowstyle="->", ls=":", color=color, lw=0.8, alpha=0.4))
    
    title = title if title is not None else f'2D Modular Sandbox Debug ($P={p}$) | Acc: {accuracy*100:.0f}%'
    plt.title(title, fontsize=14, pad=15)
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    # plt.grid(True, alpha=0.3)
    plt.axis('equal') 
    
    # plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    
    if save_dir is None:
        img_dir = os.path.join("results", "data", "get_2d_dataset")
        os.makedirs(img_dir, exist_ok=True)
    else:
        assert os.path.exists(save_dir), f"save_dir(={save_dir}) does not exists."
        img_dir = save_dir
    
    if file_name is None:
        now_str = datetime.now().strftime("%y%m%d_%H%M")
        file_path = os.path.join(img_dir, f"two_dimensional-{now_str}-accy_{accuracy:.2f}.png")
    else:
        file_path = os.path.join(img_dir, f"{file_name}.png")
        
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    P = 6 
    N = 100
    RANDOM_SEED=46
    
    seed_everything(RANDOM_SEED)
    print(f"P={P}, N={N}")
    
    # data_x, data_y = generate_2d_modular_data(p=P, samples_per_class=N, noise_std=0.05)
    
    # print(f"data_x.shape: {data_x.shape}")      # (1000, 2, 1)
    # print(f"First three data_x:\n{data_x[:3]}")   
    # print(f"First three data_y:\n{data_y[:3]}")
    
    # data_x_squeezed = data_x.squeeze(-1)
    # plot_2d_modular_data(data_x_squeezed, data_y, p=P)
    
    
    letter_dict = get_two_dimensional_image_data_dict(num_classes=P, num_images=N)
    for k, v in letter_dict.items():
        y = torch.Tensor(np.repeat(k, N))
        v = torch.stack(v, dim=0)
        accuracy, _ = evaluate_2d_accuracy(v, y, P)
        print(f"For p={k}, accuracy = {accuracy:.2f}")
        v = v.squeeze(-1)
        # n, c, d = v.shape
        # x = v.reshape(n, d, c)
        # x = x.squeeze(-1)
        # plot_2d_modular_data(x, y, p=P, file_name=f"label_plot-p={k}-accy_{accuracy:.2f}")
        plot_2d_modular_data(v, y, p=P, file_name=f"label_plot-p={k}-accy_{accuracy:.2f}")

    