import os
import glob


def get_best_checkpoint(weight_dir):
    assert os.path.exists(weight_dir), f"Invalid path : {weight_dir}"
    
    best_format = weight_dir + "/model_best.pth"
    model_best_file = glob.glob(best_format)
    if model_best_file:
        return model_best_file[0]
    else:
        print(f"Best model not found at {weight_dir}")
    
    final_format = weight_dir + "/model_final.pth"
    model_final_file = glob.glob(final_format)
    if model_final_file:
        return model_final_file[0]
    else:
        print(f"Final model not found at {weight_dir}")
    
    raise FileNotFoundError(f"No feasible weight found at : {weight_dir}")


def get_wandb_run_info(run_url):
    # Get Info
    elements = run_url.split("//")[1].split("/")
    user_name = elements[1]
    project_name = elements[2]
    run_id = elements[4].split("?")[0]
    
    return user_name, project_name, run_id    


def find_project_run_dirctory(input_dir, run_id, pos_idx=2, separator="-"):    
    for root, dirs, _ in os.walk(input_dir):
        for d in dirs:
            parts = d.split(separator)
            if len(parts) >= pos_idx+1 and parts[pos_idx] == run_id:
                full_path = os.path.join(root, d)
                return full_path
                
    return None