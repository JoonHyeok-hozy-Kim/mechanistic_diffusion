import torch
import math


class RF:
    def __init__(self, model, time_sampling_type="normal", noise_schedule='linear', radius=1.0):
        self.model = model
        self.time_sampling_type = time_sampling_type
        self.noise_schedule = noise_schedule  # 'linear', 'cosine' (VP), 've' (VE)
        self.radius = radius

    def forward(self, x, cond):
        b = x.size(0)
        if self.time_sampling_type == "normal":
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        elif self.time_sampling_type == "uniform":
            t = torch.rand((b,)).to(x.device)
        elif self.time_sampling_type == "uniform_skewed":
            half_b = (b + 1) // 2
            rest_b = b - half_b
            
            lower_bound = max(0.0, 1.0 - 1.0 / self.radius)
            
            t_skewed = lower_bound + (1.0 - lower_bound) * torch.rand((half_b,), device=x.device)
            t_rest = torch.rand((rest_b,), device=x.device)            
            t_concat = torch.cat([t_skewed, t_rest])
            t = t_concat[torch.randperm(b, device=x.device)]
        else:
            raise NotImplementedError(f"Unspecified time_sampling_type : {self.time_sampling_type}")
            
        t = torch.clamp(t, min=1e-2, max=0.99)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        
        # if self.noise_schedule == 'linear':
        #     zt = (1 - texp) * x + texp * z1
        #     target_v = z1 - x
            
        # elif self.noise_schedule == 'vp':
        #     # VP (Variance Preserving) - Cosine Schedule
        #     alpha_t = torch.cos(texp * math.pi / 2)
        #     sigma_t = torch.sin(texp * math.pi / 2)
            
        #     zt = alpha_t * x + sigma_t * z1
            
        #     # dz/dt
        #     target_v = - (math.pi / 2) * torch.sin(texp * math.pi / 2) * x + \
        #                (math.pi / 2) * torch.cos(texp * math.pi / 2) * z1
                       
        # elif self.noise_schedule == 've':
        #     sigma_max = 50.0  
            
        #     # # Linear
        #     # zt = x + (texp * sigma_max) * z1
        #     # target_v = sigma_max * z1
            
        #     # Non-linear
        #     rho = 7.0
        #     sigma_t = (texp ** rho) * sigma_max
        #     zt = x + sigma_t * z1
        #     target_v = rho * (texp ** (rho - 1.0)) * sigma_max * z1
            
        # else:
        #     raise NotImplementedError(f"Noinse Schedule {self.noise_schedule} not implemented.")

        zt = self.inject_noise(x, z1, texp)
        target_v = self.get_target_velocity_by_noise(x, z1, texp)
        vtheta = self.model(zt, t, cond)
        
        loss_unreduced = ((target_v - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        loss = (loss_unreduced * t).mean()  
        
        return loss
    
    
    def inject_noise(self, x_src, noise, t_tensor, ve_sigma_max=50.0, ve_rho=7.0):
        if self.noise_schedule == 'linear':                    
            x_tgt = (1 - t_tensor) * x_src + (t_tensor) * noise
        elif self.noise_schedule == 'vp':
            # VP (Variance Preserving) - Cosine Schedule
            alpha_t = math.cos(t_tensor * math.pi / 2)
            sigma_t = math.sin(t_tensor * math.pi / 2)
            
            x_tgt = alpha_t * x_src + sigma_t * noise
        elif self.noise_schedule == 've':            
            # # Linear
            # x_tgt = answer_img + (curr_noise_ratio * ve_sigma_max) * fixed_noise
            
            # Non-linear
            sigma_t = (t_tensor ** ve_rho) * ve_sigma_max
            x_tgt = x_src + sigma_t * noise
        else:
            raise NotImplementedError(f"Unexpected noise_schedule : {self.noise_schedule}")                    
                        
        return x_tgt
    
    
    def get_target_velocity_by_noise(self, x_src, noise, t_tensor, ve_sigma_max=50.0, ve_rho=7.0):
        if self.noise_schedule == 'linear':                    
            return noise - x_src
        elif self.noise_schedule == 'vp':
            return - (math.pi / 2) * torch.sin(t_tensor * math.pi / 2) * x_src + \
                       (math.pi / 2) * torch.cos(t_tensor * math.pi / 2) * noise
        elif self.noise_schedule == 've':            
            return ve_rho * (t_tensor ** (ve_rho - 1.0)) * ve_sigma_max * noise
        else:
            raise NotImplementedError(f"Unexpected noise_schedule : {self.noise_schedule}")                    
               

    @torch.no_grad()
    def sample(self, z, cond, sample_steps=50):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            if hasattr(self.model, 'prediction_target') and self.model.prediction_target == 'epsilon':
                t = min(t, 0.99)
            
            t_tensor = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t_tensor, cond)
            z = z - dt * vc
            images.append(z)
            
        return images