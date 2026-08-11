import torch
import math


class RF:
    def __init__(self, model, timestep_sampling_type="normal", prediction_target='x', noise_schedule='linear', radius=1.0):
        self.model = model
        self.timestep_sampling_type = timestep_sampling_type
        self.prediction_target = prediction_target  # 'x', 'v', 'epsilon'
        self.noise_schedule = noise_schedule        # 'linear', 'vp' (cosine), 've' (VE)
        self.radius = radius

    def forward(self, x, cond):
        b = x.size(0)
        if self.timestep_sampling_type == "normal":
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        elif self.timestep_sampling_type == "uniform":
            t = torch.rand((b,)).to(x.device)
        elif self.timestep_sampling_type == "uniform_skewed":
            half_b = (b + 1) // 2
            rest_b = b - half_b
            
            lower_bound = max(0.0, 1.0 - 1.0 / self.radius)
            
            t_skewed = lower_bound + (1.0 - lower_bound) * torch.rand((half_b,), device=x.device)
            t_rest = torch.rand((rest_b,), device=x.device)            
            t_concat = torch.cat([t_skewed, t_rest])
            t = t_concat[torch.randperm(b, device=x.device)]
        else:
            raise NotImplementedError(f"Unspecified timestep_sampling_type : {self.timestep_sampling_type}")
            
        t = torch.clamp(t, min=1e-2, max=0.99)
        t_expanded = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = self.inject_noise(x, z1, t_expanded)
        target_v = self.get_target_velocity_by_noise(x, z1, t_expanded)
        
        pred = self.model(zt, t, cond)
        v_pred = self.convert_pred_to_v_pred(zt, pred, t_expanded)
        
        loss_unreduced = ((target_v - v_pred) ** 2).mean(dim=list(range(1, len(x.shape))))
        loss = (loss_unreduced * t).mean()  
        
        return loss
    
    
    def inject_noise(self, x_src, noise, t_expanded, ve_sigma_max=50.0, ve_rho=7.0):
        if self.noise_schedule == 'linear':                    
            x_tgt = (1 - t_expanded) * x_src + (t_expanded) * noise
        elif self.noise_schedule == 'vp':
            # VP (Variance Preserving) - Cosine Schedule
            alpha_t = torch.cos(t_expanded * math.pi / 2)
            sigma_t = torch.sin(t_expanded * math.pi / 2)
            
            x_tgt = alpha_t * x_src + sigma_t * noise
        elif self.noise_schedule == 've':            
            # # Linear
            # x_tgt = answer_img + (curr_noise_ratio * ve_sigma_max) * fixed_noise
            
            # Non-linear
            sigma_t = (t_expanded ** ve_rho) * ve_sigma_max
            x_tgt = x_src + sigma_t * noise
        else:
            raise NotImplementedError(f"Unexpected noise_schedule : {self.noise_schedule}")                    
                        
        return x_tgt
    
    
    def get_target_velocity_by_noise(self, x_src, noise, t_expanded, ve_sigma_max=50.0, ve_rho=7.0):
        if self.noise_schedule == 'linear':                    
            return noise - x_src
        elif self.noise_schedule == 'vp':
            return - (math.pi / 2) * torch.sin(t_expanded * math.pi / 2) * x_src + \
                       (math.pi / 2) * torch.cos(t_expanded * math.pi / 2) * noise
        elif self.noise_schedule == 've':            
            return ve_rho * (t_expanded ** (ve_rho - 1.0)) * ve_sigma_max * noise
        else:
            raise NotImplementedError(f"Unexpected noise_schedule : {self.noise_schedule}")                    
    
    
    def convert_pred_to_v_pred(self, zt, pred, t_expanded):
        if self.prediction_target == 'v':
            '''
            Outut : f(z_t, t) = v : v-prediction
            '''
            return pred
        
        elif self.prediction_target == 'x':
            if self.noise_schedule == 'linear':
                '''
                Output : (z_t - f(z_t, t)) / t
                Then in RF, loss = || z_1 - x - ((z_t - f(z_t, t)) / t) ||^2
                                = || z_1 - x - ((t*z_1 + (1-t)x - f(z_t, t)) / t) ||^2
                                = || - (( x - f(z_t, t)) / t) ||^2
                Thus, x = f(z_t, t) : x-prediction
                '''
                return (zt - pred) / t_expanded
            elif self.noise_schedule == 'vp':
                alpha_t = torch.cos(t_expanded * math.pi / 2)
                sigma_t = torch.sin(t_expanded * math.pi / 2)
                return (math.pi / 2) * (alpha_t * zt - pred) / sigma_t
            elif self.noise_schedule == 've':
                ve_rho = 7.0
                return (ve_rho / t_expanded) * (zt - pred)
            
        elif self.prediction_target == 'epsilon':
            if self.noise_schedule == 'linear':
                '''
                Divide the output with t for the ε-prediction
                Output : (f(z_t, t) - z_t) / (1-t)
                Then in RF, loss = || z_1 - x - (f(z_t, t) - z_t) / (1-t) ||^2
                                = || z_1 - (z_t - t*z_1) / (1-t) - (f(z_t, t) - z_t) / (1-t) ||^2
                                = || (z_1 - f(z_t, t)) / (1-t) ||^2
                                = || (ε - f(z_t, t)) / (1-t) ||^2
                Thus, ε = f(z_t, t) : ε-prediction
                '''
                return (pred - zt) / (1 - t_expanded)
            elif self.noise_schedule == 'vp':
                alpha_t = torch.cos(t_expanded * math.pi / 2)
                sigma_t = torch.sin(t_expanded * math.pi / 2)
                return (math.pi / 2) * (pred - sigma_t * zt) / alpha_t
            elif self.noise_schedule == 've':
                ve_rho = 7.0
                ve_sigma_max = 50.0
                return ve_rho * (t_expanded ** (ve_rho - 1.0)) * ve_sigma_max * pred
        else:
            raise ValueError(f"Unexpected prediction_target : {self.prediction_target}")
                   

    @torch.no_grad()
    def sample(self, z, cond, sample_steps=50):
        b = z.size(0)
        dt = 1.0 / sample_steps
        # dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            if self.prediction_target == 'epsilon':
                t = min(t, 0.99)
                
            t_tensor = torch.tensor([t] * b).to(z.device)
            t_expanded = t_tensor.view([b, *([1] * len(z.shape[1:]))])
            
            pred = self.model(z, t_tensor, cond)
            vc = self.convert_pred_to_v_pred(z, pred, t_expanded)
            
            z = z - dt * vc
            images.append(z)
            
        return images



