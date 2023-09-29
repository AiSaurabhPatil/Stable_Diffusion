import torch 
import numpy as np 
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt:str , uncond_prompt:str , input_image=None ,strength = 0.8,
            do_cfg=True , cfg_scale=7.5 , sampler_name='ddpm',n_inference_step=50,
            models={},seed = None , device=None , idle_device=None,tokenizer=None):
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("value of strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x : x.to(idle_device)
        else:
            to_idle = lambda x : x 
        
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models['clip']
        
        if do_cfg: 
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length",max_length=77).input_ids
            # convert the tokens into tensors 
            cond_tokens = torch.tensor(cond_tokens , dtype= torch.long , device=device)
            #(batch_size , seq_len) -> (batch_size , seq_len ,dim)
            cond_tokens = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],padding='max_length', max_length= 77).input_ids
            # convert the tokens into tensors 
            uncond_tokens = torch.tensor(uncond_tokens , dtype= torch.long , device=device)
            #(batch_size , seq_len) -> (batch_size , seq_len ,dim)
            uncond_tokens = clip(uncond_tokens)

            # combining both conditional and unconditional tokens 
            context  = torch.cat([cond_tokens,uncond_tokens])

        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length',max_length=77).input_ids
            tokens = torch.tensors(tokens , dtype=torch.long, device=device)
            context = clip(tokens)

        to_idle(clip)


        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_step)
        else:
            raise ValueError('unknown sampler name')


        latent_shape = (1, 4 , LATENTS_HEIGHT ,LATENTS_WIDTH)
    
        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            image_input_tensor = input_image.resize((WIDTH,HEIGHT))

            # (Height , Width , Channels)
            image_input_tensor = np.array(image_input_tensor)
            # (Height , Width , Channels) -> (Height , Width , Channels) 
            image_input_tensor = torch.tensor(image_input_tensor , dtype=torch.float)
            # rescaling the image 
            image_input_tensor = rescale(image_input_tensor , (0,255),(-1,1) )
            # addding new dimension
            #(Height , Width , Channels) --> (Batch_size , Height , Width ,Channels)
            image_input_tensor = image_input_tensor.unsqueeze(0)
            # changing the position of shapes
            #(Batch_size , Height , Width ,Channels) -> (Batch_size, Channels, Height , Width )
            image_input_tensor = image_input_tensor.permute(0,3,1,2)

            # generating a noise for image 
            encoder_noise = torch.randn(latent_shape , generator=generator , device=device)
            #creating a latent from image for encoder
            latents= encoder(image_input_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents =sampler.add_noise(latents , sampler.timesteps[0])

            to_idle(encoder)
        
        else:
            latents = torch.randn(latent_shape , generator=generator , device=device)
        
        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i , timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents 

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2,1,1,1)
                model_output = diffusion(model_input , context , time_embedding)
                output_cond , output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.set(timestep , latent_shape , model_output)
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)
        # decode the latent to image 
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]





def rescale(x ,old_range ,new_range , clamp = False):
    old_min , old_max = old_range
    new_min , new_max = new_range

    x -= old_min
    x *= (new_max - new_min)/(old_max - old_min)
    x +=new_min

    if clamp:
        x = x.clamp(new_max,new_min)
    
    return x 


def get_time_embedding(timestep):
    # Shape : (160,)
    freqs = torch.pow(100000 , -torch.arange(start=0 , end=160 ,dtype = torch.float32)/ 160)
    # Shape : (1, 160)
    x = torch.tensor([timestep],dtype=torch.float32)[:,None]* freqs[None]
    # Shape : (1 , 160 * 2)
    return torch.cat([torch.cos(x) , torch.sin(x)] , dim=-1)
