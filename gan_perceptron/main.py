from env_parser import Env
def gan_perceptron():
    from .utils import load_dataset
    env = Env.get_instance()
    route_loader, tensor_shape = load_dataset()
    generator = Generator(env.NOISE_DIM).to(env.DEVICE)
    discriminator = Discriminator().to(env.DEVICE)