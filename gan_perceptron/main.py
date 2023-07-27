from env_parser import Env
from .generator import Generator
from .discriminator import Discriminator

def gan_perceptron():
    from .utils import load_dataset
    env = Env.get_instance()
    # route_loader, tensor_shape = load_dataset()
    discriminator = Discriminator().to(env.DEVICE)
    generator = Generator(env.NOISE_DIMENSION).to(env.DEVICE)