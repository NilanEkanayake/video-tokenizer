from .models import register, make, models
from . import transformer
from . import bottleneck
from . import loss
from . import larp_ar
from . import gptc
from . import larp_tokenizer
from .model import autoencoder
from .model_titok import titok
from .model_new import autoencoder as autoencoder_new
from .model_stat import autoencoder as autoencoder_stat
def get_model_cls(name):
    return models[name]