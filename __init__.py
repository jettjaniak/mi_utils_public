from .analysis import *
from .misc import *
from .dla import *
from .ext_imports import *
from .plotting import *
from .prompts import *
from . import variables
from .generators_split_tokens import *
from .composition import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
owner = lambda pid: subprocess.run(f"ps -u -p {pid} | grep {pid} | cut -d ' ' -f 1", shell=True, capture_output=True, text=True).stdout.strip()
if owner(os.getpid()) == "stefan":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
print(f"{os.getpid()=}")