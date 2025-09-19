from .base import DomainRandomizer
from .no_randomization import NoDomainRandomization
from .default import DefaultRandomizer
from .custom import CustomRandomizer

# register all domain randomizers
NoDomainRandomization.register()
DefaultRandomizer.register()
CustomRandomizer.register()