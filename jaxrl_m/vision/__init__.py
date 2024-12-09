from jaxrl_m.vision.bigvision_resnetv2 import resnetv2_configs
from jaxrl_m.vision.impala import impala_configs
from jaxrl_m.vision.resnet_v1 import resnetv1_configs
from jaxrl_m.vision.small_encoders import small_configs
from jaxrl_m.vision.robofume_encoders import robofume_configs

encoders = dict()
encoders.update(impala_configs)
encoders.update(resnetv2_configs)
encoders.update(resnetv1_configs)
encoders.update(small_configs)
encoders.update(robofume_configs)
