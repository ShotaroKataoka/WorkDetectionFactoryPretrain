from modeling.backbone.mobilenet import MobileNetV2

def build_backbone(backbone, output_stride, BatchNorm):
  return MobileNetV2(output_stride=output_stride, BatchNorm=BatchNorm)
