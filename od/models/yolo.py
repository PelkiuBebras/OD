from .efficient_rep import EfficientRep



def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False):
    return Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)
