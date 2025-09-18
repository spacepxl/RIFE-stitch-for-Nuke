import logging
import torch
from model.IFNet_HDv3 import IFNet

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
PATH = "model/flownet_v4.14.pkl"
TORCHSCRIPT_MODEL = "./nuke/Cattery/RIFE/RIFE_stitch.pt"


def load_flownet():
    def convert(param):
        return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

    flownet = IFNet()

    if torch.cuda.is_available():
        flownet.cuda()

    flownet.load_state_dict(convert(torch.load(PATH)), False)
    return flownet


class FlowNetNuke(torch.nn.Module):
    """
    FlowNetNuke is a module that performs optical flow estimation and frame interpolation using the RIFE algorithm.

    Args:
        scale (float): The scale factor for resizing the input frames. Default is 1.0.
        optical_flow (int): Flag indicating whether to return the optical flow and mask or the interpolated frames.
                            Set to 1 to return optical flow and mask, and 0 to return interpolated frames. Default is 0.
    """

    def __init__(
        self,
        scale: float = 1.0,
        optical_flow: int = 0,
        ensemble: int = 0,
    ):
        super().__init__()
        self.optical_flow = optical_flow
        self.scale = scale
        self.ensemble = ensemble
        self.flownet = load_flownet()
        self.flownet_half = load_flownet().half()

    def forward(self, x):
        """
        Forward pass of the RIFE model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: If `optical_flow` is True, returns the optical flow and mask concatenated along the channel dimension,
                          with shape (batch_size, 2 * channels, height, width).
                          If `optical_flow` is False, returns the interpolated frames and alpha channel concatenated along the channel dimension,
                          with shape (batch_size, (channels + 1), height, width).
        """
        b, c, h, w = x.shape
        dtype = x.dtype
        ensemble = bool(self.ensemble)
        scale = self.scale if self.scale in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0] else 1.0
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

        # Padding
        padding_factor = max(128, int(128 / scale))
        pad_h = ((h - 1) // padding_factor + 1) * padding_factor
        pad_w = ((w - 1) // padding_factor + 1) * padding_factor
        pad_dims = (0, pad_w - w, 0, pad_h - h)
        x = torch.nn.functional.pad(x, pad_dims)

        scale_list = (8.0 / scale, 4.0 / scale, 2.0 / scale, 1.0 / scale)

        if dtype == torch.float32:
            flow, mask, image = self.flownet((x), scale_list, ensemble)
        else:
            flow, mask, image = self.flownet_half((x), scale_list, ensemble)

        # Return the optical flow and mask
        if self.optical_flow:
            return torch.cat((flow[:, :, :h, :w], mask[:, :, :h, :w]), 1)

        # Return the interpolated frames
        alpha = torch.ones((b, 1, h, w), dtype=dtype, device=device)
        return torch.cat((image[:, :, :h, :w], alpha), dim=1).contiguous()


def trace_rife(model_file=TORCHSCRIPT_MODEL):
    """
    Traces the RIFE model using FlowNetNuke and saves the traced flow model.

    Returns:
        None
    """
    with torch.jit.optimized_execution(True):
        rife_nuke = torch.jit.script(FlowNetNuke().eval().requires_grad_(False))
        model_file = TORCHSCRIPT_MODEL
        rife_nuke.save(model_file)
        LOGGER.info(rife_nuke.code)
        LOGGER.info(rife_nuke.graph)
        LOGGER.info("Traced flow saved: %s", model_file)


if __name__ == "__main__":
    trace_rife()
