from typing import Tuple, Dict, Any

from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK

from nncf.torch.initialization import PTInitializingDataLoader
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args


class MyInitializingDataLoader(PTInitializingDataLoader):
    def get_inputs(self, dataloader_output: Any) -> Tuple[Tuple, Dict]:
        # your implementation - `dataloader_output` is what is returned by your dataloader,
        # and you have to turn it into a (args, kwargs) tuple that is required by your model
        # in this function, for instance, if your dataloader returns dictionaries where
        # the input image is under key `"img"`, and your YOLOv8 model accepts the input
        # images as 0-th `forward` positional arg, you would do:
        return dataloader_output["img"], {}

    def get_target(self, dataloader_output: Any) -> Any:
        # and in this function you should extract the "ground truth" value from your
        # dataloader, so, for instance, if your dataloader output is a dictionary where
        # ground truth images are under a "gt" key, then here you would write:
        return dataloader_output["gt"]


class MyCustomModel(DetectionModel):
    def __init__(self, nncf_config_dict, dataloader, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

        nncf_config = NNCFConfig.from_dict(nncf_config_dict)
        nncf_dataloader = MyInitializingDataLoader(dataloader)
        nncf_config = register_default_init_args(nncf_config, nncf_dataloader)
        self.compression_ctrl, self.model = create_compressed_model(self.model, nncf_config)


class MyTrainer(DetectionTrainer):
    def __init__(self, dataloader, nncf_config_dict, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        nncf_config = NNCFConfig.from_dict(nncf_config_dict)
        self.nncf_dataloader = MyInitializingDataLoader(dataloader)
        self.nncf_config = register_default_init_args(nncf_config, self.nncf_dataloader)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        self.compression_ctrl, model.model = create_compressed_model(model.model, self.nncf_config)
        return model


def main():
    args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3, mode='train', verbose=False)
    trainer = DetectionTrainer(overrides=args)
    trainer._setup_train(world_size=0)
    train_loader = trainer.train_loader

    nncf_config_dict = {
        "input_info": {
            "sample_size": [1, 3, 640, 640]
        },
        "log_dir": 'yolov8_output',  # The log directory for NNCF-specific logging outputs.
        "compression": {
            "algorithm": "quantization"  # Specify the algorithm here.
        },
    }

    nncf_trainer = MyTrainer(train_loader, nncf_config_dict, overrides=args)
    nncf_trainer.train()


if __name__ == '__main__':
    main()