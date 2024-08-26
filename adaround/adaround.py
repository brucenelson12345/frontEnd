#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

#pylint:disable = unused-variable, redefined-outer-name
""" AIMET Quantsim code for YOLOX """

# General Python related imports
from __future__ import absolute_import
from __future__ import division
import argparse
from functools import partial
import json
import os
import pathlib
from tqdm import tqdm

# Torch related imports
import torch

# AIMET model zoo related imports: model construction, dataloader, evaluation
from aimet_zoo_torch.yolox import YOLOX
from aimet_zoo_torch.yolox.dataloader.dataloaders import get_data_loader
from aimet_zoo_torch.yolox.evaluators.coco_evaluator import COCOEvaluator

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

def seed(seed_number):
    """Set seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)


def eval_func(model, dataloader, img_size):
    """define evaluation func to evaluate model with data_loader"""
    evaluator = COCOEvaluator(dataloader, img_size)
    return evaluator.evaluate(model)


def forward_pass(decoder, model, data_loader):
    """forward pass for compute encodings"""
    #pylint:disable = no-member
    tensor_type = torch.cuda.FloatTensor
    model = model.eval()

    for imgs, _, info_imgs, ids in tqdm(data_loader):
        with torch.no_grad():
            imgs = imgs.type(tensor_type)
            outputs = model(imgs)
            if decoder is not None:
                outputs = decoder(outputs, dtype=outputs.type())


def read_model_configs_from_model_card(model_card):
    """read necessary params from model card"""
    parent_dir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
    config_filepath = os.path.join(
        parent_dir, "model", "model_cards", f"{model_card}.json"
    )

    if not os.path.exists(config_filepath):
        raise NotImplementedError("Model_config file doesn't exist")

    with open(config_filepath) as f_in:
        cfg = json.load(f_in)
        input_shape = tuple(x if x is not None else 1 for x in cfg["input_shape"])
        default_param_bw = cfg["optimization_config"]["quantization_configuration"][
            "param_bw"
        ]

    return input_shape, default_param_bw


def equalization(model, dummy_input):
    # aimet equalization
    equalize_model(model, input_shapes=(1, 3, 640, 640))

    sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

    return sim


def batchNormFolding(model, dummy_input):
    # aimet equalization
    _ = fold_all_batch_norms(model, input_shapes=(1, 3, 640, 640))

    sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

    return sim


def adaround(model, dummy_input, dataloader):
    # aimet equalization
    #_ = fold_all_batch_norms(model, input_shapes=(1, 3, 640, 640))
    params = AdaroundParameters(data_loader=dataloader, num_batches=1, default_num_iterations=32)

    # Returns model with adarounded weights and their corresponding encodings
    ada_model = Adaround.apply_adaround(model=model, dummy_input=dummy_input, params=params,
                                        path="output", 
                                        filename_prefix='adaround', 
                                        default_param_bw=8,
                                        default_quant_scheme=QuantScheme.post_training_tf_enhanced)

    sim = QuantizationSimModel(model=ada_model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

    return sim


def evaluateAndPrint(model, dataloader, img_size, model_type):
    print("Evaluating {model_type} Model")
    mAP_model = eval_func(model, dataloader, img_size)
    del model
    torch.cuda.empty_cache()

    return f"{model_type} Model | mAP: {100*mAP_model:.2f}%"


def main():
    """main function for quantization evaluation"""
    seed(1234)

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        #model.to(torch.device('cuda'))

    input_shape, default_param_bw = read_model_configs_from_model_card(
        "yolox_s"
    )
    img_size = (input_shape[-2], input_shape[-1])

    # Get Dataloader
    dataloader = get_data_loader(
        dataset_path="/home/pride/work/AIMET/datasets/coco",
        img_size=img_size,
        batch_size=64,
        num_workers=4,
    )

    # Set dummy input and use cuda if enabled
    dummy_input = torch.rand(1, 3, 640, 640)
    if use_cuda:
        dummy_input = dummy_input.cuda()

    # Load original model
    model = YOLOX(model_config="yolox_s")
    model.from_pretrained(quantized=False)
    model_orig = model.model

    # sim_orig = model.get_quantsim(quantized=False)
    # sim_eq = equalization(model_orig, dummy_input)
    sim_bnf = batchNormFolding(model_orig, dummy_input)
    sim_ada = adaround(sim_bnf.model, dummy_input, dataloader)

    forward_func = partial(forward_pass, None)
    # sim_orig.compute_encodings(forward_func, forward_pass_callback_args=dataloader)
    # sim_eq.compute_encodings(forward_func, forward_pass_callback_args=dataloader)
    # sim_bnf.compute_encodings(forward_func, forward_pass_callback_args=dataloader)
    sim_ada.compute_encodings(forward_func, forward_pass_callback_args=dataloader)

    # # fp32_res = evaluateAndPrint(model_orig, dataloader, img_size, "Original FP32")
    # # int8_res = evaluateAndPrint(sim_orig.model, dataloader, img_size, "Original INT8")
    # # eq_res = evaluateAndPrint(sim_eq.model, dataloader, img_size, "EQN INT8")
    # # bnf_res = evaluateAndPrint(sim_bnf.model, dataloader, img_size, "BNF INT8")
    ada_res = evaluateAndPrint(sim_ada.model, dataloader, img_size, "ADA INT8")

    # print(fp32_res)
    # print(int8_res)
    # print(eq_res)
    # print(bnf_res)
    print(ada_res)
    

if __name__ == "__main__":
    main()
