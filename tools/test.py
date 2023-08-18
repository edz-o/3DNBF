import argparse
import os
import os.path as osp
import pickle as pkl
from distutils.util import strtobool
import ipdb
from loguru import logger
import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmhuman3d.apis import multi_gpu_test, single_gpu_test
from mmhuman3d.data.datasets import build_dataloader, build_dataset
from mmhuman3d.models import build_architecture


def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d test model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--work-dir', help='the dir to save evaluation results')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default='pa-mpjpe',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "pa-mpjpe" for H36M',
    )
    parser.add_argument(
        '--skip_eval',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results',
    )
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.',
    )
    parser.add_argument(
        '--eval-options',
        nargs='+',
        default={},
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing',
    )
    parser.add_argument(
        '--eval_output',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--save_partseg',
        type=lambda x: bool(strtobool(x)),
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--output_file', type=str, default='', help='Saved output file for evaluation'
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False,
        indices=cfg.data.test.hparams.get('indices', None)
        if hasattr(cfg.data.test, 'hparams')
        else None,
    )

    rank, _ = get_dist_info()
    eval_cfg = cfg.get('evaluation', args.eval_options)
    if args.skip_eval:
        args.metrics = []
    eval_cfg.update(dict(metric=args.metrics, save_partseg=args.save_partseg))

    os.makedirs(args.work_dir, exist_ok=True)
    if not (args.eval_output or eval_cfg.get('eval_saved_results', False)):
        # build the model and load checkpoint
        if hasattr(cfg.model, 'hparams'):
            setattr(cfg.model.hparams, 'work_dir', args.work_dir)
            if hasattr(cfg.model.hparams, 'VISUALIZER'):
                cfg.model.hparams.VISUALIZER.DEBUG_LOG_DIR = osp.join(
                    args.work_dir, cfg.model.hparams.VISUALIZER.DEBUG_LOG_DIR
                )
        model = build_architecture(cfg.model)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, args.checkpoint, map_location='cpu')

        if not distributed:
            if args.device == 'cpu':
                model = model.cpu()
            else:
                model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    else:
        if args.output_file.endswith('.pkl'):
            outputs = pkl.load(open(args.output_file, 'rb'))
        elif args.output_file.endswith('.json'):
            outputs = mmcv.load(args.output_file)
        else:
            raise NotImplementedError(f'output file {args.output_file} not supported')

    if rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

        results = dataset.evaluate(outputs, args.work_dir, **eval_cfg)

        for k, v in results.items():
            print(f'\n{k} : {v:.2f}')

    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
