""" import codebase """
import importlib
from src.utils.util import *
from src.utils.config import handle_arguments, params_from_file
from src.models import *
from src.trainer.trainer import (
    Trainer,
    EMTrainer,
    CLSTrainer,
)
from src.trainer.analyze import analyze
from src.task import get_task


def main(cl_arguments):
    """ 1 setup """

    # set config
    cl_args = handle_arguments(cl_arguments)
    config = params_from_file(cl_args.config_file, cl_args.overrides)

    os.makedirs(config.exp_dir, exist_ok=True)
    os.makedirs(config.run_dir, exist_ok=True)
    if config.phase == "analyze":
        config.sub_runname = "analyze"
        config.sub_run_dir = os.path.join(config.run_dir, config.sub_runname)
    if config.sub_runname != "none":
        os.makedirs(config.sub_run_dir, exist_ok=True)

    # set device
    device, cuda_device = set_device()
    # set seed
    seed_all(config.seed)

    # set log
    if config.sub_runname == "none":
        log_path = os.path.join(config.run_dir, f"{config.phase}.log")
    else:
        log_path = os.path.join(config.sub_run_dir, f"{config.phase}.log")
    if config.phase == "pretrain":
        set_log(log_path, "a")  # append mode
    elif config.phase == "target_train":
        set_log(log_path, "w")  # overwrite mode

    log.info(f"device: {device}")

    # calculate bound
    calculate_kl_bound(config)

    # log config
    log.info(f"config: {str(config)}")

    """ 2 load task """

    task, vocab = get_task(config)
    config.vocab_size = vocab.get_vocab_size("tokens")
    # 30004 = 30000 + 4 special tokens (pad 0, unk 1, sos 2, eos 3,)

    """ 3 train/ eval """

    # baseline
    if config.model == "lstm":
        encoder = LSTMEncoder(config)
    elif config.model == "transformer":
        encoder = SimpleTransformerEncoder(config)
    elif config.model == "cbow":
        encoder = CBOWEncoder(config)
    elif config.model == "quantized_transformer":
        encoder = TransformerQuantizerEncoder(config)

    # pretrain
    if config.phase == "pretrain":
        train_config = {}
        assert config.model == "quantized_transformer"

        config.transformer
        config.quantizer
        config[config.quantizer.type]

        vae = TransformerEncoderDecoder(config, encoder)
        log.info(vae)

        if config.quantizer.type != "em" and config.pretrain.em_train == 0:
            trainer = Trainer(config, vae, vocab)
        else:
            trainer = EMTrainer(config, vae, vocab)
        trainer.train(config, task)

    elif config.phase == "target_train":
        if config.model == "quantized_transformer":
            # load pretrain
            if config.ckpt_path != "none":
                if config.ckpt_path == "current":
                    ckpt_path = current_run_ckpt(config)
                load_encoder(encoder, ckpt_path)
            cls = QuantizerforClassification(config, encoder)
        else:
            cls = EncoderClassifier(config, encoder)
        log.info(cls)

        trainer = CLSTrainer(cls)
        trainer.train(config, task)

    elif "analyze" in config.phase:

        assert config.model == "quantized_transformer"
        vae = TransformerEncoderDecoder(config, encoder)

        # load checkpoint
        assert config.ckpt_path != "none"
        ckpt_path = config.ckpt_path
        if config.ckpt_path == "current":
            ckpt_path = current_run_ckpt(config)
        checkpoint = torch.load(ckpt_path, map_location=device_mapping(cuda_device))
        vae.load_state_dict(checkpoint["model"], strict=False)

        if not isinstance(config.analyze.split, list):
            config.analyze.split = [config.analysis.split]
        for split in config.analyze.split:
            analyze(config, task, vae, vocab, split=split)


if __name__ == "__main__":
    main(sys.argv[1:])
