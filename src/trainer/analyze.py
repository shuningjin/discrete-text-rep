import logging as log
import os
from tqdm import tqdm
import math
import numpy as np
import json
import torch
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.nn.util import move_to_device, device_mapping

from src import device, cuda_device
from src.trainer.trainer_util import filter_indices
from src.utils.util import input_from_batch



label_maps = {
    "ag": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech",},
    "db": {
        0: "Company",
        1: "EducationalInstitution",
        2: "Artist",
        3: "Athlete",
        4: "OfficeHolder",
        5: "MeanOfTransportation",
        6: "Building",
        7: "NaturalPlace",
        8: "Village",
        9: "Animal",
        10: "Plant",
        11: "Album",
        12: "Film",
        13: "WrittenWork",
    },
    "yelp-full": {0: "1", 1: "2", 2: "3", 3: "4", 4: "5",},
}


def analyze(config, task, model, vocab, split):
    model.to(device)

    sub_config = {
        "run_dir": "",
        "quantizer.type": "",
        "quantizer.level": "",
        "quantizer.M": "",
        "quantizer.K": "",
    }
    for k in sub_config:
        sub_config[k] = config.get(k)
    # log.info(f"\nconfig: {sub_config}")

    config_path = os.path.join(config.sub_run_dir, 'config.log')
    with open(config_path, "w") as conf:
        json.dump(sub_config, conf, indent=2)

    label_map = label_maps[config.task]

    if split == "val":
        num = task.n_val_examples
        data = task.val_data
    elif split == "train":
        num = task.n_train_examples
        data = task.train_data
    elif split == "test":
        num = task.n_test_examples
        data = task.test_data

    val_iter = BasicIterator(config.analyze.batch_size, instances_per_epoch=num)(
        data, num_epochs=1, shuffle=False
    )

    out_path = os.path.join(config.sub_run_dir, f'{split}.log')
    out_file = open(out_path, 'w')
    print(f'\nAnalyzing split: {split}, output path: {out_path}')
    print("LABEL||INPUT||NLL||CODE", file=out_file)

    model.eval()
    with torch.no_grad():
        n_batches = math.ceil(num / config.analyze.batch_size)
        for batch in tqdm(val_iter, total=n_batches):
            batch = move_to_device(batch, cuda_device)
            o = model(batch)

            input_idx = input_from_batch(batch)["enc_in"]
            input_token = np.array(
                [
                    vocab.get_token_from_index(i, namespace="tokens")
                    for i in input_idx.view(-1).cpu().numpy()
                ]
            ).reshape(input_idx.shape)
            nlls = o["nll"].cpu().numpy().tolist()
            indices = o["indices"].squeeze().cpu().numpy().tolist()
            labels = batch["labels"].cpu().numpy().tolist()
            for inp, nll, idx, label in zip(input_token, nlls, indices, labels):
                # input
                clean_inp = list(filter(lambda x: x != "@@PADDING@@", inp))
                # nll
                clean_nll = list(filter(lambda x: x > 0, nll))
                # idx
                clean_idx = filter_indices(idx)

                # format
                clean_idx = [str(x) for x in clean_idx]
                clean_nll = np.around(clean_nll, 4).astype(str).tolist()
                label = label_map[label]

                print(f"{label}||{' '.join(clean_inp)}||{' '.join(clean_nll[:-1])}||{' '.join(clean_idx)}", file=out_file)
