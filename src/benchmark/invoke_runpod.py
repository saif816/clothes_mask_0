#! /usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   invoke_runpod.py
@Time   :   2025/05/22 10:10:07
@Author :   yhao 
@Email  :   uncle.yuanl@gmail.com
@Desc   :   Test and record the execution time of runpod
'''

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(f'【{__file__}】')
import os
import json
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
# ENDPOINT = "https://api.runpod.ai/v2/13ysy4c93gy4gi/runsync"  # benchmark
# ENDPOINT = "https://api.runpod.ai/v2/w7rg34ofztuvc2/runsync"  # Prod Debug
# ENDPOINT = "https://api.runpod.ai/v2/z5exp6hb3d7amh/runsync"    # Batch with safety check
ENDPOINT = "https://api.runpod.ai/v2/mnoleejqiirmrb/runsync"    # Batch without safety check


HEADER = {
    "authorization": os.getenv("RUNPOD_KEY")
}

BODY = {
    "input": {
        "data": [
            {
                "user_id": "a1f3c2e4",
                "fitcheck_id": "d49fd42c",
                "profile_image_url": "https://fitcheck-assets.s3.eu-north-1.amazonaws.com/models/565ae0e7-1bbe-4b01-8d27-7f5b2bac47c1.jpeg",
                "garment_url": "https://fitcheck-assets.s3.eu-north-1.amazonaws.com/clothes/2e6b967e-7744-46de-b7ea-9f0000ba5963.jpeg",
                "cloth_type": "upper"
            },
            {
                "user_id": "a1f3c2xx",
                "fitcheck_id": "d49fd4xx",
                "profile_image_url": "https://fitcheck-assets.s3.eu-north-1.amazonaws.com/models/565ae0e7-1bbe-4b01-8d27-7f5b2bac47c1.jpeg",
                "garment_url": "https://fitcheck-assets.s3.eu-north-1.amazonaws.com/clothes/2e6b967e-7744-46de-b7ea-9f0000ba5963.jpeg",
                "cloth_type": "overall"
            },
            {
                "user_id": "a1f3c2e4",
                "fitcheck_id": "d49fd42c",
                "profile_image_url": "https://fitcheck-assets.s3.eu-north-1.amazonaws.com/models/565ae0e7-1bbe-4b01-8d27-7f5b2bac47c1.jpeg",
                "garment_url": "https://fitcheck-assets.s3.eu-north-1.amazonaws.com/clothes/2e6b967e-7744-46de-b7ea-9f0000ba5963.jpeg",
                "cloth_type": "upper"
            },
            {
                "user_id": "a1f3c2xx",
                "fitcheck_id": "d49fd4xx",
                "profile_image_url": "https://fitcheck-assets.s3.eu-north-1.amazonaws.com/models/565ae0e7-1bbe-4b01-8d27-7f5b2bac47c1.jpeg",
                "garment_url": "https://fitcheck-assets.s3.eu-north-1.amazonaws.com/clothes/2e6b967e-7744-46de-b7ea-9f0000ba5963.jpeg",
                "cloth_type": "overall"
            }
        ],
        "num_inference_steps": 10,
        "guidance_scale": 2.5,
        "seed": 42
    }
}


def run_till_done(nums, spesuffix="", **kwargs):
    # update request body
    BODY["input"].update(kwargs)

    for callidx in range(nums + 1):
        response = requests.post(
            url=ENDPOINT,
            headers=HEADER,
            json=BODY
        )

        if callidx == 0:
            # warm up
            continue
        
        date = datetime.today().strftime('%Y_%m_%d')
        suffix = "_".join(f'{k}_{v}' for k, v in kwargs.items()) + spesuffix
        logfolder = Path(f"{Path(__file__).parent}/logs")
        
        if response.status_code == 200:
            with open(logfolder / f"{date}_{suffix}.jsonl", "a+") as f:
                resdict = response.json()
                log = {"workerId": resdict.pop("workerId")}
                log.update(resdict)
                f.write(json.dumps(log))
                f.write("\n")
        else:
            with open(logfolder / "error.txt", "a+") as f:
                f.write(response.text)
                f.write("\n")
        
        print(f"======================= {callidx} finished ====================")


if __name__ == "__main__":
    response = run_till_done(20, spesuffix="_wosc", num_inference_steps=10)
    print()