# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Generate bash scripts for RobustMVD Benchmark
"""

import os

machine = "psc"
self_folder = os.path.dirname(os.path.abspath(__file__))


def get_model_settings(model: str, dataset: str):
    if model == "mapanything":
        return {
            "model": "mapanything",
            "model.pretrained": "\\${root_experiments_dir}/mapanything/training/mapa_curri_24v_13d_36ipg_64g/checkpoint-last.pth",
            "evaluation_resolution": "\\${dataset.resolution_options.518_1_33_ar}"
            if dataset != "kitti"
            else "\\${dataset.resolution_options.518_3_20_ar}",
        }
    elif model == "moge_1":
        return {
            "model": "moge_1",
            "evaluation_resolution": "\"'(640, 480)'\""
            if dataset != "kitti"
            else "\"'(1248,384)'\"",
        }
    elif model == "moge_2":
        return {
            "model": "moge_2",
            "evaluation_resolution": "\"'(640, 480)'\""
            if dataset != "kitti"
            else "\"'(1248,384)'\"",
        }
    elif model == "vggt":
        return {
            "model": "vggt",
            "evaluation_resolution": (
                "\\${dataset.resolution_options.518_1_33_ar}"
                if dataset != "kitti"
                else "\\${dataset.resolution_options.518_3_20_ar}"
            ),
        }
    elif model == "must3r":
        return {
            "model": "must3r",
            "evaluation_resolution": "\\${dataset.resolution_options.512_1_33_ar}"
            if dataset != "kitti"
            else "\\${dataset.resolution_options.512_3_20_ar}",
        }
    else:
        raise ValueError(f"Unknown model: {model}")


def generate_shell_for_single_experiment(
    model: str = "mapanything",
    dataset: str = "eth3d",
    conditioning: str = "image",
    alignment: str = "none",
    view: str = "multi_view",
):
    benchmark_configs = {
        "machine": machine,
        "eval_dataset": dataset,
        "evaluation_conditioning": conditioning,
        "evaluation_alignment": alignment,
        "evaluation_views": view,
        "hydra.run.dir": '"\${root_experiments_dir}/mapanything/benchmarking/rmvd_'
        + f'{conditioning}_{alignment}_{view}/{dataset}/{model}"',
    }

    benchmark_configs.update(get_model_settings(model, dataset))

    file_content = "#!/bin/bash\n\n"

    file_content += "python \\\n"
    file_content += "\tbenchmarking/rmvd_mvs_benchmark/benchmark.py \\\n"

    for key, value in benchmark_configs.items():
        file_content += f"\t{key}={value} \\\n"

    # compute the output path for the shell script
    shell_output_path = os.path.join(
        self_folder,
        (view + "_metric") if alignment == "none" else view,
        f"{model}_{dataset}_{conditioning}.sh",
    )

    # ensure the directory exists
    os.makedirs(os.path.dirname(shell_output_path), exist_ok=True)

    # write the content to the file
    with open(shell_output_path, "w") as f:
        f.write(file_content)

    # change the file permission to make it executable
    os.chmod(shell_output_path, 0o755)

    print(f"Shell script generated at: {shell_output_path}")
    print(file_content)


if __name__ == "__main__":
    # generate single view metric experiments
    for dataset in ["kitti", "scannet"]:
        # non-conditioned
        for model in ["moge_2", "mapanything"]:
            generate_shell_for_single_experiment(
                model=model,
                dataset=dataset,
                conditioning="image",
                alignment="none",
                view="single_view",
            )

        # conditioned on intrinsics
        generate_shell_for_single_experiment(
            model="mapanything",
            dataset=dataset,
            conditioning="image+intrinsics",
            alignment="none",
            view="single_view",
        )

    # generate multi view metric experiments
    for dataset in ["kitti", "scannet"]:
        # non-conditioned
        for model in ["mapanything", "must3r"]:
            generate_shell_for_single_experiment(
                model=model,
                dataset=dataset,
                conditioning="image",
                alignment="none",
                view="multi_view",
            )

        # conditioned on intrinsics
        generate_shell_for_single_experiment(
            model="mapanything",
            dataset=dataset,
            conditioning="image+intrinsics",
            alignment="none",
            view="multi_view",
        )

        # conditioned on intrinsics and pose
        generate_shell_for_single_experiment(
            model="mapanything",
            dataset=dataset,
            conditioning="image+intrinsics+pose",
            alignment="none",
            view="multi_view",
        )

    # generate single view with alignment experiments
    for dataset in ["kitti", "scannet"]:
        # non-conditioned
        for model in [
            "moge_1",
            "moge_2",
            "vggt",
            "mapanything",
        ]:
            generate_shell_for_single_experiment(
                model=model,
                dataset=dataset,
                conditioning="image",
                alignment="median",
                view="single_view",
            )

        # conditioned on intrinsics
        generate_shell_for_single_experiment(
            model="mapanything",
            dataset=dataset,
            conditioning="image+intrinsics",
            alignment="median",
            view="single_view",
        )

    # generate multi view with alignment experiments
    for dataset in ["kitti", "scannet"]:
        # non-conditioned
        for model in ["vggt", "mapanything", "must3r"]:
            generate_shell_for_single_experiment(
                model=model,
                dataset=dataset,
                conditioning="image",
                alignment="median",
                view="multi_view",
            )

        # conditioned on intrinsics
        generate_shell_for_single_experiment(
            model="mapanything",
            dataset=dataset,
            conditioning="image+intrinsics",
            alignment="median",
            view="multi_view",
        )
