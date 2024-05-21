import os
from dataclasses import dataclass
from pathlib import Path

from .. import LipSync


@dataclass
class Diff2LipArgs:
    verbose: bool = False
    sample_mode: str = "cross"  # or "reconstruction"
    num_gpus: int = 3
    generate_from_filelist: int = 0
    model_path: str = (
        Path(__file__).parent / "Diff2Lip_checkpoints/e7.24.1.3_model260000_paper.pt"
    )

    model_flags: str = (
        "--attention_resolutions 32,16,8 --class_cond False --learn_sigma True --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm False"
    )
    diffusion_flags: str = (
        "--predict_xstart False  --diffusion_steps 1000 --noise_schedule linear --rescale_timesteps False"
    )
    sample_flags: str = (
        "--sampling_seed=7 {sample_input_flags} --timestep_respacing ddim25 --use_ddim True --model_path={model_path}"
    )
    data_flags: str = "--nframes 5 --nrefer 1 --image_size 128 --sampling_batch_size=32"
    tfg_flags: str = (
        "--face_hide_percentage 0.5 --use_ref=True --use_audio=True --audio_as_style=True"
    )
    gen_flags: str = (
        "--generate_from_filelist {generate_from_filelist}  --video_path={video_path} --audio_path={audio_path} --out_path={out_path} --save_orig=False --face_det_batch_size 64 --pads 0,0,0,0 --is_voxceleb2=False"
    )

    def get_commands(self, video_path: str, audio_path: str, out_path: str) -> str:
        if self.sample_mode == "reconstruction":
            sample_input_flags = (
                "--sampling_input_type=first_frame --sampling_ref_type=first_frame"
            )
        elif self.sample_mode == "cross":
            sample_input_flags = "--sampling_input_type=gt --sampling_ref_type=gt"
        else:
            raise ValueError("Invalid sample_mode.")

        model_flags = self.model_flags
        diffusion_flags = self.diffusion_flags
        sample_flags = self.sample_flags.format(
            sample_input_flags=sample_input_flags, model_path=self.model_path
        )
        data_flags = self.data_flags
        tfg_flags = self.tfg_flags
        gen_flags = self.gen_flags.format(
            generate_from_filelist=self.generate_from_filelist,
            video_path=video_path,
            audio_path=audio_path,
            out_path=out_path,
        )
        all_flags = " ".join(
            (
                model_flags,
                diffusion_flags,
                sample_flags,
                data_flags,
                tfg_flags,
                gen_flags,
                f"--verbose {self.verbose}",
            )
        )
        parent_dir = Path(__file__).parent
        if self.num_gpus > 1:
            return f"mpiexec -n {self.num_gpus} python {parent_dir / 'generate_dist.py'} {all_flags}"
        return f"python {parent_dir / 'generate.py'} {all_flags}"


class Diff2Lip(LipSync):
    def __init__(self, args: Diff2LipArgs):
        self.args = args

    def generate(self, video_path: str, audio_path: str, out_path: str):
        os.system(self.args.get_commands(video_path, audio_path, out_path))
