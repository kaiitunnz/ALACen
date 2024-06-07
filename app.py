import ast
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

try:
    import torch

    from src.alacen.alacen import ALACen, ALACenSession
    from src.alacen.asr.whisper import Whisper
    from src.alacen.paraphrase.pegasus import PegasusAlacen
    from src.alacen.tts.voicecraft.voicecraft import VoiceCraftTTS, VoiceCraftArgs
    from src.alacen.lipsync.diff2lip.diff2lip import Diff2Lip, Diff2LipArgs
except ImportError:
    raise ImportError(
        "ALACen has not been set up. Please run `bash setup.sh` to set up ALACen."
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Verbose mode", default=False
)
parser.add_argument("-n", "--num-paraphrases", type=int, default=5)
parser.add_argument(
    "-s", "--share", action="store_true", help="Create a shared link", default=False
)
args = parser.parse_args()

VERBOSE = args.verbose
NUM_PARAPHRASES = args.num_paraphrases
SHARE = args.share


asr = Whisper()
paraphrase = PegasusAlacen()
tts = VoiceCraftTTS(model_name="330M_TTSEnhanced")
lipsync = Diff2Lip(Diff2LipArgs())

alacen = ALACen(asr, paraphrase, tts, lipsync)
session: Optional[ALACenSession] = None


def asr(video_path: str) -> str:
    global session

    session = alacen.create_session(video_path, verbose=VERBOSE, device=DEVICE)
    return session.asr()


def paraphrase(transcript: str, num_paraphrases: float) -> Optional[List[str]]:
    if session is None:
        return []
    try:
        num_paraphrases = int(num_paraphrases)
    except ValueError:
        num_paraphrases = NUM_PARAPHRASES
    return session.paraphrase(transcript, num_paraphrases)


def parse_params(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params["cut_off_sec"] = (
            None
            if params["cut_off_sec"].lower() == "none"
            else float(params["cut_off_sec"])
        )
    except ValueError:
        params["cut_off_sec"] = None

    params["beam_size"] = int(params["beam_size"])
    params["retry_beam_size"] = int(params["retry_beam_size"])
    params["stop_repetition"] = int(params["stop_repetition"])
    params["sample_batch_size"] = int(params["sample_batch_size"])
    params["codec_audio_sr"] = int(params["codec_audio_sr"])
    params["codec_sr"] = int(params["codec_sr"])
    params["top_k"] = int(params["top_k"])

    parsed_silence_tokens = ast.literal_eval(params["silence_tokens"])
    if isinstance(parsed_silence_tokens, tuple):
        params["silence_tokens"] = parsed_silence_tokens

    try:
        params["seed"] = None if params["seed"] == "None" else int(params["seed"])
    except ValueError:
        params["seed"] = None

    padding = params["padding"].lower()
    params["padding"] = padding if padding in ["begin", "end", "both"] else None

    return params


def tts(
    transcript: str,
    target_transcript: str,
    cut_off_sec: str = "None",
    margin: float = 0.08,
    cutoff_tolerance: float = 1,
    beam_size: int = 50,
    retry_beam_size: int = 200,
    stop_repetition: int = 3,
    sample_batch_size: int = 3,
    codec_audio_sr: int = 16000,
    codec_sr: int = 50,
    top_k: int = 0,
    top_p: float = 0.9,
    temperature: float = 1,
    silence_tokens: str = "(1388, 1898, 131)",
    kvcache: float = 1,
    seed: str = "None",
    padding: str = "none",
) -> Optional[Path]:
    if session is None:
        return None

    params = {
        "cut_off_sec": cut_off_sec,
        "margin": margin,
        "cutoff_tolerance": cutoff_tolerance,
        "beam_size": beam_size,
        "retry_beam_size": retry_beam_size,
        "stop_repetition": stop_repetition,
        "sample_batch_size": sample_batch_size,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "silence_tokens": silence_tokens,
        "kvcache": kvcache,
        "seed": seed,
        "padding": padding,
    }

    tts_args = VoiceCraftArgs.constructor(**parse_params(params))
    generated_audio_paths = session.tts(transcript, target_transcript, tts_args)
    return generated_audio_paths[0]


def lipsync(generated_audio_path: str) -> Optional[Path]:
    if session is None:
        return None
    return session.lipsync(generated_audio_path)


def main():
    with gr.Blocks().queue(concurrency_count=1) as demo:
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(
                    format="mp4", source="upload", interactive=True, label="Input Video"
                )
                transcript = gr.Textbox(
                    placeholder="Input video transcript",
                    interactive=False,
                    label="Transcript",
                )
                asr_button = gr.Button("Transcribe", interactive=False)
            with gr.Column():
                paraphrases = gr.Radio(
                    type="value", interactive=False, label="Paraphrases"
                )
                selected_paraphrase = gr.Textbox(
                    placeholder="Selected paraphrase",
                    interactive=False,
                    label="Selected Paraphrase",
                )
                paraphrase_button = gr.Button("Paraphrase", interactive=False)
                paraphrases.change(
                    lambda x: x, inputs=paraphrases, outputs=selected_paraphrase
                )
            with gr.Column():
                generated_audio = gr.Audio(
                    format="wav",
                    type="filepath",
                    interactive=False,
                    label="Generated Speech",
                )
                generate_audio_button = gr.Button("Text-to-Speech", interactive=False)
            with gr.Column():
                lipsynced_video = gr.Video(
                    format="mp4", interactive=False, label="Output Video"
                )
                lipsync_button = gr.Button("Lip Sync", interactive=False)

        with gr.Row():
            run_all_button = gr.Button("Run All", interactive=False)

        with gr.Accordion("Parameters", open=False):
            with gr.Group():
                num_paraphrases = gr.Number(
                    NUM_PARAPHRASES,
                    interactive=True,
                    label="Number of Paraphrases to Generate",
                )
            with gr.Group():
                cut_off_sec = gr.Textbox(
                    "None", interactive=True, label="Cut Off Seconds"
                )
                margin = gr.Number(0.08, interactive=True, label="Margin")
                cutoff_tolerance = gr.Number(
                    1, interactive=True, label="Cutoff Tolerance"
                )
                beam_size = gr.Number(50, interactive=True, label="Beam Size")
                retry_beam_size = gr.Number(
                    200, interactive=True, label="Retry Beam Size"
                )
                stop_repetition = gr.Number(
                    3, interactive=True, label="Stop Repetition"
                )
                sample_batch_size = gr.Number(
                    3, interactive=True, label="Sample Batch Size"
                )
                codec_audio_sr = gr.Number(
                    16000, interactive=True, label="Codec Audio Sample Rate"
                )
                codec_sr = gr.Number(50, interactive=True, label="Codec Sample Rate")
                top_k = gr.Number(0, interactive=True, label="Top K")
                top_p = gr.Slider(
                    minimum=0, maximum=1, value=0.9, interactive=True, label="Top P"
                )
                temperature = gr.Number(1, interactive=True, label="Temperature")
                silence_tokens = gr.Textbox(
                    "(1388, 1898, 131)", interactive=True, label="Silence Tokens"
                )
                kvcache = gr.Number(1, interactive=True, label="KV Cache")
                seed = gr.Textbox("None", interactive=True, label="Random Seed")
                padding = gr.Radio(
                    ["none", "begin", "end", "both"],
                    value="end",
                    interactive=True,
                    label="Padding",
                )

        # --------------- Register event handlers --------------- #

        def upload_video(video_path: Optional[str]):
            interactive = video_path is not None
            asr_button = gr.Button("Transcribe", interactive=interactive)
            run_all_button = gr.Button("Run All", interactive=interactive)
            return asr_button, run_all_button

        def click_asr(video_path: str, output_dict: Optional[Dict[str, Any]] = None):
            transcript = asr(video_path)
            transcript_textbox = (
                gr.Textbox(
                    placeholder="Input video transcript",
                    interactive=False,
                    label="Transcript",
                )
                if transcript is None
                else gr.Textbox(
                    value=transcript,
                    interactive=True,
                    label="Transcript",
                )
            )
            paraphrase_button = (
                gr.Button("Paraphrase", interactive=False)
                if transcript is None
                else gr.Button("Paraphrase", interactive=True)
            )
            if output_dict is not None:
                output_dict["transcript"] = transcript
            return transcript_textbox, paraphrase_button

        def click_paraphrase(
            transcript: str,
            num_paraphrases: float,
            output_dict: Optional[Dict[str, Any]] = None,
        ):
            paraphrase_list = paraphrase(transcript, num_paraphrases)
            best_paraphrase = None if paraphrase_list is None else paraphrase_list[0]
            paraphrases = gr.Radio(
                paraphrase_list,
                value=best_paraphrase,
                type="value",
                interactive=True,
                label="Paraphrases",
            )
            selected_paraphrase = (
                gr.Textbox(
                    placeholder="Selected paraphrase",
                    interactive=False,
                    label="Selected Paraphrase",
                )
                if best_paraphrase is None
                else gr.Textbox(
                    value=best_paraphrase,
                    interactive=True,
                    label="Selected Paraphrase",
                )
            )
            generate_audio_button = (
                gr.Button("Text-to-Speech", interactive=False)
                if paraphrase_list is None
                else gr.Button("Text-to-Speech", interactive=True)
            )
            if output_dict is not None:
                output_dict["best_paraphrase"] = best_paraphrase
            return paraphrases, selected_paraphrase, generate_audio_button

        def click_generate_audio(*tts_args):
            generated_audio_path = tts(*tts_args)
            lipsync_button = (
                gr.Button("Lip Sync", interactive=False)
                if generated_audio_path is None
                else gr.Button("Lip Sync", interactive=True)
            )
            return generated_audio_path, lipsync_button

        def click_lipsync(generated_audio_path: str):
            lipsynced_video = lipsync(generated_audio_path)
            return lipsynced_video

        def run_all(
            video_path: str,
            num_paraphrases: float,
            cut_off_sec: str = "None",
            margin: float = 0.08,
            cutoff_tolerance: float = 1,
            beam_size: int = 50,
            retry_beam_size: int = 200,
            stop_repetition: int = 3,
            sample_batch_size: int = 3,
            codec_audio_sr: int = 16000,
            codec_sr: int = 50,
            top_k: int = 0,
            top_p: float = 0.9,
            temperature: float = 1,
            silence_tokens: str = "(1388, 1898, 131)",
            kvcache: float = 1,
            seed: str = "None",
            padding: str = "none",
        ) -> Tuple[str, List[str], str, Path, Path]:
            output_dict = {}
            transcript, paraphrase_button = click_asr(video_path, output_dict)
            paraphrases, selected_paraphrase, generate_audio_button = click_paraphrase(
                output_dict["transcript"], num_paraphrases, output_dict
            )
            params = [
                cut_off_sec,
                margin,
                cutoff_tolerance,
                beam_size,
                retry_beam_size,
                stop_repetition,
                sample_batch_size,
                codec_audio_sr,
                codec_sr,
                top_k,
                top_p,
                temperature,
                silence_tokens,
                kvcache,
                seed,
                padding,
            ]
            generated_audio_path, lipsync_button = click_generate_audio(
                output_dict["transcript"], output_dict["best_paraphrase"], *params
            )
            output_path = click_lipsync(generated_audio_path)
            return (
                transcript,
                paraphrase_button,
                paraphrases,
                selected_paraphrase,
                generate_audio_button,
                generated_audio_path,
                lipsync_button,
                output_path,
            )

        paraphrase_param_components = [num_paraphrases]

        tts_param_components = [
            cut_off_sec,
            margin,
            cutoff_tolerance,
            beam_size,
            retry_beam_size,
            stop_repetition,
            sample_batch_size,
            codec_audio_sr,
            codec_sr,
            top_k,
            top_p,
            temperature,
            silence_tokens,
            kvcache,
            seed,
            padding,
        ]

        input_video.change(
            upload_video, inputs=input_video, outputs=[asr_button, run_all_button]
        )
        asr_button.click(
            click_asr, inputs=input_video, outputs=[transcript, paraphrase_button]
        )
        paraphrase_button.click(
            click_paraphrase,
            inputs=[transcript] + paraphrase_param_components,
            outputs=[paraphrases, selected_paraphrase, generate_audio_button],
        )
        generate_audio_button.click(
            click_generate_audio,
            inputs=[transcript, selected_paraphrase] + tts_param_components,
            outputs=[generated_audio, lipsync_button],
        )
        lipsync_button.click(
            click_lipsync, inputs=generated_audio, outputs=lipsynced_video
        )
        run_all_button.click(
            run_all,
            inputs=[input_video] + paraphrase_param_components + tts_param_components,
            outputs=[
                transcript,
                paraphrase_button,
                paraphrases,
                selected_paraphrase,
                generate_audio_button,
                generated_audio,
                lipsync_button,
                lipsynced_video,
            ],
        )

        # ------------------------------------------------------- #

        demo.launch(inbrowser=True, share=SHARE)


if __name__ == "__main__":
    main()
