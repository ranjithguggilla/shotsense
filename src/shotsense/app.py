"""Gradio front end. Run with `python -m shotsense.app`."""

from __future__ import annotations

from pathlib import Path

import joblib

from .scoring import BODY_PARTS, PLAY_PATTERNS, TECHNIQUES, ShotInput, explain

MODEL_PATH = Path("outputs/model.joblib")


def run_shot(bundle: dict, x, y, body_part, technique, play_pattern, pressure,
             first_time, one_on_one, cross) -> tuple[str, dict]:
    shot = ShotInput(x=float(x), y=float(y), body_part=body_part, technique=technique,
                     play_pattern=play_pattern, under_pressure=int(pressure),
                     first_time=int(first_time), one_on_one=int(one_on_one),
                     assist_cross=int(cross))
    result = explain(bundle, shot)
    return result["text"], {"xG": round(result["xg"], 3)}


def build_app(model_path: Path = MODEL_PATH):
    import gradio as gr

    bundle = joblib.load(model_path)
    with gr.Blocks(title="shotsense") as demo:
        gr.Markdown("## Shot quality\nAttack runs left to right on a 120 by 80 pitch; "
                    "the goal is at x = 120.")
        with gr.Row():
            with gr.Column():
                x = gr.Slider(60, 120, value=108, label="x (distance along the pitch)")
                y = gr.Slider(0, 80, value=40, label="y (width, 40 is centre)")
                body = gr.Dropdown(BODY_PARTS, value="Right Foot", label="Body part")
                tech = gr.Dropdown(TECHNIQUES, value="Normal", label="Technique")
                pattern = gr.Dropdown(PLAY_PATTERNS, value="Regular Play", label="How the move started")
                pressure = gr.Checkbox(label="Defender pressure")
                first = gr.Checkbox(label="First-time shot")
                one = gr.Checkbox(label="One-on-one with the keeper")
                cross = gr.Checkbox(label="Came from a cross")
                go = gr.Button("Score")
            with gr.Column():
                text = gr.Textbox(label="Reading", lines=5)
                number = gr.JSON(label="Expected goals")
        go.click(lambda *a: run_shot(bundle, *a),
                 [x, y, body, tech, pattern, pressure, first, one, cross], [text, number])
    return demo


if __name__ == "__main__":
    build_app().launch()
