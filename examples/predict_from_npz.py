from __future__ import annotations

import argparse
import json

import numpy as np

from tactile_force_envelope.inference import TransformerForceEnvelopePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one offline envelope prediction")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="NPZ containing model input arrays")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sample = np.load(args.input)
    predictor = TransformerForceEnvelopePredictor(args.checkpoint, device=args.device)
    prediction = predictor.predict(
        condition_tokens=sample["condition_tokens"],
        condition_mask=sample["condition_mask"],
        robot_state=sample["robot_state"],
        control_actions=sample["control_actions"],
        tactile_history=sample.get("tactile_history"),
        tactile_history_mask=sample.get("tactile_history_mask"),
    )
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()

