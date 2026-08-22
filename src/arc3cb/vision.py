"""Optional image priming for vision-capable models (gemma-4-31b).

At run start, if the model's endpoint accepted the image capability probe, the
opening frame is rendered to a PNG and the model is asked — once — what it sees
and what the goal might be. The answer is injected into the first prompt as an
explicitly-unverified hypothesis. Text remains the primary representation
throughout the run (NVIDIA AVO reached 100 RHAE text-only); this is a bounded
enhancement, not a dependency.

Requires pillow (``pip install 'arc3cb[vision]'``); degrades to no priming.
"""

from __future__ import annotations

import base64

# Official ARC-AGI-3 16-color palette (same colors the arcprize.org viewer uses).
PALETTE = [
    (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),
    (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204),
    (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
    (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214),
]

PRIMING_QUESTION = (
    "This is the opening screen of an unknown grid-based puzzle game. Describe "
    "what you see and what you think the goal of the game might be, briefly."
)


def render_png(grid: list[list[int]], scale: int = 8) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    h, w = len(grid), len(grid[0])
    img = Image.new("RGB", (w, h))
    img.putdata([PALETTE[v % 16] for row in grid for v in row])
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def prime(transport, grid: list[list[int]]) -> str | None:
    """One image round-trip; returns the model's read of the opening frame,
    or None when unsupported/unavailable. Best-effort by design."""
    png = render_png(grid)
    if png is None:
        return None
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PRIMING_QUESTION},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]
    try:
        result = transport.chat(messages, purpose="image-prime", max_output_tokens=2048)
        return result.text.strip() or None
    except Exception:
        return None
