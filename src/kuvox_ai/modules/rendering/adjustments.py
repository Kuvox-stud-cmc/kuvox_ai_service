"""Versioned visual adjustment registry mirrored by the TypeScript editor."""

from __future__ import annotations

import math
from typing import Any, Final, cast

import numpy as np
from PIL import Image

REGISTRY_VERSION: Final = 1
FILTER_PRESETS: Final = (
    "Original",
    "Cinematic",
    "Film",
    "Vintage",
    "Warm",
    "Cold",
    "Dreamy",
    "Noir",
    "Vivid",
)

ADJUSTMENT_REGISTRY: Final[dict[str, dict[str, float]]] = {
    "exposure": {"neutral": 0, "min": -100, "max": 100},
    "brightness": {"neutral": 100, "min": 0, "max": 200},
    "contrast": {"neutral": 100, "min": 0, "max": 200},
    "temperature": {"neutral": 0, "min": -100, "max": 100},
    "tint": {"neutral": 0, "min": -100, "max": 100},
    "saturation": {"neutral": 100, "min": 0, "max": 200},
    "vibrance": {"neutral": 100, "min": 0, "max": 200},
    "lift": {"neutral": 0, "min": -100, "max": 100},
    "gamma": {"neutral": 0, "min": -100, "max": 100},
    "gain": {"neutral": 0, "min": -100, "max": 100},
}

PRESET_ADJUSTMENTS: Final[dict[str, dict[str, float]]] = {
    "Original": {},
    "Cinematic": {
        "exposure": -4,
        "brightness": 98,
        "contrast": 118,
        "temperature": -6,
        "tint": 2,
        "saturation": 106,
        "vibrance": 110,
        "lift": -4,
        "gamma": 8,
        "gain": 4,
    },
    "Film": {
        "exposure": 2,
        "contrast": 108,
        "temperature": 8,
        "tint": 4,
        "saturation": 88,
        "vibrance": 95,
        "lift": 6,
        "gamma": -4,
        "gain": -3,
    },
    "Vintage": {
        "brightness": 104,
        "contrast": 92,
        "temperature": 18,
        "tint": 8,
        "saturation": 78,
        "vibrance": 86,
        "lift": 12,
        "gamma": -8,
        "gain": -6,
    },
    "Warm": {
        "brightness": 103,
        "temperature": 25,
        "tint": 3,
        "saturation": 108,
        "vibrance": 106,
    },
    "Cold": {
        "brightness": 101,
        "temperature": -25,
        "tint": -3,
        "saturation": 104,
        "vibrance": 108,
    },
    "Dreamy": {
        "exposure": 8,
        "brightness": 108,
        "contrast": 88,
        "temperature": 6,
        "tint": 8,
        "saturation": 96,
        "vibrance": 112,
        "lift": 10,
        "gamma": -8,
        "gain": 4,
    },
    "Noir": {
        "brightness": 96,
        "contrast": 132,
        "saturation": 0,
        "vibrance": 0,
        "gamma": 12,
        "gain": 4,
    },
    "Vivid": {
        "brightness": 102,
        "contrast": 116,
        "saturation": 125,
        "vibrance": 135,
        "gain": 8,
    },
}


def resolve_visual_style(item: dict[str, Any]) -> dict[str, Any]:
    preset = resolve_filter_preset(item)
    intensity = _clamp(_number_property(item, "filters", "intensity", 100), 0, 100)
    blend = _clamp(_number_property(item, "filters", "blend", 100), 0, 100)
    preset_mix = (intensity / 100) * (blend / 100)
    preset_values = PRESET_ADJUSTMENTS[preset]
    resolved: dict[str, float] = {}
    for name, definition in ADJUSTMENT_REGISTRY.items():
        neutral = definition["neutral"]
        group = "color" if name in {"lift", "gamma", "gain"} else "adjust"
        user_value = _number_property(item, group, name, neutral)
        preset_value = preset_values.get(name, neutral)
        combined = user_value + (preset_value - neutral) * preset_mix
        resolved[name] = _round(_clamp(combined, definition["min"], definition["max"]))
    return {
        "registryVersion": REGISTRY_VERSION,
        "preset": preset,
        "intensity": _round(intensity),
        "blend": _round(blend),
        "adjustments": resolved,
        "filter": filter_values_for_adjustments(resolved),
    }


def filter_values_for_adjustments(adjustments: dict[str, float]) -> dict[str, float]:
    return {
        "brightness": _round(
            _clamp(
                adjustments["brightness"] / 100
                + adjustments["exposure"] / 240
                + adjustments["lift"] / 320,
                0.08,
                3,
            )
        ),
        "contrast": _round(
            _clamp(
                adjustments["contrast"] / 100
                + adjustments["gamma"] / 260
                + adjustments["gain"] / 420,
                0.08,
                3,
            )
        ),
        "saturation": _round(
            _clamp(
                adjustments["saturation"] / 100
                + (adjustments["vibrance"] - 100) / 260
                + adjustments["gain"] / 340,
                0,
                3.5,
            )
        ),
        "hueRotate": _round(
            _clamp(adjustments["temperature"] * -0.18 + adjustments["tint"] * 0.22, -45, 45)
        ),
        "sepia": _round(_clamp(max(0, adjustments["temperature"]) / 420, 0, 0.28)),
    }


def resolve_filter_preset(item: dict[str, Any]) -> str:
    property_name = "builtIn" if item.get("type") == "video" else "filterType"
    built_in = _string_property(item, "filters", property_name, "Original")
    lut = _string_property(item, "filters", "lutLibrary", "Original")
    selected = built_in if built_in not in {"", "None", "Original"} else lut
    if selected == "Cool":
        selected = "Cold"
    if selected in {"", "None"}:
        selected = "Original"
    return selected if selected in FILTER_PRESETS else "Original"


def property_value(item: dict[str, Any], group_name: str, property_name: str, fallback: Any) -> Any:
    properties = item.get("properties")
    if not isinstance(properties, dict):
        return fallback
    group = properties.get(group_name)
    if not isinstance(group, dict) or property_name not in group:
        return fallback
    value = group[property_name]
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def property_has_keyframes(item: dict[str, Any], group_name: str, property_name: str) -> bool:
    properties = item.get("properties")
    group = properties.get(group_name) if isinstance(properties, dict) else None
    value = group.get(property_name) if isinstance(group, dict) else None
    return (
        isinstance(value, dict)
        and isinstance(value.get("keyframes"), list)
        and bool(value["keyframes"])
    )


def apply_visual_style(image: Image.Image, style: Any) -> Image.Image:
    """Apply the same CSS-filter primitives emitted by the editor to one RGBA layer."""

    filter_values = getattr(style, "filter", None)
    if filter_values is None:
        return image
    brightness = float(filter_values.brightness)
    contrast = float(filter_values.contrast)
    saturation = float(filter_values.saturation)
    hue_rotate = float(filter_values.hue_rotate)
    sepia = float(filter_values.sepia)
    if (
        abs(brightness - 1) < 1e-6
        and abs(contrast - 1) < 1e-6
        and abs(saturation - 1) < 1e-6
        and abs(hue_rotate) < 1e-6
        and abs(sepia) < 1e-6
    ):
        return image

    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    rgb *= brightness
    rgb = (rgb - 0.5) * contrast + 0.5
    luminance = np.sum(
        rgb * np.array([0.213, 0.715, 0.072], dtype=np.float32), axis=-1, keepdims=True
    )
    rgb = luminance + saturation * (rgb - luminance)
    if abs(hue_rotate) >= 1e-6:
        radians = math.radians(hue_rotate)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        hue_matrix = np.array(
            [
                [
                    0.213 + cosine * 0.787 - sine * 0.213,
                    0.715 - cosine * 0.715 - sine * 0.715,
                    0.072 - cosine * 0.072 + sine * 0.928,
                ],
                [
                    0.213 - cosine * 0.213 + sine * 0.143,
                    0.715 + cosine * 0.285 + sine * 0.140,
                    0.072 - cosine * 0.072 - sine * 0.283,
                ],
                [
                    0.213 - cosine * 0.213 - sine * 0.787,
                    0.715 - cosine * 0.715 + sine * 0.715,
                    0.072 + cosine * 0.928 + sine * 0.072,
                ],
            ],
            dtype=np.float32,
        )
        rgb = np.matmul(rgb, hue_matrix.T)
    if sepia > 0:
        sepia_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
            dtype=np.float32,
        )
        rgb = rgb * (1 - sepia) + np.matmul(rgb, sepia_matrix.T) * sepia
    rgba[..., :3] = np.clip(rgb, 0, 1)
    return Image.fromarray(cast(Any, np.rint(rgba * 255).astype(np.uint8)), mode="RGBA")


def _number_property(
    item: dict[str, Any], group_name: str, property_name: str, fallback: float
) -> float:
    value = property_value(item, group_name, property_name, fallback)
    return (
        float(value) if isinstance(value, int | float) and math.isfinite(float(value)) else fallback
    )


def _string_property(
    item: dict[str, Any], group_name: str, property_name: str, fallback: str
) -> str:
    value = property_value(item, group_name, property_name, fallback)
    return value if isinstance(value, str) else fallback


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _round(value: float) -> float:
    return float(f"{value:.6f}")
