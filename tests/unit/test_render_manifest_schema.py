"""Round-trip tests for the editor render manifest schema."""

from __future__ import annotations

from uuid import uuid4

from kuvox_ai.schemas import Plan, TrimOperation, VideoRenderManifest


def test_render_manifest_round_trips_camel_case_payload() -> None:
    payload = {
        "schemaVersion": 1,
        "projectId": "project-1",
        "settings": {
            "preset": "h264-1080p",
            "format": "mp4",
            "resolution": "1920x1080",
            "width": 1920,
            "height": 1080,
            "frameRate": 30,
            "quality": "standard",
            "destinationLabel": "Project h264-1080p",
        },
        "durationSeconds": 12.5,
        "mediaSources": [
            {
                "mediaId": "media-video",
                "kind": "video",
                "name": "Video.mp4",
                "durationSeconds": 12.5,
                "width": 1920,
                "height": 1080,
                "mimeType": "video/mp4",
                "canonical": {
                    "variant": "canonical",
                    "url": "/bff/media/media-video/object/canonical?v=media%2Fvideo%2Fcanonical",
                    "storageKey": "media/video/canonical",
                },
            },
            {
                "mediaId": "media-audio",
                "kind": "audio",
                "name": "Audio.wav",
                "durationSeconds": 10,
                "canonical": {
                    "variant": "canonical",
                    "url": "/bff/media/media-audio/object/canonical?v=media%2Faudio%2Fcanonical",
                    "storageKey": "media/audio/canonical",
                },
            },
        ],
        "visualItems": [
            {
                "itemId": "clip-1",
                "trackId": "v1",
                "type": "video",
                "mediaId": "media-video",
                "shotId": "media-video:shot:000001",
                "timelineStart": 0,
                "duration": 8,
                "sourceIn": 1,
                "sourceOut": 9,
                "speed": 1,
                "layerOrder": 0,
                "transform": {"x": 0, "y": 0, "scaleX": 1, "scaleY": 1, "rotation": 0},
                "crop": {"top": 0, "right": 0, "bottom": 0, "left": 0},
                "opacity": 1,
            },
            {
                "itemId": "image-1",
                "trackId": "o1",
                "type": "image",
                "mediaId": "media-video",
                "timelineStart": 8,
                "duration": 4.5,
                "layerOrder": 2,
                "transform": {"x": 10, "y": 20, "scaleX": 1, "scaleY": 1, "rotation": 0},
                "opacity": 0.8,
            },
        ],
        "audioItems": [
            {
                "itemId": "audio-1",
                "trackId": "a1",
                "mediaId": "media-audio",
                "timelineStart": 0,
                "duration": 10,
                "sourceIn": 0,
                "sourceOut": 10,
                "speed": 1,
                "volume": 0.75,
                "muted": False,
                "fades": {"fadeInDuration": 0.5, "fadeOutDuration": 1},
                "layerOrder": 1,
            }
        ],
        "textOverlays": [
            {
                "itemId": "text-1",
                "trackId": "t1",
                "text": "Title",
                "timelineStart": 1,
                "duration": 3,
                "style": {
                    "fontFamily": "Inter",
                    "fontSize": 48,
                    "color": "#ffffff",
                    "fontWeight": "semibold",
                    "textAlign": "center",
                },
                "transform": {"x": 0, "y": 320, "scaleX": 1, "scaleY": 1, "rotation": 0},
                "opacity": 1,
                "layerOrder": 10,
            }
        ],
    }

    manifest = VideoRenderManifest.model_validate(payload)
    assert manifest.project_id == "project-1"
    assert manifest.media_sources[0].media_id == "media-video"
    assert manifest.visual_items[0].shot_id == "media-video:shot:000001"
    assert manifest.visual_items[1].shot_id is None
    assert manifest.audio_items[0].fades.fade_out_duration == 1
    assert manifest.text_overlays[0].style.font_family == "Inter"

    dumped = manifest.model_dump(by_alias=True)
    assert dumped["schemaVersion"] == 1
    assert dumped["settings"]["frameRate"] == 30
    assert dumped["mediaSources"][0]["mediaId"] == "media-video"
    assert dumped["visualItems"][0]["shotId"] == "media-video:shot:000001"
    assert "shotId" in dumped["visualItems"][1]
    assert dumped["visualItems"][0]["stackOrder"] == 0
    assert dumped["textOverlays"][0]["stackOrder"] == 1


def test_render_manifest_v2_round_trips_animation_and_required_stack_order() -> None:
    payload = {
        "schemaVersion": 2,
        "projectId": "project-2",
        "settings": {
            "preset": "h264-720p",
            "format": "mp4",
            "resolution": "1280x720",
            "width": 320,
            "height": 180,
            "frameRate": 24,
            "quality": "draft",
            "destinationLabel": "Animated",
        },
        "durationSeconds": 2,
        "mediaSources": [
            {
                "mediaId": "image",
                "kind": "image",
                "name": "image.png",
                "width": 100,
                "height": 200,
                "canonical": {"variant": "canonical", "url": "", "storageKey": "image.png"},
            }
        ],
        "visualItems": [
            {
                "itemId": "image-1",
                "trackId": "v1",
                "type": "image",
                "mediaId": "image",
                "timelineStart": 0,
                "duration": 2,
                "layerOrder": 0,
                "stackOrder": 0,
                "transform": {"x": 0, "y": 0, "scaleX": 1, "scaleY": 1, "rotation": 0},
                "crop": {"top": 0, "right": 0, "bottom": 0, "left": 0},
                "opacity": 1,
                "animation": {
                    "transform": {
                        "x": {
                            "keyframes": [
                                {"time": 0, "value": 0},
                                {"time": 2, "value": 100, "easing": [0.42, 0, 0.58, 1]},
                            ]
                        }
                    }
                },
            }
        ],
        "audioItems": [],
        "textOverlays": [],
    }

    manifest = VideoRenderManifest.model_validate(payload)
    assert manifest.schema_version == 2
    assert manifest.visual_items[0].animation is not None
    assert manifest.visual_items[0].animation.transform is not None
    assert manifest.visual_items[0].animation.transform.x is not None
    assert manifest.visual_items[0].animation.transform.x.keyframes[-1].value == 100


def test_legacy_plan_schema_still_uses_uuid_shot_operations() -> None:
    shot_id = uuid4()
    plan = Plan(operations=[TrimOperation(shot_id=shot_id, start_seconds=0, end_seconds=2)])
    reparsed = Plan.model_validate_json(plan.model_dump_json())
    operation = reparsed.operations[0]
    assert isinstance(operation, TrimOperation)
    assert operation.shot_id == shot_id
